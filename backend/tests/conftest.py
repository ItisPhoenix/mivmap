"""Offline, deterministic test isolation with two independent opt-ins.

Markers (independent of each other):

* ``real_embedding`` - use Mivmap's actual ``Embedder`` implementation instead
  of the deterministic test provider. Settings are still offline and all
  network access is still blocked; architecture tests use mocked transports.
* ``integration`` - genuine provider test: no offline patching and no network
  guard. Skipped unless ``MIVMAP_TEST_REMOTE_EMBEDDINGS`` or
  ``MIVMAP_TEST_LOCAL_MODEL`` is set.

Every ordinary test gets empty embedding settings, a fixed TF-IDF/LSA provider,
a fresh cache directory, reset module state, and a socket-level network guard
that permits only loopback traffic.
"""
from __future__ import annotations

import shutil
import socket
import types
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

NETWORK_BLOCKED = "network access is disabled in the default test suite"


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: genuine provider test, opt-in only, real network permitted")
    config.addinivalue_line("markers", "real_embedding: use the real Embedder implementation with mocked transport")


@lru_cache(maxsize=1)
def _deterministic_embedding_space():
    """One fixed TF-IDF/LSA space shared by every Embedder in the test suite."""
    try:
        from test_pipeline import synthetic_transcript
        docs = [entry["text"] for entry in synthetic_transcript()]
    except Exception:  # pragma: no cover - fallback keeps conftest importable
        docs = ["quantum computing uses qubits and superposition", "entanglement links qubits"]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    matrix = vectorizer.fit_transform(docs)
    components = min(128, matrix.shape[1] - 1, len(docs) - 1)
    svd = TruncatedSVD(n_components=components, n_iter=5, random_state=42).fit(matrix) if components >= 2 else None
    return vectorizer, svd


class _DeterministicEmbedder:
    """Fixed, offline embeddings: identical vectors for identical text anywhere."""

    def __init__(self, corpus: list[str]):
        self._vectorizer, self._svd = _deterministic_embedding_space()

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        dense = self._vectorizer.transform(texts)
        dense = self._svd.transform(dense) if self._svd is not None else dense.toarray()
        dense = np.asarray(dense, dtype=np.float32)
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return dense / norms


_ORIGINAL_CACHE_CLEARERS = None


def _reset_embedding_module(embeddings) -> None:
    global _ORIGINAL_CACHE_CLEARERS
    if _ORIGINAL_CACHE_CLEARERS is None:
        _ORIGINAL_CACHE_CLEARERS = (
            embeddings._remote_client.cache_clear,
            embeddings._local_model.cache_clear,
            embeddings._sentence_model.cache_clear,
        )
    for client in list(embeddings._clients.values()):
        try:
            client.close()
        except Exception:  # pragma: no cover - defensive for fake clients
            pass
    embeddings._clients.clear()
    for clear in _ORIGINAL_CACHE_CLEARERS:
        clear()
    embeddings._MEM.clear()
    embeddings._backend_dims.clear()
    embeddings._cache_stats.update(hits=0, misses=0, writes=0, invalidated=0)
    embeddings._last_remote_status.update(status="not-used", strategy=None)
    if hasattr(embeddings, "_runtime_fallbacks"):
        with embeddings._runtime_lock:
            embeddings._runtime_fallbacks = 0


@pytest.fixture(autouse=True)
def clean_label_cache():
    from app.config import settings
    cache = settings.data_dir / "cache" / "labels"
    if cache.exists():
        shutil.rmtree(cache)
    cache.mkdir(parents=True, exist_ok=True)
    yield


@pytest.fixture(autouse=True)
def offline_embeddings(request, monkeypatch, tmp_path):
    from app.pipeline import embeddings as E

    # Genuine provider integration tests own their settings and network access.
    if request.node.get_closest_marker("integration"):
        _reset_embedding_module(E)
        yield
        _reset_embedding_module(E)
        return

    # Every ordinary test, including `real_embedding` architecture tests, gets
    # empty embedding configuration. No value can come from the private .env.
    monkeypatch.setattr(E, "settings", types.SimpleNamespace(
        embedding_base_url="", embedding_model="", embedding_api_key="",
        embedding_local_model="", embedding_timeout=5.0, embedding_batch_size=8,
        embedding_cache_dir=tmp_path / "embeddings",
    ))
    monkeypatch.setattr(E, "_CACHE_DIR", tmp_path / "embeddings")
    if not request.node.get_closest_marker("real_embedding"):
        monkeypatch.setattr(E, "Embedder", _DeterministicEmbedder)
    _reset_embedding_module(E)
    yield
    _reset_embedding_module(E)


def _is_loopback(address) -> bool:
    if address is None:
        return True
    if isinstance(address, (tuple, list)):
        address = address[0]
    if isinstance(address, bytes):
        address = address.decode("utf-8", "replace")
    host = str(address).lower()
    return host in {"127.0.0.1", "::1", "localhost", "ip6-localhost", ""} or host.startswith("127.")


@pytest.fixture(autouse=True)
def _no_external_network(monkeypatch, request):
    """Block all non-loopback outbound network access in ordinary tests."""
    if request.node.get_closest_marker("integration"):
        yield
        return

    import httpx
    from urllib.parse import urlparse

    def blocked(*_args, **_kwargs):
        raise AssertionError(NETWORK_BLOCKED)

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_sendto = socket.socket.sendto
    real_create_connection = socket.create_connection
    real_getaddrinfo = socket.getaddrinfo
    real_gethostbyname = socket.gethostbyname
    real_gethostbyname_ex = socket.gethostbyname_ex
    real_gethostbyaddr = socket.gethostbyaddr
    real_urlopen = urllib.request.urlopen

    def guarded_connect(self, address, *args, **kwargs):  # TCP and connected UDP
        if not _is_loopback(address):
            raise AssertionError(NETWORK_BLOCKED)
        return real_connect(self, address, *args, **kwargs)

    def guarded_connect_ex(self, address, *args, **kwargs):
        if not _is_loopback(address):
            raise AssertionError(NETWORK_BLOCKED)
        return real_connect_ex(self, address, *args, **kwargs)

    def guarded_sendto(self, data, address, *args, **kwargs):  # unconnected UDP
        if not _is_loopback(address):
            raise AssertionError(NETWORK_BLOCKED)
        return real_sendto(self, data, address, *args, **kwargs)

    def guarded_create_connection(address, *args, **kwargs):
        if not _is_loopback(address):
            raise AssertionError(NETWORK_BLOCKED)
        return real_create_connection(address, *args, **kwargs)

    def guarded_getaddrinfo(host, port, *args, **kwargs):  # DNS resolution
        if not _is_loopback(host):
            raise AssertionError(NETWORK_BLOCKED)
        return real_getaddrinfo(host, port, *args, **kwargs)

    def guarded_gethostbyname(host):
        if not _is_loopback(host):
            raise AssertionError(NETWORK_BLOCKED)
        return real_gethostbyname(host)

    def guarded_gethostbyname_ex(host):
        if not _is_loopback(host):
            raise AssertionError(NETWORK_BLOCKED)
        return real_gethostbyname_ex(host)

    def guarded_gethostbyaddr(host):
        if not _is_loopback(host):
            raise AssertionError(NETWORK_BLOCKED)
        return real_gethostbyaddr(host)

    def guarded_urlopen(url, *args, **kwargs):
        host = urlparse(url).hostname if isinstance(url, str) else None
        if not _is_loopback(host):
            raise AssertionError(NETWORK_BLOCKED)
        return real_urlopen(url, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket.socket, "sendto", guarded_sendto)
    monkeypatch.setattr(socket, "create_connection", guarded_create_connection)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    monkeypatch.setattr(socket, "gethostbyname", guarded_gethostbyname)
    monkeypatch.setattr(socket, "gethostbyname_ex", guarded_gethostbyname_ex)
    monkeypatch.setattr(socket, "gethostbyaddr", guarded_gethostbyaddr)
    monkeypatch.setattr(urllib.request, "urlopen", guarded_urlopen)

    def _request_host(args, kwargs):
        for value in list(args[1:]) + list(kwargs.values()):
            candidate = getattr(value, "url", value)
            if isinstance(candidate, (str, httpx.URL)):
                host = urlparse(str(candidate)).hostname
                if host:
                    return host
        return None

    def guarded_request(original):
        def wrapper(self, *args, **kwargs):
            # MockTransport is an in-process offline test double, not a network.
            if isinstance(getattr(self, "_transport", None), httpx.MockTransport):
                return original(self, *args, **kwargs)
            # Real loopback HTTP(S) is legitimate local test infrastructure.
            if _is_loopback(_request_host(args, kwargs)):
                return original(self, *args, **kwargs)
            raise AssertionError(NETWORK_BLOCKED)
        return wrapper

    for client in (httpx.Client, httpx.AsyncClient):
        for method in ("request", "send"):
            monkeypatch.setattr(client, method, guarded_request(getattr(client, method)))
    yield
