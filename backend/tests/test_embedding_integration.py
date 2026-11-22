"""Genuine embedding-provider integration tests (explicit opt-in only).

These tests are skipped in the default offline suite. Enable exactly one mode
at a time:

* ``MIVMAP_TEST_REMOTE_EMBEDDINGS=1 pytest tests/test_embedding_integration.py``
  uses the real endpoint/model from the private ``.env`` (never printed).
* ``MIVMAP_TEST_LOCAL_MODEL=<id-or-path> pytest tests/test_embedding_integration.py``
  configures Mivmap's real local path with an already-cached model.

Both tests write only to pytest's temporary directory, so the application's
persistent embedding cache is never read or modified. No credentials or
endpoints are hard-coded here, and nothing is downloaded.
"""
from __future__ import annotations

import math
import os
from dataclasses import replace

import numpy as np
import pytest

from app.pipeline import embeddings as E

pytestmark = pytest.mark.integration  # real providers: no offline patching, real network permitted

_REMOTE = os.getenv("MIVMAP_TEST_REMOTE_EMBEDDINGS", "").strip().lower() in {"1", "true", "yes", "on"}
_LOCAL = os.getenv("MIVMAP_TEST_LOCAL_MODEL", "").strip()

# Pristine references: the local test temporarily replaces _sentence_model.
_ORIG_REMOTE_CLIENT = E._remote_client
_ORIG_LOCAL_MODEL = E._local_model
_ORIG_SENTENCE_MODEL = E._sentence_model


def _clear_embedding_state() -> None:
    for client in list(E._clients.values()):
        try:
            client.close()
        except Exception:  # pragma: no cover
            pass
    E._clients.clear()
    _ORIG_REMOTE_CLIENT.cache_clear()
    _ORIG_LOCAL_MODEL.cache_clear()
    _ORIG_SENTENCE_MODEL.cache_clear()
    E._MEM.clear()
    E._backend_dims.clear()
    E._cache_stats.update(hits=0, misses=0, writes=0, invalidated=0)
    E._last_remote_status.update(status="not-used", strategy=None)
    if hasattr(E, "_runtime_fallbacks"):
        with E._runtime_lock:
            E._runtime_fallbacks = 0


@pytest.fixture(autouse=True)
def _fresh_embedding_state():
    _clear_embedding_state()
    yield
    _clear_embedding_state()


def _assert_vectors(vectors: np.ndarray, expected: int) -> None:
    assert vectors.shape[0] == expected
    assert vectors.ndim == 2 and vectors.shape[1] > 0
    assert np.all(np.isfinite(vectors))
    assert all(math.isclose(float(np.linalg.norm(row)), 1.0, abs_tol=1e-4) for row in vectors)


@pytest.mark.skipif(not _REMOTE, reason="set MIVMAP_TEST_REMOTE_EMBEDDINGS=1 to run real remote inference")
def test_real_remote_inference(monkeypatch, tmp_path):
    """Prove real HTTP inference: negotiation, POSTs and fresh cache writes."""
    from app.config import settings

    cache = tmp_path / "remote-embeddings"
    monkeypatch.setattr(E, "_CACHE_DIR", cache)
    monkeypatch.setattr(E, "settings", replace(settings, embedding_cache_dir=cache))
    _clear_embedding_state()
    assert (E.settings.embedding_base_url or "").strip(), "remote endpoint must come from the environment"
    assert E.embedding_cache_stats()["writes"] == 0, "integration cache must start empty"

    posts: list[str] = []
    real_client_factory = E._client  # capture the real factory, never mock the transport

    class CountingClient:
        """Delegate every call to the real client; only record observed POST URLs."""

        def __init__(self, base):
            self._inner = real_client_factory(base)

        @property
        def is_closed(self):
            return self._inner.is_closed

        def post(self, *args, **kwargs):
            posts.append(args[0] if args else kwargs.get("url", ""))
            return self._inner.post(*args, **kwargs)

        def close(self):
            self._inner.close()

    monkeypatch.setattr(E, "_client", CountingClient)

    texts = ["quantum computing", "entanglement"]
    vectors = E.Embedder(texts).encode(texts)

    assert posts, "no real HTTP POST was observed"
    assert E._last_remote_status["status"] == "ok", "endpoint negotiation did not succeed"
    assert E._remote_client()._strategy is not None, "no wire format was negotiated"
    assert E.embedding_cache_stats()["writes"] >= len(texts), "fresh inference was not cached"
    _assert_vectors(vectors, len(texts))
    assert E.embedding_mode() == "remote"


@pytest.mark.skipif(not _LOCAL, reason="set MIVMAP_TEST_LOCAL_MODEL to a cached model id/path to run local inference")
def test_real_local_inference(monkeypatch, tmp_path):
    """Exercise Mivmap's real local Embedder path and prove ST encode ran."""
    from app.config import settings
    from sentence_transformers import SentenceTransformer

    cache = tmp_path / "local-embeddings"
    monkeypatch.setattr(E, "_CACHE_DIR", cache)
    monkeypatch.setattr(E, "settings", replace(
        settings,
        embedding_base_url="", embedding_model="", embedding_api_key="",
        embedding_local_model=_LOCAL, embedding_cache_dir=cache,
    ))
    _clear_embedding_state()
    assert not (E.settings.embedding_base_url or "").strip()
    assert E.embedding_cache_stats()["writes"] == 0, "integration cache must start empty"

    loaded: list[object] = []
    encoded: list[object] = []
    real_sentence_model = E._sentence_model  # real lru_cached loader

    def spy_sentence_model(model_id):
        model = real_sentence_model(model_id)  # genuine load, local_files_only=True
        loaded.append(model)
        original_encode = model.encode

        def counting_encode(*args, **kwargs):
            encoded.append(args[0] if args else kwargs.get("texts"))
            return original_encode(*args, **kwargs)

        model.encode = counting_encode
        return model

    monkeypatch.setattr(E, "_sentence_model", spy_sentence_model)

    texts = ["quantum computing", "entanglement"]
    vectors = E.Embedder(texts).encode(texts)

    assert loaded and isinstance(loaded[0], SentenceTransformer), "real SentenceTransformer was not loaded"
    assert encoded and any(item is not None for item in encoded), "SentenceTransformer.encode was not executed"
    assert E.embedding_cache_stats()["writes"] >= len(texts), "fresh local inference was not cached"
    _assert_vectors(vectors, len(texts))
    assert E.embedding_mode() == "local"
