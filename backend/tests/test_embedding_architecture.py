"""Deterministic tests for the provider-independent embedding architecture."""
from __future__ import annotations

import math
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
import types
from pathlib import Path

import httpx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline import embeddings as E  # noqa: E402

# Keep pristine references: individual tests monkeypatch these module attributes.
_ORIG_REMOTE_CLIENT = E._remote_client
_ORIG_LOCAL_MODEL = E._local_model
_ORIG_SENTENCE_MODEL = E._sentence_model

real = pytest.mark.real_embedding  # actual Embedder, mocked transport, still offline


@pytest.fixture(autouse=True)
def _reset_embedding_module():
    """Keep mocked backends and caches from leaking into other test modules."""
    def reset():
        for client in list(E._clients.values()):
            try:
                client.close()
            except Exception:  # pragma: no cover - defensive for fake clients
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
    reset()
    yield
    reset()


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://endpoint.invalid/v1/embeddings")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("failure", request=request, response=response)

    def json(self):
        return self._payload


class FakeClient:
    """Deterministic client: echoes one vector per submitted text."""

    def __init__(self, error=None, status=200, dim=4, payload=None):
        self.error = error
        self.status = status
        self.dim = dim
        # Optional callable(url, body) -> response payload, used by validation tests.
        self.payload = payload
        self.calls = []
        self.closed = False

    @property
    def is_closed(self):
        return self.closed

    def post(self, url, json=None, headers=None):
        self.calls.append({"url": url, "json": json, "headers": headers or {}})
        if self.error is not None:
            raise self.error
        if self.status >= 400:
            return FakeResponse({}, status=self.status)
        if self.payload is not None:
            return FakeResponse(self.payload(url, json))
        texts = json.get("input") or json.get("prompt") or []
        if isinstance(texts, str):
            texts = [texts]
        return FakeResponse(_openai_payload(texts, self.dim))

    def close(self):
        self.closed = True


def _config(monkeypatch, **overrides):
    values = dict(
        embedding_base_url="", embedding_model="", embedding_api_key="",
        embedding_local_model="", embedding_timeout=5.0, embedding_batch_size=8,
        embedding_cache_dir=Path("unused"),
    )
    values.update(overrides)
    monkeypatch.setattr(E, "settings", types.SimpleNamespace(**values))
    monkeypatch.setattr(E, "_CACHE_DIR", Path(tempfile.mkdtemp(prefix="mivmap-embed-")))
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
    return values


def _openai_payload(texts, dim=4):
    return {"data": [{"index": i, "embedding": [float(i + 1)] + [0.0] * (dim - 1)}
                     for i in range(len(texts))]}


def _client_with_payload(monkeypatch, payload):
    """Return a client whose responses are fully controlled by `payload`."""
    client = FakeClient(payload=lambda url, body: payload)
    monkeypatch.setattr(E, "_client", lambda base: client)
    return client


# ---------------------------------------------------------------------------
# 1â€“2. Remote configuration and model selection
# ---------------------------------------------------------------------------
@real
def test_remote_endpoint_configuration_and_model_selection(monkeypatch):
    cfg = _config(monkeypatch, embedding_base_url="https://endpoint.invalid/v1",
                  embedding_model="test-model", embedding_api_key="test-token")
    client = FakeClient()
    monkeypatch.setattr(E, "_client", lambda base: client)

    vectors = E.Embedder(["one"]).encode(["one"])

    assert E.backend_name() == "remote:test-model"
    assert E.embedding_status()["mode"] == "remote"
    real_call = client.calls[0]
    assert real_call["json"]["model"] == cfg["embedding_model"]
    assert real_call["json"]["input"] == ["one"]
    assert real_call["headers"]["Authorization"] == "Bearer test-token"
    assert vectors.shape == (1, 4)
    assert np.all(np.isfinite(vectors))
    assert math.isclose(float(np.linalg.norm(vectors[0])), 1.0, abs_tol=1e-6)


@real
def test_remote_batch_processing_and_consistent_dimensions(monkeypatch):
    _config(monkeypatch, embedding_base_url="https://endpoint.invalid/v1",
            embedding_model="test-model", embedding_batch_size=2)
    client = FakeClient()
    monkeypatch.setattr(E, "_client", lambda base: client)

    vectors = E.Embedder(["a", "b", "c"]).encode(["a", "b", "c"])
    assert vectors.shape == (3, 4)
    # The first format-negotiation request already carries the first batch: no duplicate probe.
    assert [len(call["json"]["input"]) for call in client.calls] == [2, 1]


@real
def test_remote_requires_model_identifier(monkeypatch):
    _config(monkeypatch, embedding_base_url="https://endpoint.invalid/v1", embedding_model="")
    with pytest.raises(E.EmbeddingBackendError, match="EMBEDDING_MODEL"):
        E.Embedder(["x"]).encode(["x"])


# ---------------------------------------------------------------------------
# 3â€“5. Explicit failures, no silent provider switching
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("failure,fragment", [
    (httpx.ConnectError("down"), "unreachable"),
    (httpx.ConnectTimeout("slow"), "unreachable"),
])
@real
def test_remote_connectivity_failure_is_explicit(monkeypatch, failure, fragment):
    _config(monkeypatch, embedding_base_url="https://endpoint.invalid/v1", embedding_model="m")
    client = FakeClient(error=failure)
    monkeypatch.setattr(E, "_client", lambda base: client)
    with pytest.raises(E.EmbeddingConnectionError, match=fragment):
        E.Embedder(["x"]).encode(["x"])
    assert len(client.calls) == 1, "connection failures must not probe other formats"


@real
def test_remote_authentication_failure_is_explicit(monkeypatch):
    _config(monkeypatch, embedding_base_url="https://endpoint.invalid/v1",
            embedding_model="m", embedding_api_key="bad")
    monkeypatch.setattr(E, "_client", lambda base: FakeClient(status=401))
    with pytest.raises(E.EmbeddingBackendError, match="authentication"):
        E.Embedder(["x"]).encode(["x"])


@real
def test_remote_failure_never_switches_provider(monkeypatch):
    _config(monkeypatch, embedding_base_url="https://endpoint.invalid/v1", embedding_model="m")
    monkeypatch.setattr(E, "_client", lambda base: FakeClient(error=httpx.ConnectError("down")))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("local fallback must not be consulted in remote mode")

    monkeypatch.setattr(E, "_local_model", forbidden)
    monkeypatch.setattr(E, "_sentence_model", forbidden)
    encoder = E.Embedder(["x"])
    assert encoder._mode == "remote"
    with pytest.raises(E.EmbeddingBackendError):
        encoder.encode(["x"])
    assert E.backend_name() == "remote:m"
    assert E.embedding_status()["mode"] == "remote"


# ---------------------------------------------------------------------------
# 6â€“7. Local reuse, discovery, ambiguity and LSA fallback
# ---------------------------------------------------------------------------
@real
def test_local_model_is_reused_without_download(monkeypatch):
    _config(monkeypatch, embedding_local_model="local-model-id")
    created = {}

    class FakeSentenceTransformer:
        def __init__(self, model_id, **kwargs):
            created["model_id"] = model_id
            created["kwargs"] = kwargs

        def encode(self, texts, **kwargs):
            return np.tile(np.linspace(0.1, 0.4, 4, dtype=np.float32), (len(texts), 1))

    fake_module = types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    monkeypatch.setattr(E, "has_module", lambda name: name == "sentence_transformers")

    vectors = E.Embedder(["a", "b"]).encode(["a", "b"])

    assert created["model_id"] == "local-model-id"
    assert created["kwargs"].get("local_files_only") is True
    assert vectors.shape == (2, 4)
    assert E.backend_name() == "local:local-model-id"


def test_local_discovery_is_unambiguous_and_never_guesses(monkeypatch):
    _config(monkeypatch)
    monkeypatch.setattr(E, "has_module", lambda name: True)
    monkeypatch.setattr(E, "_cached_sentence_models", lambda: ["sentence-transformers/one"])
    E._local_model.cache_clear()
    assert E._local_model() == "sentence-transformers/one"

    E._local_model.cache_clear()
    monkeypatch.setattr(E, "_cached_sentence_models", lambda: ["a", "b"])
    assert E._local_model() is None
    assert E.backend_name() == "tfidf-lsa"


def test_remote_model_id_is_never_used_as_local_model(monkeypatch):
    _config(monkeypatch, embedding_local_model="remote-style:tag")
    assert E._local_model() is None
    assert E.backend_name() == "tfidf-lsa"


@real
def test_tfidf_lsa_fallback_is_finite_normalized(monkeypatch):
    _config(monkeypatch)
    monkeypatch.setattr(E, "has_module", lambda name: False)
    encoder = E.Embedder(["alpha beta gamma", "beta gamma delta", "gamma delta epsilon"])
    vectors = encoder.encode(["alpha beta gamma", "alpha beta delta"])

    assert vectors.shape[0] == 2
    assert np.all(np.isfinite(vectors))
    assert all(math.isclose(float(np.linalg.norm(row)), 1.0, abs_tol=1e-5) for row in vectors)
    assert E.backend_name() == "tfidf-lsa"


# ---------------------------------------------------------------------------
# 8â€“9. Cache isolation and environment separation
# ---------------------------------------------------------------------------
def test_cache_identity_separates_endpoints_and_models(monkeypatch):
    _config(monkeypatch, embedding_base_url="https://a.invalid/v1", embedding_model="model-a")
    key_a = E._cache_key("same text")
    assert E._cache_identity().startswith("remote|")
    assert "model-a" in E._cache_identity()

    _config(monkeypatch, embedding_base_url="https://b.invalid/v1", embedding_model="model-a")
    assert E._cache_key("same text") != key_a

    _config(monkeypatch, embedding_base_url="https://a.invalid/v1", embedding_model="model-b")
    assert E._cache_key("same text") != key_a


def test_cache_hits_avoid_second_backend_call(monkeypatch):
    _config(monkeypatch, embedding_local_model="local-model")
    calls = []

    def encode(batch):
        calls.append(list(batch))
        return np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (len(batch), 1))

    first = E.cached_encode(["a", "b"], encode)
    second = E.cached_encode(["a", "b"], encode)

    assert len(calls) == 1
    np.testing.assert_allclose(first, second)
    assert E.embedding_cache_stats()["hits"] >= 2


# ---------------------------------------------------------------------------
# 10. Deep Mode detail expansion stays grounded
# ---------------------------------------------------------------------------
def test_deep_mode_details_are_grounded_and_expanded():
    from test_pipeline import synthetic_transcript
    from app.pipeline.builder import build_skeleton

    transcript = synthetic_transcript()
    mm = build_skeleton({
        "videoId": "TESTVIDEO01", "title": "Quantum", "transcript": transcript,
        "mode": "deep", "useLLM": False, "allowWhisper": False,
    })
    concepts = [c for s in mm["root"]["children"] for c in s.get("children", [])
                if c.get("type") == "concept"]
    details = [d for c in concepts for d in c.get("children", []) if d.get("type") == "detail"]

    assert details
    assert not any(c.get("collapsed") for c in concepts)
    assert not any(d.get("more") for d in details)
    starts = {round(float(entry["start"]), 2) for entry in transcript}
    for detail in details:
        assert detail.get("source")
        assert any(abs(float(detail.get("start") or -1) - ts) <= 1 for ts in starts)


# ---------------------------------------------------------------------------
# Review remediation: paths, exact cache keys, URLs, dimension safety,
# fallback consistency and strict response validation.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("path", [
    r"C:\models\embed",            # Windows drive, backslash
    "C:/models/embed",             # Windows drive, forward slash
    r"D:embed",                    # Windows drive-relative
    r"\\server\share\model",       # Windows UNC
    "//server/share/model",        # UNC with forward slashes
    "/opt/models/x",               # POSIX absolute
    "./models/x",                  # explicit relative, forward slash
    r".\models\x",                 # explicit relative, backslash
    "../models/x",                 # parent relative, forward slash
    r"..\models\x",                # parent relative, backslash
    "models/embed",                # bare relative, forward slash
    r"models\embed",               # bare relative, backslash
    "sentence-transformers/all-MiniLM-L6-v2",  # hub id is not a remote tag
])
def test_local_path_categories_are_not_mistaken_for_remote(monkeypatch, path):
    _config(monkeypatch)
    assert E._looks_remote_model(path) is False, path
    assert E._is_local_path(path) is True, path


@pytest.mark.parametrize("tag", [
    "some-model:latest", "org/model:tag", "model:tag", ":tag", "host:9000/model",
])
def test_remote_style_tags_are_rejected_as_local(monkeypatch, tag):
    _config(monkeypatch)
    assert E._looks_remote_model(tag) is True, tag


def test_existing_filesystem_path_is_local_and_usable(monkeypatch, tmp_path):
    _config(monkeypatch)
    provider = tmp_path / "provider"
    provider.mkdir()
    monkeypatch.setattr(E, "has_module", lambda name: True)
    assert E._looks_remote_model(str(provider)) is False
    _config(monkeypatch, embedding_local_model=str(provider))
    assert E._local_model() == str(provider)
    assert E._local_model_usable(str(provider)) is True
    assert E.embedding_mode() == "local"
    # A colon-containing identifier with no real path stays a remote tag.
    assert E._looks_remote_model("host:9000/model") is True


def test_nonexistent_hub_id_is_not_reported_as_local(monkeypatch):
    _config(monkeypatch, embedding_local_model="unrecognised/hub-id")
    monkeypatch.setattr(E, "has_module", lambda name: True)
    monkeypatch.setattr(E, "_cached_sentence_models", lambda: [])
    assert E._local_model_usable("unrecognised/hub-id") is False
    assert E.embedding_mode() == "fallback"


def test_cached_and_uncached_org_model_ids(monkeypatch, tmp_path):
    _config(monkeypatch)
    monkeypatch.setattr(E, "has_module", lambda name: True)
    cache = tmp_path / "hub"
    _make_cached_model(cache, "some-org/cached-model")
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "empty-home"))
    assert E._local_model_usable("some-org/cached-model") is True
    assert E._local_model_usable("some-org/uncached-model") is False


def test_relative_filesystem_path_usability(monkeypatch, tmp_path):
    _config(monkeypatch)
    monkeypatch.setattr(E, "has_module", lambda name: True)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "existing-model").mkdir()
    assert E._local_model_usable("existing-model") is True
    assert E._local_model_usable("missing-model") is False
    assert E._local_model_usable("./existing-model") is True
    assert E._local_model_usable("../missing-model") is False


@pytest.mark.parametrize("path", [
    r"C:\models\embed", "C:/models/embed", r"\\server\share\model",
    "/opt/models/x", "./models/x", r".\models\x",
])
def test_explicit_paths_are_not_hub_ids(monkeypatch, path):
    _config(monkeypatch)
    monkeypatch.setattr(E, "has_module", lambda name: True)
    assert E._is_explicit_path(path) is True
    # An explicitly declared but absent path is unusable, never auto-downloaded.
    assert E._local_model_usable(path) is _path_exists_for_test(path)


def _path_exists_for_test(value: str) -> bool:
    try:
        return Path(value).exists()
    except OSError:
        return False


def test_windows_drive_path_configuration_is_preserved(monkeypatch):
    _config(monkeypatch, embedding_local_model=r"C:\models\embed")
    assert E._local_model() == r"C:\models\embed"
    assert E.backend_name() == r"local:C:\models\embed"


def test_exact_text_cache_separation(monkeypatch):
    _config(monkeypatch, embedding_local_model="local-model")
    assert E._cache_key(" x ") != E._cache_key("x")
    assert E._cache_key("hello") != E._cache_key("hello ")

    calls = []

    def encode(batch):
        calls.append(list(batch))
        return np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (len(batch), 1))

    E.cached_encode([" x "], encode)
    E.cached_encode(["x"], encode)
    assert len(calls) == 2


@real
def test_v1_base_url_is_appended_exactly_once(monkeypatch):
    for base in ("https://endpoint.invalid", "https://endpoint.invalid/"):
        _config(monkeypatch, embedding_base_url=base, embedding_model="m")
        client = FakeClient()
        monkeypatch.setattr(E, "_client", lambda b, c=client: c)
        E.Embedder(["x"]).encode(["x"])
        assert all(call["url"] == "https://endpoint.invalid/v1/embeddings" for call in client.calls)

    _config(monkeypatch, embedding_base_url="https://endpoint.invalid/v1", embedding_model="m")
    client = FakeClient()
    monkeypatch.setattr(E, "_client", lambda b, c=client: c)
    E.Embedder(["x"]).encode(["x"])
    assert all(call["url"] == "https://endpoint.invalid/v1/embeddings" for call in client.calls)
    assert not any("v1/v1" in call["url"] for call in client.calls)


def test_cached_dimension_mismatch_is_detected_and_recomputed(monkeypatch):
    _config(monkeypatch, embedding_local_model="local-model")
    calls = []

    def encode(batch):
        calls.append(list(batch))
        return np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (len(batch), 1))

    first = E.cached_encode(["a", "b"], encode)
    assert first.shape == (2, 4)

    # Corrupt one cached entry with an incompatible vector.
    E._store_cached(E._cache_key("a"), np.ones(3, dtype=np.float32))
    repaired = E.cached_encode(["a", "b"], encode)

    assert repaired.shape == (2, 4)
    assert calls[-1] == ["a"]
    assert E.embedding_cache_stats()["invalidated"] >= 1
    np.testing.assert_allclose(repaired[0], first[0])


def test_missing_dimension_record_is_relearned(monkeypatch):
    _config(monkeypatch, embedding_local_model="local-model")
    calls = []

    def encode(batch):
        calls.append(list(batch))
        return np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (len(batch), 1))

    E.cached_encode(["a"], encode)
    identity = E._cache_identity()
    E._backend_dims.pop(identity, None)
    E._dim_path(identity).unlink(missing_ok=True)

    again = E.cached_encode(["a"], encode)
    assert again.shape == (1, 4)
    assert calls[-1] == ["a"]
    assert E._backend_dims[identity] == 4


@real
def test_local_fallback_keeps_one_vector_space_across_calls(monkeypatch):
    _config(monkeypatch)
    monkeypatch.setattr(E, "has_module", lambda name: False)
    corpus = ["alpha beta gamma", "beta gamma delta", "gamma delta epsilon", "epsilon zeta eta"]
    encoder = E.Embedder(corpus)
    assert encoder._mode == "fallback"

    first = encoder.encode(["alpha beta gamma", "beta gamma delta"])
    vectorizer = encoder._vectorizer
    second = encoder.encode(["gamma delta epsilon"])
    third = encoder.encode(["alpha beta gamma"])

    assert encoder._vectorizer is vectorizer
    assert encoder._mode == "fallback"
    assert first.shape[1] == second.shape[1] == third.shape[1]
    np.testing.assert_allclose(first[0], third[0])


@real
def test_local_failure_after_vectors_produced_raises_explicitly(monkeypatch):
    _config(monkeypatch, embedding_local_model="local-model")
    calls = []

    class FlakySentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, texts, **kwargs):
            calls.append(list(texts))
            if len(calls) > 1:
                raise RuntimeError("model crashed")
            return np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (len(texts), 1))

    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        types.SimpleNamespace(SentenceTransformer=FlakySentenceTransformer))
    monkeypatch.setattr(E, "has_module", lambda name: True)

    encoder = E.Embedder(["a", "b"])
    assert encoder.encode(["a"]).shape == (1, 4)
    with pytest.raises(E.EmbeddingBackendError, match="refusing to switch vector spaces"):
        encoder.encode(["b"])
    assert encoder._mode == "local"


@pytest.mark.parametrize("bad_payload", [
    {"data": [{"index": 9, "embedding": [1.0, 0.0, 0.0, 0.0]}]},  # out of range
    {"data": [{"index": True, "embedding": [1.0, 0.0, 0.0, 0.0]}]},  # boolean index
    {"data": [{"index": "0", "embedding": [1.0, 0.0, 0.0, 0.0]}]},  # string index
])
@real
def test_remote_response_rejects_invalid_indices(monkeypatch, bad_payload):
    _config(monkeypatch, embedding_base_url="https://endpoint.invalid/v1", embedding_model="m")
    _client_with_payload(monkeypatch, bad_payload)
    with pytest.raises(E.EmbeddingBackendError, match="unavailable"):
        E.Embedder(["x"]).encode(["x"])


@real
def test_remote_response_rejects_duplicate_indices(monkeypatch):
    _config(monkeypatch, embedding_base_url="https://endpoint.invalid/v1", embedding_model="m")
    _client_with_payload(monkeypatch, {
        "data": [{"index": 0, "embedding": [1.0, 0.0, 0.0, 0.0]},
                 {"index": 0, "embedding": [0.0, 1.0, 0.0, 0.0]}],
    })
    # Two texts, two vectors, but both claim index 0 -> rejected.
    with pytest.raises(E.EmbeddingBackendError, match="unavailable"):
        E.Embedder(["a", "b"]).encode(["a", "b"])


@real
def test_remote_response_rejects_zero_and_nonfinite_vectors(monkeypatch):
    for bad in ([0.0, 0.0, 0.0, 0.0], [1.0, float("nan"), 0.0, 0.0]):
        _config(monkeypatch, embedding_base_url="https://endpoint.invalid/v1", embedding_model="m")
        _client_with_payload(monkeypatch, {"data": [{"index": 0, "embedding": bad}]})
        with pytest.raises(E.EmbeddingBackendError, match="unavailable"):
            E.Embedder(["x"]).encode(["x"])


@real
def test_authentication_failure_stops_endpoint_probing(monkeypatch):
    _config(monkeypatch, embedding_base_url="https://endpoint.invalid/v1",
            embedding_model="m", embedding_api_key="bad")
    client = FakeClient(status=401)
    monkeypatch.setattr(E, "_client", lambda base: client)
    with pytest.raises(E.EmbeddingAuthenticationError, match="authentication"):
        E.Embedder(["x"]).encode(["x"])
    assert len(client.calls) == 1, "no further wire formats should be probed after 401"
    assert E._last_remote_status["status"] == "error"


@real
@pytest.mark.parametrize("status", [500, 502, 503])
def test_server_errors_abort_immediately(monkeypatch, status):
    _config(monkeypatch, embedding_base_url="https://endpoint.invalid/v1", embedding_model="m")
    client = FakeClient(status=status)
    monkeypatch.setattr(E, "_client", lambda base: client)
    with pytest.raises(E.EmbeddingConnectionError, match="server error"):
        E.Embedder(["x"]).encode(["x"])
    assert len(client.calls) == 1, "5xx must not trigger four format probes"


@real
def test_incompatible_route_falls_back_to_next_wire_format(monkeypatch):
    _config(monkeypatch, embedding_base_url="https://endpoint.invalid/v1", embedding_model="m")

    class NotFoundClient(FakeClient):
        def post(self, url, json=None, headers=None):
            self.calls.append({"url": url, "json": json, "headers": headers or {}})
            if len(self.calls) == 1:
                return FakeResponse({}, status=404)
            return FakeResponse(_openai_payload(json.get("input") or [json.get("prompt")]))

    client = NotFoundClient()
    monkeypatch.setattr(E, "_client", lambda base: client)
    vectors = E.Embedder(["x"]).encode(["x"])
    assert vectors.shape == (1, 4)
    assert len(client.calls) == 2
    assert client.calls[0]["url"] == "https://endpoint.invalid/v1/embeddings"
    assert client.calls[1]["url"] == "https://endpoint.invalid/embeddings"


@real
def test_malformed_success_is_not_treated_as_inference(monkeypatch):
    _config(monkeypatch, embedding_base_url="https://endpoint.invalid/v1", embedding_model="m")
    client = FakeClient(payload=lambda url, body: {"unexpected": "payload"})
    monkeypatch.setattr(E, "_client", lambda base: client)
    with pytest.raises(E.EmbeddingBackendError, match="unavailable"):
        E.Embedder(["x"]).encode(["x"])
    assert len(client.calls) == len(E._STRATEGIES)


def _make_cached_model(root: Path, model_id: str, marker: str | None = "modules.json") -> None:
    directory = root / ("models--" + model_id.replace("/", "--"))
    snapshot = directory / "snapshots" / "abc123"
    snapshot.mkdir(parents=True, exist_ok=True)
    if marker:
        (snapshot / marker).write_text("{}", encoding="utf-8")


def test_local_discovery_covers_any_org_and_hf_cache_locations(monkeypatch, tmp_path):
    _config(monkeypatch)
    monkeypatch.setattr(E, "has_module", lambda name: True)

    cache = tmp_path / "hub"
    _make_cached_model(cache, "some-org/embed-model")          # non-ST org, valid marker
    _make_cached_model(cache, "sentence-transformers/ok", "config_sentence_transformers.json")
    _make_cached_model(cache, "some-org/not-a-embedder", None)  # unrelated cached model
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    # Empty the default home cache root so only the configured roots are visible.
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "empty-home"))

    models = E._cached_sentence_models()
    assert "some-org/embed-model" in models
    assert "sentence-transformers/ok" in models
    assert "some-org/not-a-embedder" not in models
    # Two unambiguous candidates must not be guessed between.
    E._local_model.cache_clear()
    assert E._local_model() is None

    # HF_HOME/hub is honoured as another supported cache root.
    hf_home = tmp_path / "hf-home"
    other = hf_home / "hub"
    _make_cached_model(other, "another-org/only-one")
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(hf_home))
    E._local_model.cache_clear()
    assert E._local_model() == "another-org/only-one"


# ---------------------------------------------------------------------------
# Health accuracy: mode is capability-only and reflects actual local usability
# ---------------------------------------------------------------------------
def test_embedding_mode_falls_back_when_local_model_unavailable(monkeypatch):
    _config(monkeypatch, embedding_local_model="missing-local-model")
    monkeypatch.setattr(E, "has_module", lambda name: True)
    assert E.embedding_mode() == "fallback"
    assert E.embedding_status() == {"mode": "fallback"}
    assert "model" not in E.embedding_status()


def test_embedding_mode_falls_back_when_sentence_transformers_missing(monkeypatch):
    _config(monkeypatch, embedding_local_model="local-model-id")
    monkeypatch.setattr(E, "has_module", lambda name: False)
    assert E.embedding_mode() == "fallback"


@real
def test_embedding_mode_reflects_runtime_local_fallback(monkeypatch):
    _config(monkeypatch, embedding_local_model="local-model-id")
    monkeypatch.setattr(E, "has_module", lambda name: True)

    class BrokenSentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, *_args, **_kwargs):
            raise RuntimeError("model unavailable")

    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        types.SimpleNamespace(SentenceTransformer=BrokenSentenceTransformer))
    encoder = E.Embedder(["alpha beta", "gamma delta"])
    vectors = encoder.encode(["alpha beta"])
    assert vectors.shape[0] == 1
    assert encoder.mode == "fallback"
    assert E.embedding_mode() == "fallback"
    assert E.embedding_runtime()["observed_fallbacks"] == 1


# ---------------------------------------------------------------------------
# Runtime mode accuracy: configured capability, per-instance truth,
# thread-safe observed fallback
# ---------------------------------------------------------------------------
@real
def test_runtime_mode_local_success_and_fresh_instance(monkeypatch, tmp_path):
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    _config(monkeypatch, embedding_local_model=str(model_dir))
    monkeypatch.setattr(E, "has_module", lambda name: True)

    class WorkingST:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, texts, **kwargs):
            return np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (len(texts), 1))

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=WorkingST))
    assert E.embedding_runtime() == {"configured": "local", "observed_fallbacks": 0}
    first = E.Embedder(["alpha"])
    first.encode(["alpha"])
    assert first.mode == "local"
    assert E.embedding_mode() == "local"

    second = E.Embedder(["beta"])
    assert second.mode == "local"
    assert E.embedding_mode() == "local"


@real
def test_runtime_mode_fallback_does_not_contaminate_next_instance(monkeypatch):
    _config(monkeypatch, embedding_local_model="local-model-id")
    monkeypatch.setattr(E, "has_module", lambda name: True)

    class BrokenST:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, *_args, **_kwargs):
            raise RuntimeError("model unavailable")

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=BrokenST))
    first = E.Embedder(["alpha"])
    first.encode(["alpha"])
    assert E.embedding_mode() == "fallback"

    class WorkingST:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, texts, **kwargs):
            return np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (len(texts), 1))

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=WorkingST))
    _ORIG_SENTENCE_MODEL.cache_clear()
    second = E.Embedder(["beta"])
    second.encode(["beta"])
    # Per-instance truth is local; health stays conservative because a fallback
    # was observed in this process. It never claims local is working globally.
    assert second.mode == "local"
    assert E.embedding_runtime()["observed_fallbacks"] == 1
    assert E.embedding_mode() == "fallback"
    assert "model" not in E.embedding_status()


@real
def test_health_reports_runtime_mode_without_secrets(monkeypatch):
    _config(monkeypatch, embedding_base_url="https://endpoint.invalid/v1",
            embedding_model="env-model", embedding_api_key="private-key")
    E.Embedder(["alpha"])
    status = E.embedding_status()
    assert E.configured_embedding_mode() == "remote"
    assert E.embedding_mode() == "remote"
    assert status["mode"] == "remote"
    assert "model" not in status
    assert "base_url" not in status
    assert "api_key" not in status
    assert "private-key" not in repr(status)


@real
def test_concurrent_local_success_and_fallback_health(monkeypatch, tmp_path):
    """One local instance falls back while another succeeds; health stays truthful."""
    model_a = tmp_path / "model-a"
    model_b = tmp_path / "model-b"
    model_a.mkdir()
    model_b.mkdir()
    _config(monkeypatch, embedding_local_model=str(model_a))
    monkeypatch.setattr(E, "has_module", lambda name: True)
    monkeypatch.setattr(E, "_cache_identity", lambda: "")  # isolate the two runtimes

    class WorkingST:
        def __init__(self, model_id, **kwargs):
            self.model_id = model_id

        def encode(self, texts, **kwargs):
            return np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (len(texts), 1))

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=WorkingST))
    original_encode_local = E._encode_local

    def selective_encode(texts, model_id):
        if model_id == str(model_b):
            raise RuntimeError("broken runtime")
        return original_encode_local(texts, model_id)

    monkeypatch.setattr(E, "_encode_local", selective_encode)

    good = E.Embedder(["alpha beta", "gamma delta"])         # model-a, will succeed
    _config(monkeypatch, embedding_local_model=str(model_b))
    broken = E.Embedder(["alpha beta", "gamma delta"])       # model-b, will fall back

    assert good.mode == "local"
    assert broken.mode == "local"
    assert E.embedding_mode() == "local"  # no fallback observed yet

    with ThreadPoolExecutor(max_workers=2) as pool:
        good_future = pool.submit(good.encode, ["alpha beta"])
        broken_future = pool.submit(broken.encode, ["alpha beta"])
        good_vectors = good_future.result()
        broken_vectors = broken_future.result()
        during = E.embedding_mode()

    for vectors in (good_vectors, broken_vectors):
        assert vectors.shape[0] == 1
        assert np.all(np.isfinite(vectors))
        assert math.isclose(float(np.linalg.norm(vectors[0])), 1.0, abs_tol=1e-5)

    assert good.mode == "local"
    assert broken.mode == "fallback"
    assert during == "fallback", "health must not claim local is working while a runtime fell back"
    assert E.embedding_mode() == "fallback"
    assert E.embedding_runtime()["observed_fallbacks"] == 1
    np.testing.assert_allclose(good_vectors, good.encode(["alpha beta"]))
