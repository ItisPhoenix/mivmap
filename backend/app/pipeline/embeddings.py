"""Provider-independent sentence embeddings.

Selection is driven entirely by the environment:

1. ``EMBEDDING_BASE_URL`` set -> remote mode. ``EMBEDDING_MODEL`` is the model
   identifier understood by that endpoint and ``EMBEDDING_API_KEY`` is optional.
   ``EMBEDDING_BASE_URL`` is the *service root*; a trailing ``/v1`` version
   prefix is accepted and normalised so endpoint paths are appended exactly
   once. Endpoints that accept a JSON ``input`` array or a single ``prompt``
   are supported; no host, vendor or model is assumed. Remote failures are
   explicit and never switch providers.
2. No endpoint -> local mode. ``EMBEDDING_LOCAL_MODEL`` may name an installed
   model id/path (including Windows drive paths); otherwise an unambiguous,
   already-cached compatible model is discovered. Nothing is downloaded.
3. No usable local model -> a corpus-fitted TF-IDF + truncated SVD (LSA) space.

Every vector is finite, L2-normalised and comparable across calls. Remote and
local vectors are cached on disk under a provider/model identity and a
dimension record per identity prevents stale entries from poisoning a batch.
The corpus-dependent fallback is never cached.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import numbers
import os
import re
import tempfile
import threading
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from ..config import has_module, settings  # `has_module` is part of this module's public surface

log = logging.getLogger("mivmap.embeddings")
_lock = threading.Lock()

# Bump when the cache key layout changes so entries from an older version are
# never read back with assumptions that no longer hold.
_CACHE_VERSION = "v2"

# Endpoint paths relative to the *service root* (the normalised base URL).
# Names are wire-format descriptors, not provider names.
_STRATEGIES: tuple[tuple[str, str, str, bool], ...] = (
    ("json-input-batch", "/v1/embeddings", "input", False),
    ("json-input-batch-alt", "/embeddings", "input", False),
    ("array-input-batch", "/api/embed", "input", False),
    ("single-text-prompt", "/api/embeddings", "prompt", True),
)
_clients: dict[str, httpx.Client] = {}
_last_remote_status: dict[str, Any] = {"status": "not-used", "strategy": None}
# Thread-safe observation of runtime fallbacks. Health reports configured
# capability plus whether *any* local runtime has fallen back; it never reports
# one Embedder's mode as the application-wide mode.
_runtime_lock = threading.Lock()
_runtime_fallbacks = 0


class EmbeddingBackendError(RuntimeError):
    """A configured embedding backend failed; no silent substitute is used."""


class EmbeddingAuthenticationError(EmbeddingBackendError):
    """The remote endpoint rejected the credentials; probing stops immediately."""


class EmbeddingConnectionError(EmbeddingBackendError):
    """Transport failure, timeout or server error: never retry another format."""


class EmbeddingRouteError(EmbeddingBackendError):
    """Response indicates an incompatible route/request format; try the next one."""


# ---------------------------------------------------------------------------
# Vector hygiene
# ---------------------------------------------------------------------------
def _normalize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat[None, :]
    if not np.all(np.isfinite(mat)):
        raise EmbeddingBackendError("embedding backend returned non-finite values")
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise EmbeddingBackendError("embedding backend returned a zero vector")
    return mat / norms


def _unit_vector(value: Any) -> np.ndarray:
    """A single finite, L2-normalised 1-D vector for cache storage."""
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise EmbeddingBackendError("embedding backend returned non-finite values")
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        raise EmbeddingBackendError("embedding backend returned a zero vector")
    return arr / norm


def _vector(value: Any) -> list[float] | None:
    """A finite numeric 1-D vector, or None if the value is malformed."""
    if not isinstance(value, (list, tuple)):
        return None
    out: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, numbers.Real):
            return None
        number = float(item)
        if not math.isfinite(number):
            return None
        out.append(number)
    return out or None


def _nonzero(vec: list[float]) -> bool:
    norm = float(np.linalg.norm(np.asarray(vec, dtype=np.float32)))
    return math.isfinite(norm) and norm > 0.0


def _extract_vectors(data: Any, expected: int) -> list[list[float]] | None:
    """Validate and return `expected` ordered vectors from a JSON response.

    Rejects malformed payloads, duplicate/out-of-range/non-integer indices,
    zero/non-finite vectors and inconsistent dimensions. Returns None only for
    an unusable response so endpoint negotiation never mistakes one for success.
    """
    if expected < 1:
        return []
    if isinstance(data, dict):
        if isinstance(data.get("data"), list):
            items: list[Any] = data["data"]
        elif isinstance(data.get("embeddings"), list):
            items = data["embeddings"]
        elif "embedding" in data or "vector" in data:
            items = [data]
        else:
            return None
    elif isinstance(data, list):
        items = data
    else:
        return None
    if len(items) != expected:
        return None

    indexed: list[tuple[int, list[float]]] = []
    explicit = False
    for pos, item in enumerate(items):
        if isinstance(item, dict):
            value = item.get("embedding", item.get("vector"))
            raw_index = item.get("index")
            if raw_index is not None:
                explicit = True
                if isinstance(raw_index, bool) or not isinstance(raw_index, numbers.Integral):
                    return None
                index = int(raw_index)
                if index < 0 or index >= expected:
                    return None
            else:
                index = pos
        else:
            value = item
            index = pos
        vec = _vector(value)
        if vec is None or not _nonzero(vec):
            return None
        indexed.append((index, vec))

    if explicit:
        indices = [i for i, _ in indexed]
        if len(set(indices)) != len(indices) or set(indices) != set(range(expected)):
            return None
        indexed.sort(key=lambda pair: pair[0])

    if len({len(vec) for _, vec in indexed}) != 1:
        return None
    return [vec for _, vec in indexed]


# ---------------------------------------------------------------------------
# Remote mode (endpoint is the only source of truth)
# ---------------------------------------------------------------------------
def _normalize_base(base: str) -> str:
    """Canonical service root; an optional trailing /v1 is removed once."""
    base = (base or "").strip().rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base.rstrip("/")


def _client(base: str) -> httpx.Client:
    with _lock:
        client = _clients.get(base)
        if client is None or client.is_closed:
            # Use the configured timeout for every phase; no hidden hard-coded connect timeout.
            client = httpx.Client(timeout=httpx.Timeout(settings.embedding_timeout))
            _clients[base] = client
        return client


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if settings.embedding_api_key:
        headers["Authorization"] = f"Bearer {settings.embedding_api_key}"
    return headers


def _remote_config() -> tuple[str, str]:
    base = _normalize_base(settings.embedding_base_url)
    model = (settings.embedding_model or "").strip()
    if not base:
        raise EmbeddingBackendError("no remote embedding endpoint configured")
    if not model:
        raise EmbeddingBackendError("EMBEDDING_MODEL is required when a remote endpoint is configured")
    return base, model


class RemoteEmbeddings:
    """Speaks the endpoint's embeddings API; never selects or downloads a model."""

    def __init__(self) -> None:
        self.base, self.model = _remote_config()
        self._strategy: tuple[str, str, str, bool] | None = None

    def _request(self, strategy: tuple[str, str, str, bool], texts: list[str]) -> Any:
        _name, path, field, single = strategy
        payload: dict[str, Any] = {"model": self.model}
        payload[field] = texts[0] if single else texts
        url = self.base + path
        try:
            response = _client(self.base).post(url, json=payload, headers=_headers())
        except (httpx.NetworkError, httpx.TimeoutException) as exc:
            # Connection/DNS/read failures are unchanged by picking another wire format.
            raise EmbeddingConnectionError(f"remote embedding endpoint unreachable: {type(exc).__name__}") from exc
        except httpx.HTTPError as exc:
            raise EmbeddingConnectionError(f"remote embedding endpoint request failed: {type(exc).__name__}") from exc

        status = response.status_code
        if status in (401, 403):
            raise EmbeddingAuthenticationError("remote embedding endpoint rejected request (authentication)")
        if status in (400, 404, 405):
            # The route exists but this request shape is not the one it expects.
            raise EmbeddingRouteError(f"remote embedding endpoint rejected request format (HTTP {status})")
        if status >= 500:
            raise EmbeddingConnectionError(f"remote embedding endpoint server error (HTTP {status})")
        if status >= 400:
            raise EmbeddingBackendError(f"remote embedding endpoint rejected request (HTTP {status})")
        try:
            data = response.json()
        except ValueError as exc:
            raise EmbeddingRouteError("remote embedding endpoint returned a non-JSON response") from exc
        if not isinstance(data, (dict, list)):
            raise EmbeddingRouteError("remote embedding endpoint returned a malformed response")
        return data

    def _negotiate(self, texts: list[str]) -> tuple[tuple[str, str, str, bool], np.ndarray, int]:
        """Select a wire format using the real request; its vectors are reused.

        Authentication and transport/server failures abort immediately. Only an
        incompatible route/request-format response causes the next format to be
        tried, so a dead endpoint never triggers four slow connection attempts.
        """
        failures: list[str] = []
        batch = max(1, settings.embedding_batch_size)
        for strategy in _STRATEGIES:
            single = strategy[3]
            probe = texts[:1] if single else texts[:batch]
            try:
                data = self._request(strategy, probe)
            except EmbeddingRouteError as exc:
                # Incompatible route/request format: the next strategy may fit.
                failures.append(f"{strategy[0]}: {exc}")
                continue
            except (EmbeddingAuthenticationError, EmbeddingConnectionError, EmbeddingBackendError) as exc:
                _last_remote_status.update(status="error", strategy=None)
                raise
            vectors = _extract_vectors(data, len(probe))
            if vectors is None:
                failures.append(f"{strategy[0]}: unusable response")
                continue
            self._strategy = strategy
            _last_remote_status.update(status="ok", strategy=strategy[0])
            log.info("remote embedding endpoint using wire format %s", strategy[0])
            return strategy, np.asarray(vectors, dtype=np.float32), len(probe)
        _last_remote_status.update(status="error", strategy=None)
        raise EmbeddingBackendError("remote embedding endpoint unavailable: " + "; ".join(failures[:4]))

    def encode(self, texts: list[str]) -> np.ndarray:
        first: np.ndarray | None = None
        used = 0
        if self._strategy is None:
            _strategy, first, used = self._negotiate(texts)
        assert self._strategy is not None
        single = self._strategy[3]
        batch = 1 if single else max(1, settings.embedding_batch_size)
        chunks: list[np.ndarray] = []
        dim: int | None = None

        def accept(matrix: np.ndarray, expected: int) -> None:
            nonlocal dim
            if matrix.ndim != 2 or matrix.shape[0] != expected:
                raise EmbeddingBackendError("remote embedding endpoint returned a malformed batch")
            if dim is None:
                dim = matrix.shape[1]
            elif matrix.shape[1] != dim:
                raise EmbeddingBackendError("remote embedding endpoint returned inconsistent dimensions")
            chunks.append(matrix)

        if first is not None:
            accept(first, used)
        remaining = texts[used:]
        for start in range(0, len(remaining), batch):
            part = remaining[start:start + batch]
            vectors = _extract_vectors(self._request(self._strategy, part), len(part))
            if vectors is None:
                raise EmbeddingBackendError("remote embedding endpoint returned an unusable vector batch")
            accept(np.asarray(vectors, dtype=np.float32), len(part))
        if not chunks:
            return np.zeros((0, dim or 1), dtype=np.float32)
        return _normalize(np.concatenate(chunks, axis=0))


@lru_cache(maxsize=1)
def _remote_client() -> RemoteEmbeddings:
    return RemoteEmbeddings()


# ---------------------------------------------------------------------------
# Local mode (reuse only what is already installed/cached)
# ---------------------------------------------------------------------------
_DRIVE_PREFIX = re.compile(r"^[A-Za-z]:")


def _path_exists(value: str) -> bool:
    try:
        return Path(value).exists()
    except OSError:  # pragma: no cover - invalid path on this platform
        return False


def _is_local_path(value: str) -> bool:
    """Recognise every supported filesystem path spelling, in either slash direction."""
    if not value:
        return False
    if _DRIVE_PREFIX.match(value):          # C:\models, C:/models, C:models
        return True
    if value.startswith(("/", "\\")):      # POSIX absolute or Windows UNC
        return True
    if re.match(r"^\.{1,2}[\\/]", value):  # ./models, ../models, .\models, ..\models
        return True
    if ":" in value:                       # a colon is remote unless it really exists
        return _path_exists(value)
    if "/" in value or "\\" in value:      # any relative path with a separator
        return True
    return False


def _is_explicit_path(value: str) -> bool:
    """A declared filesystem path (as opposed to a hub id like ``org/model``)."""
    if not value:
        return False
    if _DRIVE_PREFIX.match(value):          # C:\models, C:/models, C:models
        return True
    if value.startswith(("/", "\\")):      # POSIX absolute or Windows UNC
        return True
    return bool(re.match(r"^\.{1,2}[\\/]", value))  # ./ or ../ in either slash direction


def _looks_remote_model(identifier: str) -> bool:
    """True only for remote-style ``name:tag`` ids, never for a local path.

    Drive paths, UNC paths, POSIX absolute paths and relative paths in either
    slash direction are local. An existing filesystem path is local even when
    it contains a colon; only a non-path ``name:tag`` is a remote identifier.
    """
    value = identifier.strip()
    if not value:
        return True
    if ":" not in value:
        return False
    return not _is_local_path(value)


# A cached model is offered for automatic reuse only when a snapshot carries a
# sentence-transformers marker. A plain `config.json` is not enough: it would
# pick unrelated cached trainers/ASR models.
_ST_MARKERS = ("modules.json", "config_sentence_transformers.json", "sentence_bert_config.json")


def _hf_cache_roots() -> list[Path]:
    roots: list[Path] = []
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        value = os.getenv(name)
        if value and value.strip():
            roots.append(Path(value.strip()))
    hf_home = os.getenv("HF_HOME")
    if hf_home and hf_home.strip():
        roots.append(Path(hf_home.strip()) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    return roots


def _cached_sentence_models() -> list[str]:
    """Discover compatible models already cached by Hugging Face tooling.

    Inspects ``HF_HUB_CACHE``, ``HUGGINGFACE_HUB_CACHE``, ``HF_HOME/hub`` and
    the default ``~/.cache/huggingface/hub``. Any organisation is accepted;
    a model is included only when one of its snapshots contains a
    sentence-transformers marker file. Nothing is fetched, and callers select
    a candidate only when discovery is unambiguous.
    """
    found: set[str] = set()
    for root in _hf_cache_roots():
        if not root.is_dir():
            continue
        try:
            entries = list(root.iterdir())
        except OSError:
            continue
        for entry in entries:
            if not entry.is_dir() or not entry.name.startswith("models--"):
                continue
            parts = entry.name.split("--")
            if len(parts) < 3:
                continue
            model_id = "/".join(parts[1:])
            try:
                snapshots = entry / "snapshots"
                compatible = any(
                    (snapshots / revision / marker).exists()
                    for revision in os.listdir(snapshots)
                    for marker in _ST_MARKERS
                ) if snapshots.is_dir() else False
            except OSError:
                compatible = False
            if compatible:
                found.add(model_id)
    return sorted(found)


@lru_cache(maxsize=1)
def _local_model() -> str | None:
    configured = (settings.embedding_local_model or "").strip()
    if configured:
        if _looks_remote_model(configured):
            log.warning("configured local embedding model %r is a remote-style identifier; ignoring it", configured)
            return None
        return configured
    if not has_module("sentence_transformers"):
        return None
    candidates = _cached_sentence_models()
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        log.warning("multiple cached local embedding models found (%d); not guessing one", len(candidates))
    return None


@lru_cache(maxsize=1)
def _sentence_model(model_id: str):
    from sentence_transformers import SentenceTransformer

    log.info("using locally available embedding model %s", model_id)
    # local_files_only guarantees no weights are downloaded.
    return SentenceTransformer(model_id, local_files_only=True)


# ---------------------------------------------------------------------------
# Durable cache (stateless backends only; identity isolates provider + model)
# ---------------------------------------------------------------------------
_CACHE_DIR = settings.embedding_cache_dir
_MEM: OrderedDict[str, np.ndarray] = OrderedDict()
_MEM_LIMIT = 512
_cache_lock = threading.Lock()
_cache_stats: dict[str, int] = {"hits": 0, "misses": 0, "writes": 0, "invalidated": 0}
_backend_dims: dict[str, int] = {}


def _cache_identity() -> str:
    base = (settings.embedding_base_url or "").strip()
    if base:
        return "remote|" + hashlib.sha256(_normalize_base(base).encode("utf-8")).hexdigest()[:16] + "|" + (settings.embedding_model or "")
    model = _local_model()
    return "local|" + model if model else ""


def _cache_key(text: str) -> str:
    # Hash the exact text submitted to the backend: no trimming or normalising.
    identity = _cache_identity()
    payload = _CACHE_VERSION + "\x00" + identity + "\x00" + text
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_path(key: str) -> Path:
    return _CACHE_DIR / key[:2] / f"{key}.npy"


def _dim_path(identity: str) -> Path:
    return _CACHE_DIR / ("dim-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16] + ".json")


def _load_dim(identity: str) -> int | None:
    if identity in _backend_dims:
        return _backend_dims[identity]
    try:
        dim = int(json.loads(_dim_path(identity).read_text(encoding="utf-8"))["dim"])
        if dim > 0:
            _backend_dims[identity] = dim
            return dim
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def _store_dim(identity: str, dim: int) -> None:
    _backend_dims[identity] = dim
    try:
        path = _dim_path(identity)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"dim": dim}), encoding="utf-8")
    except OSError as exc:
        log.warning("embedding dimension record write failed: %s", type(exc).__name__)


def _load_cached(key: str) -> np.ndarray | None:
    with _cache_lock:
        if key in _MEM:
            _MEM.move_to_end(key)
            _cache_stats["hits"] += 1
            return _MEM[key]
    try:
        vec = _unit_vector(np.load(_cache_path(key), allow_pickle=False))
    except (OSError, ValueError, EmbeddingBackendError):
        with _cache_lock:
            _cache_stats["misses"] += 1
        return None
    with _cache_lock:
        _MEM[key] = vec
        _MEM.move_to_end(key)
        _cache_stats["hits"] += 1
        while len(_MEM) > _MEM_LIMIT:
            _MEM.popitem(last=False)
    return vec


def _drop_cached(key: str) -> None:
    with _cache_lock:
        _MEM.pop(key, None)
        _cache_stats["invalidated"] += 1
    try:
        _cache_path(key).unlink(missing_ok=True)
    except OSError:
        pass


def _store_cached(key: str, vec: np.ndarray) -> None:
    vec = _unit_vector(vec)
    with _cache_lock:
        _MEM[key] = vec
        _MEM.move_to_end(key)
        _cache_stats["writes"] += 1
        while len(_MEM) > _MEM_LIMIT:
            _MEM.popitem(last=False)
    path = _cache_path(key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=f".{key}-", suffix=".npy", dir=str(path.parent))
        os.close(fd)
        np.save(tmp, vec)
        os.replace(tmp, path)
    except OSError as exc:  # cache is an optimisation, never a generation failure
        log.warning("embedding cache write failed: %s", type(exc).__name__)


def _vec_dim(vec: np.ndarray | None) -> int:
    return int(vec.shape[0]) if vec is not None and vec.ndim == 1 and vec.size else -1


def cached_encode(texts: list[str], encode_fn) -> np.ndarray:
    """Batch cache lookup; validate dimensions and recompute stale entries."""
    identity = _cache_identity()
    if not identity:  # LSA is corpus dependent
        return encode_fn(texts)

    keys = [_cache_key(text) for text in texts]
    vectors: list[np.ndarray | None] = [_load_cached(key) for key in keys]
    known_dim = _load_dim(identity)

    # Drop cached vectors that are malformed or disagree with the active backend.
    for i, vec in enumerate(vectors):
        if vec is not None and (_vec_dim(vec) < 1 or (known_dim is not None and _vec_dim(vec) != known_dim)):
            _drop_cached(keys[i])
            vectors[i] = None

    pending = [i for i, vec in enumerate(vectors) if vec is None]
    if known_dim is None and not pending:
        # Without a dimension on record a fully cached batch cannot be trusted.
        # Recompute one text to learn and verify the backend's dimension.
        _drop_cached(keys[0])
        vectors[0] = None
        pending = [0]

    while pending:
        fresh = _normalize(encode_fn([texts[i] for i in pending]))
        if fresh.shape[0] != len(pending):
            raise EmbeddingBackendError("embedding backend returned the wrong number of vectors")
        dim = int(fresh.shape[1])
        if known_dim is not None and dim != known_dim:
            raise EmbeddingBackendError("active embedding backend changed vector dimensions")
        _store_dim(identity, dim)
        for position, index in enumerate(pending):
            vectors[index] = _unit_vector(fresh[position])
            _store_cached(keys[index], fresh[position])
        # A newly learned dimension can invalidate other cached entries.
        stale = [i for i, vec in enumerate(vectors) if vec is not None and _vec_dim(vec) != dim]
        for index in stale:
            _drop_cached(keys[index])
            vectors[index] = None
        pending = stale

    return _normalize(np.stack([v for v in vectors if v is not None], axis=0))


def embedding_cache_stats() -> dict[str, Any]:
    with _cache_lock:
        stats = dict(_cache_stats)
    stats.update(identity=_cache_identity(), memory_entries=len(_MEM), dimensions=dict(_backend_dims))
    return stats


# ---------------------------------------------------------------------------
# Public encoding API
# ---------------------------------------------------------------------------
def _local_model_usable(model_id: str) -> bool:
    """True only when the selected local model can be loaded offline.

    An identifier with ``/`` is *not* automatically a path: ``org/model`` is a
    hub id. Resolution order: an existing filesystem path wins; an explicit
    path that does not exist is unusable; otherwise treat the value as a hub id
    and require it in the recognised SentenceTransformer cache.
    """
    if not has_module("sentence_transformers"):
        return False
    value = model_id.strip()
    if _looks_remote_model(value):
        return False
    if _path_exists(value):
        return True
    if _is_explicit_path(value):
        return False
    return model_id in _cached_sentence_models()


def configured_embedding_mode() -> str:
    """Configured capability only, independent of any runtime instance."""
    if (settings.embedding_base_url or "").strip():
        return "remote"
    model = _local_model()
    return "local" if model and _local_model_usable(model) else "fallback"


def embedding_runtime() -> dict[str, Any]:
    """Explicit runtime semantics for diagnostics (no endpoints/models/paths)."""
    with _runtime_lock:
        fallbacks = _runtime_fallbacks
    return {"configured": configured_embedding_mode(), "observed_fallbacks": fallbacks}


def embedding_mode() -> str:
    """Capability label for public health responses (no model/endpoint).

    Semantics: configured capability, downgraded to ``fallback`` when any local
    Embedder runtime has actually fallen back. A remote endpoint stays remote;
    it never switches providers. Per-instance truth is ``Embedder.mode``.
    """
    with _runtime_lock:
        fallbacks = _runtime_fallbacks
    configured = configured_embedding_mode()
    if configured == "remote":
        return "remote"
    if fallbacks or configured == "fallback":
        return "fallback"
    return "local"


def backend_name() -> str:
    """Detailed label for internal logs/CLI only; never put in public health."""
    base = (settings.embedding_base_url or "").strip()
    if base:
        return f"remote:{settings.embedding_model or 'unconfigured'}"
    model = _local_model()
    return f"local:{model}" if model else "tfidf-lsa"


def embedding_status() -> dict[str, Any]:
    """Capability report; never exposes a model id, endpoint, key or path."""
    status: dict[str, Any] = {"mode": embedding_mode()}
    if (settings.embedding_base_url or "").strip():
        status.update(_last_remote_status)
    return status


def warmup() -> None:
    """Best-effort health warm-up; never changes the selected backend."""
    try:
        if (settings.embedding_base_url or "").strip():
            _remote_client().encode(["embedding warm-up"])
        elif _local_model():
            _sentence_model(_local_model()).encode(["warm-up"], show_progress_bar=False)
    except Exception as exc:  # noqa: BLE001 - health probes must not crash startup
        log.warning("embedding warm-up failed: %s", type(exc).__name__)


def _encode_remote(texts: list[str]) -> np.ndarray:
    return cached_encode(texts, lambda batch: _remote_client().encode(batch))


def _encode_local(texts: list[str], model_id: str) -> np.ndarray:
    def encode(batch: list[str]) -> np.ndarray:
        with _lock:
            vectors = _sentence_model(model_id).encode(
                batch, batch_size=max(1, settings.embedding_batch_size),
                show_progress_bar=False, normalize_embeddings=True,
            )
        return np.asarray(vectors, dtype=np.float32)

    return cached_encode(texts, encode)


class Embedder:
    """Encoder whose vectors are comparable across calls within one backend.

    Remote mode raises on backend failure (never switches providers/local/LSA).
    Local mode uses one corpus-fitted LSA space only when no local model is
    usable; once vectors have been produced it never changes vector spaces.
    """

    def __init__(self, corpus: list[str]):
        self._corpus = list(corpus)
        self._vectorizer = None
        self._svd = None
        self._lsa_ready = False
        self._produced_vectors = False
        if (settings.embedding_base_url or "").strip():
            self._mode = "remote"
            self._model_id = None
        elif _local_model():
            self._mode = "local"
            self._model_id = _local_model()
        else:
            self._mode = "fallback"
            self._model_id = None
            self._ensure_lsa()

    @property
    def mode(self) -> str:
        """This instance's active backend mode (independent of other instances)."""
        return self._mode

    def _fit_lsa(self, corpus: list[str]) -> None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        docs = [d for d in corpus if d and d.strip()] or ["empty"]
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        try:
            mat = self._vectorizer.fit_transform(docs)
        except ValueError:  # empty vocabulary (only stop words)
            self._vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df=1, token_pattern=r"(?u)\b\w+\b")
            mat = self._vectorizer.fit_transform(docs + ["placeholder"])
        n_components = min(128, mat.shape[1] - 1, len(docs) - 1)
        self._svd = None
        if n_components >= 2:
            self._svd = TruncatedSVD(n_components=n_components, n_iter=3, random_state=42).fit(mat)

    def _ensure_lsa(self) -> None:
        """Fit the fallback space once from the original corpus, never per call."""
        if self._lsa_ready:
            return
        self._fit_lsa(self._corpus)
        self._lsa_ready = True

    def _encode_lsa(self, texts: list[str]) -> np.ndarray:
        self._ensure_lsa()
        mat = self._vectorizer.transform(texts)
        dense = self._svd.transform(mat) if self._svd is not None else mat.toarray()
        dense = np.asarray(dense, dtype=np.float32)
        # Empty/stop-word-only text has no features; give it a stable unit vector
        # so the strict normaliser never sees a zero row.
        empty = np.linalg.norm(dense, axis=1) == 0
        if np.any(empty):
            dense[empty, 0] = 1.0
        return _normalize(dense)

    def encode(self, texts: list[str]) -> np.ndarray:
        global _runtime_fallbacks
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        if self._mode == "remote":
            vectors = _encode_remote(texts)
        elif self._mode == "local":
            try:
                vectors = _encode_local(texts, self._model_id)
            except Exception as exc:  # noqa: BLE001 - local inference failure
                if self._produced_vectors:
                    raise EmbeddingBackendError(
                        "local embedding model failed after vectors were produced; "
                        "refusing to switch vector spaces"
                    ) from exc
                log.warning("local embedding model failed (%s); using corpus-fitted TF-IDF/LSA", type(exc).__name__)
                self._mode = "fallback"
                with _runtime_lock:
                    _runtime_fallbacks += 1
                vectors = self._encode_lsa(texts)
        else:
            vectors = self._encode_lsa(texts)
        self._produced_vectors = True
        return vectors


def embed(texts: list[str], corpus: list[str] | None = None) -> np.ndarray:
    """One-shot embedding; `corpus` lets the LSA fallback learn more vocabulary."""
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    return Embedder(list(corpus or []) + list(texts)).encode(texts)


# ---------------------------------------------------------------------------
# Long-span embedding: chunk at sentence boundaries, pool back to one vector
# ---------------------------------------------------------------------------
# One ~6k character span takes ~18 s on the configured remote endpoint; a batch
# of eight of them exceeds a 60 s read timeout. ~900 characters keeps each
# single request comfortably inside that budget without changing providers.
MAX_SPAN_CHARS = 900
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?।])\s+|\n+")


def _validated_chunk_size(max_chars: int) -> int:
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars <= 0:
        raise ValueError("max_chars must be a positive integer")
    return max_chars


def split_span(text: str, max_chars: int = MAX_SPAN_CHARS) -> list[str]:
    """Split a long span at sentence boundaries, never inside a sentence.

    Only a sentence longer than `max_chars` is split further, at word boundaries
    (and as a last resort at character boundaries). Short spans are untouched.
    """
    max_chars = _validated_chunk_size(max_chars)
    value = (text or "").strip()
    if not value:
        return []
    if len(value) <= max_chars:
        return [value]

    sentences = [s.strip() for s in _SENTENCE_BOUNDARY.split(value) if s.strip()] or [value]
    chunks: list[str] = []
    current = ""

    def hard_split(sentence: str) -> None:
        nonlocal current
        if current:
            chunks.append(current)
            current = ""
        piece = ""
        for word in sentence.split(" "):
            while len(word) > max_chars:
                if piece:
                    chunks.append(piece)
                    piece = ""
                chunks.append(word[:max_chars])
                word = word[max_chars:]
            candidate = f"{piece} {word}".strip()
            if len(candidate) <= max_chars:
                piece = candidate
            else:
                if piece:
                    chunks.append(piece)
                piece = word
        if piece:
            chunks.append(piece)

    for sentence in sentences:
        if len(sentence) > max_chars:
            hard_split(sentence)
            continue
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks or [value[:max_chars]]


def encode_long_texts(encoder, texts: list[str], max_chars: int = MAX_SPAN_CHARS) -> np.ndarray:
    """Embed each span as chunks, then pool to one L2-normalised vector per span.

    All chunks are submitted in a single call to the existing encoder, so the
    configured provider selection, batching, durable cache and dimension checks
    are reused unchanged. No parallel requests are introduced.
    """
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    max_chars = _validated_chunk_size(max_chars)
    groups = [split_span(text, max_chars) for text in texts]
    if any(not group for group in groups):
        raise ValueError("encode_long_texts requires non-empty spans")
    chunks = [chunk for group in groups for chunk in group]
    # Deduplicate within this call as well as through the durable cache: repeated
    # sentences across sections must not generate duplicate backend requests.
    unique = list(dict.fromkeys(chunks))
    unique_vectors = np.asarray(encoder.encode(unique), dtype=np.float32)
    if unique_vectors.shape[0] != len(unique):
        raise EmbeddingBackendError("embedding backend returned the wrong number of chunk vectors")
    by_text = {text: unique_vectors[index] for index, text in enumerate(unique)}

    pooled: list[np.ndarray] = []
    for group in groups:
        part = np.stack([by_text[chunk] for chunk in group])
        weights = np.asarray([len(chunk) for chunk in group], dtype=np.float32)
        pooled.append((part * weights[:, None]).sum(axis=0) / weights.sum())
    return _normalize(np.stack(pooled, axis=0))


def cosine_matrix(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    b = a if b is None else b
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)))
    return np.clip(a @ b.T, -1.0, 1.0)
