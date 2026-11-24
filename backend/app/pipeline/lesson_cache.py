"""Durable, content-addressed cache for validated Deep Mode lessons.

A lesson is cached by a digest of the concept's topic, transcript evidence, provider
model and prompt version. Validated lessons are never regenerated; only missing or
rejected concepts are retried. Cache writes are atomic so an interrupted run leaves
usable checkpoints.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any

from ..config import settings

PROMPT_VERSION = "deep-lesson-v4-unicode-cache"

# Process-local counters for honest cache-hit reporting. No credentials, no map
# content and no model endpoints are recorded here.
CACHE_STATS = {"hits": 0, "recovered": 0, "misses": 0}
_stats_lock = threading.Lock()


def _count_cache(kind: str) -> None:
    with _stats_lock:
        CACHE_STATS[kind] = CACHE_STATS.get(kind, 0) + 1


def lesson_key(topic: str, evidence: list[dict], model: str, base_url: str,
               teaching_language: str = "en") -> str:
    """Content-addressed key that preserves every Unicode script in the source.

    ASCII-only normalisation collapsed different Devanagari passages onto the
    same key. Serialise the full UTF-8 payload deterministically and hash it
    with SHA-256 so distinct evidence can never collide silently.
    """
    payload = {
        "prompt_version": PROMPT_VERSION,
        "model": str(model or ""),
        "base_url": str(base_url or ""),
        "teaching_language": str(teaching_language or "en"),
        "topic": str(topic or ""),
        "evidence": [
            {
                "start": round(float(e.get("start") or 0), 1),
                "text": str(e.get("text") or "").strip(),
            }
            for e in (evidence or [])
        ],
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def cache_dir() -> Path:
    path = settings.data_dir / "cache" / "lessons"
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_cached_lesson(topic: str, evidence: list[dict],
                       teaching_language: str = "en") -> dict[str, Any] | None:
    """Recover a validated lesson whose cache key no longer matches the provider.

    Keys include the model and base URL, so changing provider configuration can
    orphan perfectly good lessons. This lookup identifies a candidate by topic
    plus grounded transcript timestamps; callers MUST still run the full quality
    gate on the result before teaching it. Nothing here relaxes validation.
    """
    from .text_utils import unicode_norm

    wanted = unicode_norm(str(topic or "")).casefold()
    if not wanted:
        return None
    spans = []
    for e in evidence or []:
        try:
            start = float(e.get("start") or 0)
            end = float(e.get("end") or start)
        except (TypeError, ValueError):
            continue
        spans.append((start, end))
    best: tuple[int, dict[str, Any]] | None = None
    for path in sorted(cache_dir().glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict) or not isinstance(data.get("lesson"), dict):
            continue
        cached_topic = unicode_norm(str(data.get("topic") or "")).casefold()
        if cached_topic != wanted:
            continue
        lesson = data["lesson"]
        lang = str(lesson.get("teaching_language") or "en")
        if teaching_language and lang and lang.split("-")[0] != teaching_language.split("-")[0]:
            continue
        evidence_meta = lesson.get("evidence") or {}
        stamps = evidence_meta.get("timestamps") or evidence_meta.get("starts") or []
        grounded = 0
        for value in stamps:
            try:
                stamp = float(value)
            except (TypeError, ValueError):
                continue
            if any(lo - 2 <= stamp <= hi + 2 for lo, hi in spans):
                grounded += 1
        if not grounded:
            continue
        if best is None or grounded > best[0]:
            best = (grounded, lesson)
    if best is not None:
        _count_cache("recovered")
    return best[1] if best else None


def load(key: str) -> dict[str, Any] | None:
    try:
        data = json.loads((cache_dir() / f"{key}.json").read_text(encoding="utf-8"))
        result = data if isinstance(data, dict) else None
        _count_cache("hits" if result else "misses")
        return result
    except (OSError, ValueError):
        _count_cache("misses")
        return None


def save(key: str, lesson: dict) -> None:
    """Atomic write: a crash during generation never corrupts an existing lesson."""
    path = cache_dir() / f"{key}.json"
    fd, tmp_name = tempfile.mkstemp(prefix=f".{key}-", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(lesson, handle, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
