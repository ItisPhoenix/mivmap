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
from pathlib import Path
from typing import Any

from ..config import settings

PROMPT_VERSION = "deep-lesson-v4-unicode-cache"


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


def load(key: str) -> dict[str, Any] | None:
    try:
        data = json.loads((cache_dir() / f"{key}.json").read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
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
