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

PROMPT_VERSION = "deep-lesson-v3-study-ready"


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def lesson_key(topic: str, evidence: list[dict], model: str, base_url: str,
               teaching_language: str = "en") -> str:
    evidence_text = "|".join(
        f"{round(float(e.get('start') or 0), 1)}:{_norm(e.get('text', ''))}" for e in evidence
    )
    raw = f"{PROMPT_VERSION}|{model}|{base_url}|{teaching_language}|{_norm(topic)}|{evidence_text}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]


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
