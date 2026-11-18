"""Keep label-cache state from leaking between tests."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from app.config import settings


@pytest.fixture(autouse=True)
def clean_label_cache():
    cache = settings.data_dir / "cache" / "labels"
    if cache.exists():
        shutil.rmtree(cache)
    cache.mkdir(parents=True, exist_ok=True)
    yield
