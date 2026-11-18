"""The extension CORS allowlist must stay strict and registrable without source edits."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app import origins


def test_only_exact_extension_origins_are_valid():
    good = "chrome-extension://" + "a" * 32
    assert origins.is_valid_origin(good)
    assert origins.normalize_id("A" * 32) == good
    assert origins.normalize_id(good + "/") == good
    for bad in ["*", "http://localhost", "chrome-extension://*", f"chrome-extension://{'a' * 31}",
                f"https://{'a' * 32}", "", None]:
        assert not origins.is_valid_origin(bad)
        assert origins.normalize_id(bad) is None


def test_user_origins_are_loaded_from_local_file_without_source_edits(tmp_path, monkeypatch):
    custom = "chrome-extension://" + "b" * 32
    allowlist = tmp_path / "allowed_origins.json"
    allowlist.write_text(json.dumps({"origins": [custom, "*", "http://evil.example"]}), encoding="utf-8")
    monkeypatch.setattr(origins, "ALLOWLIST_FILE", allowlist)
    result = origins.allowed_origins()
    assert custom in result
    assert origins.LEGACY_EXTENSION_ORIGIN in result
    assert origins.MIVMAP_EXTENSION_ORIGIN in result
    assert "*" not in result
    assert "http://evil.example" not in result


def test_backend_rejects_unknown_origin_and_allows_builtin():
    from app.main import app

    with TestClient(app) as client:
        bad = client.get("/api/health", headers={"Origin": "chrome-extension://" + "d" * 32})
        assert bad.status_code == 403
        ok = client.get("/api/health", headers={"Origin": origins.MIVMAP_EXTENSION_ORIGIN})
        assert ok.status_code == 200
        assert ok.headers.get("access-control-allow-origin") == origins.MIVMAP_EXTENSION_ORIGIN
