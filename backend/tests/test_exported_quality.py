"""Validate the final exported Brave-viewer JSON, not just internal fixtures."""
from __future__ import annotations

import json
from pathlib import Path

from app.quality import analyze_map

EXPORT = Path(__file__).parents[2] / "examples" / "http-deep-final.json"


def _load() -> dict:
    return json.loads(EXPORT.read_text(encoding="utf-8"))


def _generated_by(map_: dict) -> dict:
    def walk(node):
        yield node
        for child in node.get("children") or []:
            yield from walk(child)

    out = {}
    for node in walk(map_["root"]):
        key = node.get("generatedBy")
        if key:
            out[key] = out.get(key, 0) + 1
    return out


def test_exported_map_passes_all_quality_checks():
    map_ = _load()
    report = analyze_map(map_)
    assert report.passed, report.failures


def test_exported_map_reports_real_llm_usage():
    map_ = _load()
    assert not map_["meta"]["llm"].startswith("heuristic")
    assert map_["meta"]["labelled"] is True
    generated = _generated_by(map_)
    assert any(key.startswith("llm") for key in generated), generated
