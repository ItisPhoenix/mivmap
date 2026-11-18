"""Regression checks against the exported HTTP Fundamentals map.

The exported map is the 'before' fixture: it documents the defects we fixed.  The 'after'
map is rebuilt offline from the same transcript, so these tests need no network or API key.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.pipeline.builder import build_skeleton
from app.quality import analyze_map

FIXTURE = Path(__file__).parent / "fixtures" / "http-fundamentals-before.json"


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_exported_fixture_documents_the_defects():
    report = analyze_map(_fixture())
    assert not report.passed
    assert report.checks["concepts_not_transcript_like"]["count"] >= 1
    assert report.checks["http_topic_coverage"]["missing"]
    assert report.checks["detail_timestamp_alignment"]["count"] >= 1


def test_rebuilt_http_map_passes_quality_checks():
    old = _fixture()
    request = {
        "videoId": old["meta"]["videoId"],
        "title": old["meta"].get("originalTitle") or old["meta"].get("title"),
        "transcript": old.get("transcript") or [],
        "mode": "academic",
        "useLLM": False,
        "allowWhisper": False,
    }
    new = build_skeleton(request)
    report = analyze_map(new)
    for key in ("concepts_not_transcript_like", "concepts_not_vague",
                "detail_timestamp_alignment"):
        assert report.checks[key]["count"] == 0 if "count" in report.checks[key] else not report.checks[key]["missing"], (key, report.checks[key])
