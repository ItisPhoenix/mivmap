"""Fresh explanation generation regressions — mocked LLM, no API quota."""
from __future__ import annotations

import json

import pytest

from app.features import explanations
from app.pipeline import labels, lesson_cache, llm
from test_explanation_recovery import VALID_LESSON, _map_with_concept


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(lesson_cache, "cache_dir", lambda: tmp_path)
    return tmp_path


def test_no_provider_never_generates_and_reports_honestly(monkeypatch):
    mm = _map_with_concept()
    monkeypatch.setattr(llm, "fast_available", lambda: False)

    def forbidden(*a, **kw):
        raise AssertionError("LLM must not be called without a configured provider")

    monkeypatch.setattr(labels.llm, "chat_json", forbidden)
    report = explanations.generate_explanations(mm, ["c1"])
    assert report["generated"] == []
    assert "no llm provider" in report["failed"][0]["reason"].lower()
    assert report["provider"] == llm.provider_name()
    node = mm["root"]["children"][0]["children"][0]
    assert not node.get("lesson_available")
    assert explanations.has_explanation(node) is False


def test_mocked_provider_generates_validates_patches_and_caches(monkeypatch, isolated_cache):
    mm = _map_with_concept()
    monkeypatch.setattr(llm, "fast_available", lambda: True)
    monkeypatch.setattr(labels.llm, "fast_available", lambda: True)
    monkeypatch.setattr(labels.llm, "chat_json", lambda *a, **kw: {"c1": dict(VALID_LESSON)})

    report = explanations.generate_explanations(mm, ["c1"])
    assert report["generated"] == ["c1"], report
    node = mm["root"]["children"][0]["children"][0]
    assert node["explain"].startswith("ONT (Optical Network Terminal)")
    assert node["lesson_available"] is True
    assert node["generatedBy"] == "llm-deep"
    assert node["knowledge_check"]["question"]
    assert len(list(isolated_cache.glob("*.json"))) == 1


def test_validation_rejection_is_never_marked_successful(monkeypatch):
    mm = _map_with_concept()
    bad = dict(VALID_LESSON)
    bad["explain"] = "Too short."
    monkeypatch.setattr(llm, "fast_available", lambda: True)
    monkeypatch.setattr(labels.llm, "fast_available", lambda: True)
    monkeypatch.setattr(labels.llm, "chat_json", lambda *a, **kw: {"c1": bad})
    monkeypatch.setattr(llm.time, "sleep", lambda *_: None)

    report = explanations.generate_explanations(mm, ["c1"])
    assert report["generated"] == []
    assert report["failed"]
    node = mm["root"]["children"][0]["children"][0]
    assert node.get("lesson_available") is not True
    assert explanations.has_explanation(node) is False


def test_recall_nodes_are_not_sent_for_concept_generation(monkeypatch):
    mm = _map_with_concept()
    section = mm["root"]["children"][0]
    section["children"].append({
        "id": "recall-1", "type": "concept", "text": "A network is not the internet",
        "recall": True, "qa": True, "question": "Q?", "answer": "A longer answer here.",
        "summary": "", "children": [],
    })
    monkeypatch.setattr(llm, "fast_available", lambda: True)
    monkeypatch.setattr(labels.llm, "chat_json", lambda *a, **kw: pytest.fail("recall must not generate"))

    report = explanations.generate_explanations(mm, ["recall-1"])
    assert report["generated"] == []
    assert report["attempted"] == 0
    assert "already has an explanation or is not a teachable concept" in report["failed"][0]["reason"]


def test_existing_valid_explanation_is_not_regenerated(monkeypatch):
    mm = _map_with_concept(explain="ONT converts the optical fibre signal into a home network connection.")
    monkeypatch.setattr(labels.llm, "chat_json", lambda *a, **kw: pytest.fail("must not regenerate"))
    report = explanations.generate_explanations(mm, ["c1"])
    assert report["generated"] == [] and report["attempted"] == 0
    assert report["skipped"] == ["c1"]


def test_generation_endpoint_returns_map_and_report(monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app

    mm = _map_with_concept()
    monkeypatch.setattr(llm, "fast_available", lambda: False)
    client = TestClient(app)
    response = client.post("/api/explanations/generate", json={"map": mm, "conceptIds": ["c1"]})
    assert response.status_code == 200
    data = response.json()
    assert "map" in data and data["report"]["generated"] == []
    assert json.dumps(data["map"], ensure_ascii=False)
