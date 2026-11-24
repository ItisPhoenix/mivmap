"""Deterministic regressions for fresh-lesson knowledge_check repair.

No provider calls: chat_json is mocked. The acceptance run with the real model
is separate and deliberately not part of this file.
"""
from __future__ import annotations

import json

import pytest

from app.pipeline import labels, lesson_cache, llm
from app.pipeline.lessons import knowledge_check_only, validate_lesson
from test_explanation_recovery import EVIDENCE, TOPIC, VALID_LESSON, _map_with_concept


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(lesson_cache, "cache_dir", lambda: tmp_path)
    monkeypatch.setattr(llm.time, "sleep", lambda *_: None)
    return tmp_path


BAD_CHECK = {
    "question": "Where does the carrier line end, and which box owns that endpoint?",
    "answer": "The subscriber termination box belongs inside the premises where the carrier line ends.",
    "explanation": "This follows from the topology described in the lesson.",
}


def _bad_lesson():
    lesson = dict(VALID_LESSON)
    lesson["knowledge_check"] = dict(BAD_CHECK)
    return lesson


def _concept(node):
    return labels.ConceptCand(cid=node["id"], sid="", node=node)


def _first_node(mm):
    return mm["root"]["children"][0]["children"][0]


def test_paraphrase_is_rejected_by_the_existing_support_gate():
    lesson = _bad_lesson()
    ok, errors = validate_lesson(lesson, " ".join(e["text"] for e in EVIDENCE), TOPIC, EVIDENCE, "en")
    assert not ok
    assert "knowledge check answer not supported by the lesson" in errors
    assert knowledge_check_only(errors)


def test_non_knowledge_check_error_never_qualifies_for_targeted_repair():
    assert not knowledge_check_only(["definition too short"])
    assert not knowledge_check_only(["knowledge check too short", "fewer than three mechanism steps"])
    assert knowledge_check_only(["knowledge check too short"])


def test_prompt_requires_exact_wording_reuse(monkeypatch):
    seen = []

    def fake(system, user, **kw):
        seen.append(system)
        return _bad_lesson()

    monkeypatch.setattr(llm, "chat_json", fake)
    mm = _map_with_concept()
    node = _first_node(mm)
    labels._llm_deep_lessons(mm, {node["id"]: _concept(node)})
    assert seen, "generation must call chat_json"
    assert "reuse the exact wording" in seen[0]
    assert "retrieval test" in seen[0]


def test_one_bounded_repair_preserves_the_rest_of_the_lesson(monkeypatch):
    calls = []

    def fake(system, user, **kw):
        calls.append(system)
        if system == labels.KC_REPAIR_SYSTEM:
            return {"knowledge_check": dict(VALID_LESSON["knowledge_check"])}
        return _bad_lesson()

    monkeypatch.setattr(llm, "chat_json", fake)
    mm = _map_with_concept()
    node = _first_node(mm)
    accepted, failed, rejections = labels._llm_deep_lessons(mm, {node["id"]: _concept(node)})

    assert not failed, rejections
    assert node["id"] in accepted
    lesson = accepted[node["id"]]
    assert lesson["explain"].startswith("ONT (Optical Network Terminal)")
    assert lesson["knowledge_check"] == VALID_LESSON["knowledge_check"]
    assert calls.count(labels.KC_REPAIR_SYSTEM) == 1
    assert len(calls) == 2, "one generation call + exactly one targeted repair call"


def test_failed_repair_is_never_marked_successful(monkeypatch):
    def fake(system, user, **kw):
        return _bad_lesson()  # repair also returns a non-compliant check

    monkeypatch.setattr(llm, "chat_json", fake)
    mm = _map_with_concept()
    node = _first_node(mm)
    accepted, failed, rejections = labels._llm_deep_lessons(mm, {node["id"]: _concept(node)})

    assert accepted == {}
    assert failed
    assert rejections[0]["failureKind"] == "validation"
    assert "knowledge_check" not in node or node.get("explain", "").startswith("Explanation unavailable")


def test_nearby_evidence_uses_only_real_relevant_spans():
    mm = _map_with_concept()
    mm["transcript"] = [
        {"start": 1914.0, "end": 1918.0, "text": "the ONT is mounted at the customer premises"},
        {"start": 1920.0, "end": 1922.0, "text": "thanks for watching this channel"},
        {"start": 3000.0, "end": 3002.0, "text": "ONT again much later"},
    ]
    item = {"topic": TOPIC, "evidence": list(EVIDENCE)}
    out = labels._reinforce_evidence(mm, item)
    assert len(out) == 4
    assert out[-1] == {"start": 1914.0, "end": 1918.0, "text": "the ONT is mounted at the customer premises"}
    assert all(e["text"] != "thanks for watching this channel" for e in out)
    assert all(e["start"] < 2000 for e in out)


def test_no_relevant_nearby_evidence_leaves_evidence_unchanged():
    mm = _map_with_concept()
    mm["transcript"] = [
        {"start": 1914.0, "end": 1918.0, "text": "please remember to like and subscribe"},
    ]
    item = {"topic": TOPIC, "evidence": list(EVIDENCE)}
    assert labels._reinforce_evidence(mm, item) == EVIDENCE
