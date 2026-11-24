"""Deterministic regressions for concept selection and generation efficiency."""
from __future__ import annotations

import json

import pytest

from app.config import settings
from app.pipeline import labels, llm, lesson_cache
from app.pipeline.lessons import validate_lesson, validate_deep_teaching
from test_explanation_recovery import EVIDENCE, TOPIC, VALID_LESSON, _map_with_concept


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(lesson_cache, "cache_dir", lambda: tmp_path)
    monkeypatch.setattr(llm.time, "sleep", lambda *_: None)
    return tmp_path


def cand(cid, text, evidence, **extra):
    node = {"id": cid, "text": text, "topic": text, "type": "concept",
            "evidence": [{"start": 1.0, "text": e} for e in evidence], "children": []}
    node.update(extra)
    return labels.ConceptCand(cid=cid, sid="s1", node=node)


def test_filler_and_recall_candidates_are_rejected_before_generation():
    concepts = {
        "greet": cand("greet", "Computer Networks", ["Hello everyone, I welcome you all to this wonderful course on computer networks."]),
        "gate": cand("gate", "Gate", ["And this course is useful for the students who are preparing for their GATE exams."]),
        "scenario": cand("scenario", "Interesting Scenario", ["And there is an interesting scenario for you."]),
        "lecture": cand("lecture", "Lecture", ["Upon the completion of this lecture, the learner will be able to understand what is computer network."]),
        "result": cand("result", "Result", ["Now if your result is matching with this result, congratulations to you."]),
        "recall": cand("recall", "A communication link can be wired", ["A communication link can be a wired link or a wireless link"], recall=True),
        "legit": cand("legit", "Network Supports", ["Because network supports the way we learn, network supports the way we communicate, network supports the way we work and play."]),
    }
    selected = labels._select_deep_concepts(concepts)
    assert set(selected) == {"legit"}, selected


def test_duplicate_topics_keep_only_best_evidenced_instance():
    concepts = {
        "a": cand("a", "Node", ["A computer network is a set of nodes connected by communication links.", "The keywords are nodes and communication links."]),
        "b": cand("b", "Node", ["A node is a device on the network."]),
    }
    selected = labels._select_deep_concepts(concepts)
    assert set(selected) == {"a"}


def test_legitimate_technical_concepts_survive():
    concepts = {
        "node": cand("node", "Node", ["A computer network is a set of nodes connected by communication links."]),
        "printer": cand("printer", "Printer", ["This printer is a node, desktop computers are nodes, tablet is a node."]),
        "internet": cand("internet", "Internet (network of networks)", ["In reality, internet contains lots of intermediary devices."]),
    }
    assert set(labels._select_deep_concepts(concepts)) == {"node", "printer", "internet"}


def test_retry_feedback_is_bounded_and_does_not_resend_full_lesson(monkeypatch):
    captured = []
    bad = dict(VALID_LESSON)
    bad["explain"] = "Too short."  # non-knowledge-check error: triggers a feedback retry

    def fake(system, user, **kw):
        captured.append(user)
        return bad if len(captured) == 1 else dict(VALID_LESSON)

    monkeypatch.setattr(llm, "chat_json", fake)
    mm = _map_with_concept()
    node = mm["root"]["children"][0]["children"][0]
    accepted, failed, _ = labels._llm_deep_lessons(mm, {node["id"]: labels.ConceptCand(node["id"], "s1", node)})
    assert node["id"] in accepted and not failed
    assert len(captured) == 2
    retry = json.loads(captured[1])[0]["repairFeedback"]
    assert "previousRejectedLessonExcerpt" in retry
    assert "previousRejectedLesson" not in retry
    assert len(captured[1]) < 2500
    assert "definition too short" in retry["previousErrors"]


def test_response_format_rejection_is_recognised_genericly():
    body = '{"error":{"message":"\'response_format.type\' must be \'json_schema\' or \'text\'"}}'
    assert llm._rejected_parameter(body) == "response_format"


def test_chat_json_skips_json_mode_on_known_unsupported_endpoint(monkeypatch):
    url = llm.llm_chat_url() or "https://example.invalid/v1/chat/completions"
    seen = []

    def fake(url_, key, model, messages, max_tokens, temperature, json_mode, attempts=2, max_wait=3.0):
        seen.append(json_mode)
        return '{"ok": 1}'

    monkeypatch.setattr(llm, "_openai_compatible", fake)
    model = settings.llm_model
    monkeypatch.setattr(llm, "_NO_JSON_MODE", set([llm.json_capability_key(url, model)]), raising=False)
    result = llm.chat_json("system", "user", max_tokens=100)
    assert result == {"ok": 1}
    assert seen == [False], "no wasted json_object attempt once the endpoint is known"


def test_accepted_lesson_writes_reinforced_evidence_back_to_node(monkeypatch):
    mm = _map_with_concept()
    mm["transcript"] = [{"start": 1914.0, "end": 1918.0,
                         "text": "the ONT is the terminal installed at your home"}]
    node = mm["root"]["children"][0]["children"][0]
    original = len(node.get("evidence") or [])
    monkeypatch.setattr(llm, "chat_json", lambda *a, **kw: dict(VALID_LESSON))
    accepted, failed, _ = labels._llm_deep_lessons(
        mm, {node["id"]: labels.ConceptCand(node["id"], "s1", node)})
    assert node["id"] in accepted and not failed
    assert len(node["evidence"]) > original
    starts = {float(e.get("start") or 0) for e in node["evidence"]}
    stamps = (accepted[node["id"]].get("lesson_evidence") or {}).get("timestamps") or []
    assert all(float(s) in starts for s in stamps)


def test_quality_gates_are_unchanged():
    ok, errors = validate_lesson(VALID_LESSON, " ".join(e["text"] for e in EVIDENCE), TOPIC, EVIDENCE, "en")
    assert ok, errors
    ok2, errors2 = validate_deep_teaching(VALID_LESSON, TOPIC, "en")
    assert ok2, errors2
    bad = dict(VALID_LESSON)
    bad["explain"] = "Too short."
    ok, errors = validate_lesson(bad, " ".join(e["text"] for e in EVIDENCE), TOPIC, EVIDENCE, "en")
    assert not ok and "definition too short" in errors
