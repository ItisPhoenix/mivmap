"""Focused regressions for the final independent review corrections (D1-D7).

Deterministic and offline: all provider calls use httpx.MockTransport, all lesson
writes use a temporary cache, and no live LLM quota is consumed.
"""
from __future__ import annotations

import json
from dataclasses import replace

import httpx
import pytest

from app.config import settings
from app.features import study
from app.pipeline import labels, llm, lesson_cache
from test_explanation_recovery import EVIDENCE, TOPIC, VALID_LESSON, _map_with_concept


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(lesson_cache, "cache_dir", lambda: tmp_path)
    monkeypatch.setattr(llm.time, "sleep", lambda *_: None)
    state = {"no_json": set(llm._NO_JSON_MODE), "stats": dict(llm.STATS),
             "cache_stats": dict(lesson_cache.CACHE_STATS)}
    # Start from a known capability state so assertions are order-independent.
    llm._NO_JSON_MODE.clear()
    yield tmp_path
    llm._NO_JSON_MODE.clear()
    llm._NO_JSON_MODE.update(state["no_json"])
    llm.STATS.clear()
    llm.STATS.update(state["stats"])
    lesson_cache.CACHE_STATS.clear()
    lesson_cache.CACHE_STATS.update(state["cache_stats"])


def cand(cid, text, evidence, **extra):
    node = {"id": cid, "text": text, "topic": text, "type": "concept",
            "evidence": [{"start": 1.0, "end": 4.0, "text": e} for e in evidence],
            "children": []}
    node.update(extra)
    return labels.ConceptCand(cid=cid, sid="s1", node=node)


def _json_ok(content='{"ok": 1}'):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}], "usage": {}})


class _ClientFactory:
    def __init__(self, handler):
        self.payloads = []
        self.clients = []

        def wrapped(request):
            self.payloads.append(json.loads(request.content))
            return handler(request, self.payloads[-1])

        def make():
            client = httpx.Client(transport=httpx.MockTransport(wrapped))
            self.clients.append(client)
            return client

        self.factory = make

    def __call__(self):
        return self.factory()

    def close_all(self):
        for client in self.clients:
            client.close()
        self.clients.clear()


# ---------------------------------------------------------------------------
# D1 — endpoint + model scoped capability cache
# ---------------------------------------------------------------------------
def test_two_models_sharing_one_endpoint_have_independent_capability():
    url = llm.llm_chat_url() or "https://gateway.invalid/v1/chat/completions"
    llm.mark_json_mode_unsupported(url, "model-A")
    assert llm.json_mode_supported(url, "model-A") is False
    assert llm.json_mode_supported(url, "model-B") is True
    assert llm.json_capability_key(url + "/", "model-A") == llm.json_capability_key(url, "model-A")


def test_capability_key_never_contains_credentials():
    key = llm.json_capability_key("https://user:secret@gateway.invalid/v1", "model-A")
    assert "secret" not in repr(key) and "user" not in repr(key)
    assert "secret" not in llm._redact("https://user:secret@host/v1/chat/completions failed")


def test_provider_and_model_switch_does_not_inherit_capability(monkeypatch):
    seen = []

    def fake(url, key, model, messages, max_tokens, temperature, json_mode, attempts=2, max_wait=3.0):
        seen.append((url, model, json_mode))
        return '{"ok": 1}'

    monkeypatch.setattr(llm, "_openai_compatible", fake)
    llm.mark_json_mode_unsupported("https://gateway.invalid/v1/chat/completions", "model-A")
    monkeypatch.setattr(llm, "settings", replace(
        settings, llm_api_key="test-key", llm_base_url="https://gateway.invalid/v1",
        llm_model="model-A", groq_api_key="", hf_token="", hf_local_model=""))
    assert llm.chat_json("s", "u") == {"ok": 1}
    monkeypatch.setattr(llm, "settings", replace(llm.settings, llm_model="model-B"))
    assert llm.chat_json("s", "u") == {"ok": 1}
    assert seen[-2][2] is False and seen[-1][2] is True


# ---------------------------------------------------------------------------
# D2 — precise HTTP 400 parameter detection
# ---------------------------------------------------------------------------
UNRELATED_400 = '{"error":{"message":"Invalid request: model quota must be positive; response_format was ignored"}}'


def test_unrelated_400_does_not_disable_json_mode_or_remove_response_format(monkeypatch):
    factory = _ClientFactory(lambda req, payload: httpx.Response(400, text=UNRELATED_400))
    monkeypatch.setattr(llm, "client", factory)
    try:
        with pytest.raises(llm.LLMError, match="HTTP 400"):
            llm._openai_compatible("https://endpoint.invalid/v1/chat/completions", "k", "m",
                                   [{"role": "user", "content": "hi"}], 100, 0.2, json_mode=True)
    finally:
        factory.close_all()
    assert len(factory.payloads) == 1
    assert "response_format" in factory.payloads[0]
    assert llm._rejected_parameter(UNRELATED_400) is None
    assert llm._NO_JSON_MODE == set()


def test_structured_param_conflicts_win_over_message():
    body = json.dumps({"error": {"message": "Unsupported value: 'response_format' is not supported.",
                                 "param": "temperature"}})
    assert llm._rejected_parameter(body) == "temperature"
    unknown = json.dumps({"error": {"message": "Unsupported value: 'response_format' must be 1.",
                                    "param": "top_p"}})
    assert llm._rejected_parameter(unknown) is None


def test_quoted_field_elsewhere_in_unrelated_message_is_insufficient():
    body = json.dumps({"error": {"message": "Your payload mentioned 'response_format'; model quota must be positive."}})
    assert llm._rejected_parameter(body) is None
    assert llm._rejected_parameter('{"error":{"message":"reasoning tokens were exhausted"}}') is None


def test_valid_response_format_rejection_formats():
    assert llm._rejected_parameter(
        '{"error":{"message":"\'response_format.type\' must be \'json_schema\' or \'text\'"}}'
    ) == "response_format"
    assert llm._rejected_parameter(json.dumps({
        "error": {"message": "Unsupported value: 'response_format' json_object is not supported.",
                  "param": "response_format"}}
    )) == "response_format"
    assert llm._rejected_parameter(json.dumps({
        "error": {"message": "The parameter response_format is not supported."}}
    )) == "response_format"


def test_confirmed_rejection_only_marks_the_matching_endpoint_and_model(monkeypatch):
    url = "https://endpoint.invalid/v1/chat/completions"
    body = json.dumps({"error": {"message": "Unsupported value: 'response_format' is not supported.",
                                 "param": "response_format"}})
    factory = _ClientFactory(lambda req, payload: httpx.Response(400, text=body))
    monkeypatch.setattr(llm, "client", factory)
    try:
        with pytest.raises(llm.LLMError):
            llm._openai_compatible(url, "k", "model-A",
                                   [{"role": "user", "content": "hi"}], 100, 0.2, json_mode=True)
    finally:
        factory.close_all()
    assert llm.json_mode_supported(url, "model-A") is False
    assert llm.json_mode_supported(url, "model-B") is True
    assert len(factory.payloads) == 2  # one bounded compatibility retry, preserved


# ---------------------------------------------------------------------------
# D3 — concise, multilingual, substantive concept eligibility
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("topic,text", [
    ("Bit", "A bit is a binary digit."),
    ("Node", "A node is a device on the network."),
    ("Checksum", "Checksum ensures data integrity."),
    ("Encryption", "Encryption transforms plaintext into ciphertext."),
    ("Router", "राउटर पैकेट को एक नेटवर्क से दूसरे नेटवर्क तक ले जाता है।"),
    ("Network", "Network device ko packet aage forward karta hai."),
])
def test_concise_and_multilingual_definitions_are_eligible(topic, text):
    selected = labels._select_deep_concepts({"x": cand("x", topic, [text])})
    assert list(selected) == ["x"], (topic, text)


def test_filler_and_label_only_evidence_still_rejected():
    concepts = {
        "greet": cand("greet", "Computer Networks",
                      ["Hello everyone, I welcome you all to this wonderful course on computer networks."]),
        "transition": cand("transition", "Next Topic",
                           ["Let us move on to the next topic in this video."]),
        "label": cand("label", "Bit", ["Bit"]),
        "good": cand("good", "Bit", ["A bit is a binary digit."]),
    }
    assert set(labels._select_deep_concepts(concepts)) == {"good"}


def test_user_authored_concept_is_preserved_even_without_evidence():
    concepts = {"mine": cand("mine", "My custom note", ["My own note"], userAuthored=True)}
    assert list(labels._select_deep_concepts(concepts)) == ["mine"]


# ---------------------------------------------------------------------------
# D4 — evidence-preserving duplicate merges
# ---------------------------------------------------------------------------
def test_duplicate_merge_keeps_unique_evidence_and_dedupes_identical_lines():
    a = cand("a", "Node",
             ["A network is a set of nodes connected by links.", "The keywords are nodes and links."])
    b = cand("b", "Node",
             ["A node is a device on the network.", "Routers are nodes too.",
              "The keywords are nodes and links."])
    selected, stats = labels._select_deep_concepts_with_stats({"a": a, "b": b})
    assert stats["duplicates_merged"] == 1
    winner = next(iter(selected))
    texts = [e["text"] for e in selected[winner].node["evidence"]]
    assert texts == [
        "A network is a set of nodes connected by links.",
        "The keywords are nodes and links.",
        "A node is a device on the network.",
        "Routers are nodes too.",
    ]


def test_distinct_concepts_are_not_merged_by_similar_labels():
    concepts = {
        "a": cand("a", "Fibre cable", ["A fibre cable carries light signals."]),
        "b": cand("b", "Fibre equipment", ["Fibre equipment includes an ONT at the home."]),
    }
    selected, stats = labels._select_deep_concepts_with_stats(concepts)
    assert set(selected) == {"a", "b"}
    assert stats["duplicates_merged"] == 0


def test_user_authored_and_generated_concepts_with_same_label_are_not_merged():
    concepts = {
        "gen": cand("gen", "Node", ["A node is a device on the network."]),
        "mine": cand("mine", "Node", ["My own note about nodes."], userAuthored=True),
    }
    selected, stats = labels._select_deep_concepts_with_stats(concepts)
    assert set(selected) == {"gen", "mine"}
    assert stats["duplicates_merged"] == 0


# ---------------------------------------------------------------------------
# D5 — no filler / unavailable concepts in study sets
# ---------------------------------------------------------------------------
def _study_map():
    return {"root": {"id": "root", "type": "root", "text": "R", "children": [{
        "id": "s", "type": "section", "text": "S", "children": [
            {"id": "good", "type": "concept", "text": "Node: a network device", "start": 1, "children": []},
            {"id": "legacy", "type": "concept", "text": "Bit: a binary digit", "start": 2, "children": []},
            {"id": "filler", "type": "concept", "text": "Interesting Scenario", "start": 3,
             "conceptUnavailable": True, "lesson_available": False, "children": []},
            {"id": "rejected", "type": "concept", "text": "Checksum: verifies data", "start": 4,
             "available": False, "children": []},
            {"id": "mine", "type": "concept", "text": "My note", "start": 5,
             "userAuthored": True, "children": []},
        ]}]}}


def test_study_pool_excludes_unavailable_and_unsupported_user_nodes():
    ids = [c["id"] for c in study._concepts(_study_map())]
    assert ids == ["good", "legacy"]


def test_study_focus_ids_cannot_resurrect_excluded_concepts(monkeypatch):
    monkeypatch.setattr(llm, "available", lambda: False)
    result = study.generate_study_set(_study_map(), count=4, focus_ids=["filler", "rejected"])
    assert result["flashcards"] == []
    assert result["quiz"] == []


def test_study_user_node_with_material_can_yield_questions(monkeypatch):
    mm = _study_map()
    mine = mm["root"]["children"][0]["children"][4]
    mine["explain"] = "A user supplied explanation of my note."
    monkeypatch.setattr(llm, "available", lambda: False)
    result = study.generate_study_set(mm, count=6, focus_ids=["mine"])
    assert result["flashcards"] and all(c["nodeId"] == "mine" for c in result["flashcards"])


# ---------------------------------------------------------------------------
# D6 — retry decisions use only current-call diagnostics
# ---------------------------------------------------------------------------
def test_stale_global_status_does_not_cancel_a_valid_retry(monkeypatch):
    bad = dict(VALID_LESSON)
    bad["explain"] = "Too short."
    calls = []

    def fake(system, user, **kw):
        calls.append(user)
        return bad if len(calls) == 1 else dict(VALID_LESSON)

    monkeypatch.setattr(llm, "chat_json", fake)
    monkeypatch.setattr(llm, "LAST_STATUS", 400)  # left over from another call/thread
    mm = _map_with_concept()
    node = mm["root"]["children"][0]["children"][0]
    accepted, failed, _ = labels._llm_deep_lessons(
        mm, {node["id"]: labels.ConceptCand(node["id"], "s1", node)})
    assert node["id"] in accepted and not failed
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# D7 — honest coverage and request statistics
# ---------------------------------------------------------------------------
def _map_with_filler():
    mm = _map_with_concept()
    mm["root"]["children"][0]["children"].append({
        "id": "c2", "type": "concept", "text": "Unrelated filler", "topic": "Unrelated filler",
        "start": 10, "end": 12, "source": "thank you for watching this course",
        "evidence": [{"start": 10, "end": 12, "text": "thank you for watching this course"}],
        "children": [],
    })
    return mm


def test_filler_exclusion_does_not_reduce_legitimate_educational_coverage(monkeypatch):
    from app.pipeline import labels as _labels

    monkeypatch.setattr(_labels.llm, "chat_json", lambda *a, **kw: {"c1": dict(VALID_LESSON)})
    outcome = _labels.label_map(_map_with_filler())
    stats = outcome["stats"]
    for key in ("conceptsExtracted", "conceptsFillerExcluded", "conceptsDuplicatesMerged",
                "conceptsEligible", "legitimateConcepts", "legitimateConceptsUnexplained",
                "lessonsAttempted", "lessonsValidated", "lessonsRejected",
                "conceptsUnexplained", "cacheHits", "logicalLlmCalls", "httpRequests"):
        assert key in stats, key
    # The reported example: 2 extracted, 1 filler excluded, 1 legitimate taught.
    assert stats["conceptsExtracted"] == 2
    assert stats["conceptsFillerExcluded"] == 1
    assert stats["legitimateConcepts"] == 1
    assert stats["lessonsValidated"] == 1
    assert stats["legitimateConceptsUnexplained"] == 0, "filler is not an unexplained lesson"
    assert stats["conceptsUnexplained"] == 1, "source-level count keeps its meaning"
    assert outcome["meta"]["deepStatus"] == "ok: 1 of 1 eligible concepts taught"
    applied = _labels.apply_outcome(dict(_map_with_filler()), outcome)
    label_stats = applied["meta"]["labelStats"]
    assert label_stats["legitimateConceptsUnexplained"] == 0
    assert label_stats["conceptsUnexplained"] == 1


def test_genuinely_unexplained_legitimate_concept_reports_partial(monkeypatch):
    from app.pipeline import labels as _labels

    mm = _map_with_concept()
    mm["root"]["children"][0]["children"].append({
        "id": "c2", "type": "concept", "text": "Fibre cable", "topic": "Fibre cable",
        "start": 20, "end": 24,
        "evidence": [{"start": 20, "end": 24, "text": "A fibre cable carries light signals across long distances."}],
        "source": "A fibre cable carries light signals across long distances.",
        "children": [],
    })
    # One eligible concept is left without a validated lesson, with no rejection
    # recorded: this is a genuine educational coverage gap, not a filler exclusion.
    monkeypatch.setattr(_labels, "_llm_deep_lessons",
                        lambda mm, concepts: ({"c1": dict(VALID_LESSON)}, [], []))
    outcome = _labels.label_map(mm)
    stats = outcome["stats"]
    assert stats["conceptsFillerExcluded"] == 0
    assert stats["legitimateConcepts"] == 2
    assert stats["legitimateConceptsUnexplained"] == 1
    assert outcome["meta"]["deepStatus"] == "partial: 1 of 2 eligible concepts taught"
