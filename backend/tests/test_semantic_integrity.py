"""Semantic-integrity regressions. Self-contained: no copyrighted exports required."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline import labels, llm  # noqa: E402
from app.pipeline.builder import generate_mindmap  # noqa: E402
from app.pipeline.terminology import (  # noqa: E402
    misleading_detail, normalize_label, accuracy_notes_for,
)
from app.quality import analyze_map  # noqa: E402


def _map() -> dict:
    """Minimal study-ready map with two taught concepts and a Q&A recall."""
    def concept(cid: str, text: str, topic: str) -> dict:
        return {
            "type": "concept", "id": cid, "text": text, "topic": topic,
            "start": 1, "end": 2, "source": "Transcript evidence for this concept.",
            "explain": f"{text} is a networking concept taught in this lesson with enough detail to be useful.",
            "intuition": "A plain-language mental model for the topic.",
            "why_it_matters": "It matters for understanding how networks work.",
            "steps": ["First step in the mechanism.", "Second step in the mechanism.",
                      "Third step in the mechanism."],
            "worked_example": "A concrete example specific to the topic.",
            "exercise": "Draw the path and label each hop.",
            "knowledge_check": {"question": "What is the purpose of this concept?",
                                "answer": "It explains a specific networking mechanism.",
                                "explanation": "The lesson explains the mechanism step by step."},
            "lesson_evidence": {"timestamps": [1.0], "quotes": []},
            "provenance": {"source": "video+ai", "video_supported": ["explain"],
                           "ai_explanation": ["intuition"], "external_supplement": [],
                           "example_source": "ai-generated"},
            "lesson_available": True, "available": True, "children": [],
        }

    return {
        "meta": {"mode": "deep", "teachingLanguage": "en", "generationMs": 1},
        "root": {"type": "root", "text": "Test", "summary": "WAN · LAN",
                 "keywords": ["WAN", "LAN"], "children": [
                     {"type": "section", "text": "Section", "start": 0, "end": 10,
                      "children": [concept("c1", "WAN", "WAN"), concept("c2", "LAN", "LAN")]},
                     {"type": "section", "recall": True, "summary": "WAN · LAN", "children": [
                         {"type": "concept", "oneline": True, "text": "In one line: WAN · LAN"},
                         {"type": "concept", "qa": True, "question": "What is a WAN?",
                          "answer": "It explains a specific networking mechanism.", "text": "Q: What is a WAN?"},
                     ]},
                 ]},
        "transcript": [],
    }


def _faults(map_: dict) -> str:
    return "\n".join(analyze_map(map_).failures)


def test_detected_wrong_detail_parent():
    map_ = _map()
    map_["root"]["children"][0]["children"][1]["children"] = [
        {"type": "detail", "tag": "Def", "text": "Def: status code 200 indicates success", "start": 1}]
    assert "detail_parent_semantics" in _faults(map_)


def test_detected_raw_misleading_detail():
    map_ = _map()
    node = map_["root"]["children"][0]["children"][0]
    node["topic"] = "Cell tower"
    node["accuracy_notes"] = accuracy_notes_for("Cell tower")
    node["children"] = [{"type": "detail", "tag": "Def",
                         "text": "Def: goes to the tower located at the US-based providers", "start": 1}]
    assert "detail_no_contradiction" in _faults(map_)


def test_detected_false_external_provenance():
    map_ = _map()
    prov = map_["root"]["children"][0]["children"][0]["provenance"]
    prov["source"] = "video+ai+external"
    prov["external_supplement"] = []
    prov["example_source"] = "external"
    assert "provenance_honest" in _faults(map_)


def test_detected_generic_recall_and_root_summary():
    map_ = _map()
    map_["root"]["summary"] = "Covers WAN & LAN"
    recall = map_["root"]["children"][1]
    recall["summary"] = "Covers WAN & LAN"
    recall["children"] = [{"type": "concept", "oneline": True, "text": "In one line: Covers WAN & LAN"}]
    faults = _faults(map_)
    assert "recall_meaningful" in faults and "root_summary_clean" in faults


def test_fiber_label_maps_to_the_topic_actually_taught():
    outage = "Local network independence during a fiber outage keeps local printing working."
    assert normalize_label("Fiber", outage) == "Local network during a fiber outage"
    assert normalize_label("Fiber", "Fiber cable carries data under the ocean.") == "Fibre cable"


def test_devanagari_fragment_becomes_a_stable_concept_name():
    assert normalize_label("इससे एक ही इमारत के उपकरण फाइलें साझा कर सकते हैं। राउटर...", "") == "Router"
    assert normalize_label("फाइबर केबल समुद्र के नीचे से जाती है", "") == "Fibre cable"


def test_misleading_detail_detection():
    assert misleading_detail("goes to the tower located at the US-based providers", "Cell tower")
    assert not misleading_detail("The tower is a radio access point.", "Cell tower")


def test_external_notes_have_citations():
    for note in accuracy_notes_for("ARPANET") + accuracy_notes_for("OSI model"):
        assert note["title"]
        assert note["url"].startswith("https://")


def test_failed_lesson_keeps_meaningful_concept_as_unavailable(monkeypatch):
    transcript = [
        {"start": 0.0, "end": 6.0, "text": "राउटर पैकेट को एक नेटवर्क से दूसरे नेटवर्क तक ले जाता है।"},
        {"start": 6.0, "end": 12.0, "text": "यह एक इमारत के अंदर devices को जोड़ता है।"},
        {"start": 12.0, "end": 18.0, "text": "LAN और WAN दोनों नेटवर्क के प्रकार हैं।"},
    ]
    monkeypatch.setattr(llm, "fast_available", lambda: True)
    monkeypatch.setattr(labels, "_llm_deep_lessons", lambda mm, cs: ({}, ["provider failed"]))
    mm = generate_mindmap({"videoId": "UNAVAIL01", "title": "नेटवर्क",
                           "transcript": transcript, "mode": "deep",
                           "useLLM": True, "allowWhisper": False, "teachingLanguage": "hi"})
    concepts = [c for s in mm["root"]["children"] if not s.get("recall")
                for c in s.get("children", []) if c.get("type") == "concept"]
    assert concepts
    assert all(c.get("conceptUnavailable") for c in concepts)
    assert all(not c.get("lesson_available") for c in concepts)
    assert all(not any(ch.get("type") == "detail" for ch in c.get("children", [])) for c in concepts)
    assert all(isinstance(c.get("start"), (int, float)) for c in concepts)
