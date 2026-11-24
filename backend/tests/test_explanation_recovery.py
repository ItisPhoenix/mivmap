"""Explanation recovery regressions — deterministic, offline, no API quota.

Covers the two root causes behind the empty "Summary / explanation" field:
  * concept explanations live in `explain`, never in `summary`;
  * a provider change can orphan a validated lesson under an old cache key,
    leaving an honest placeholder on an already-cached skeleton map.
"""
from __future__ import annotations

import json

import pytest

from app.features.explanations import has_explanation, recover_explanations
from app.pipeline import lesson_cache


@pytest.fixture(autouse=True)
def isolated_lesson_cache(tmp_path, monkeypatch):
    """Never read the real on-disk lesson cache from a unit test."""
    monkeypatch.setattr(lesson_cache, "cache_dir", lambda: tmp_path)
    return tmp_path

TOPIC = "ONT and router"
EVIDENCE = [
    {"start": 1913.31, "end": 1917.84, "text": "an ONT, Optical Network Terminal; this is the terminal."},
    {"start": 1944.6, "end": 1947.73, "text": "So, this is the ONT that is installed at your home."},
    {"start": 1953.61, "end": 1959.32, "text": "And many times, if you take it from a good ISP, the ONT plus router comes in a single device."},
]

VALID_LESSON = {
    "title": "ONT and Router at Home",
    "explain": "ONT (Optical Network Terminal) is the device at the customer premises that terminates the incoming fiber-optic connection from the ISP. It converts the optical signal on that fiber into a usable connection for the home network, and it is often paired with a router that distributes that connection to devices inside the home.",
    "intuition": "Think of the ONT as the front door of your home network: the fiber from the street stops there, and everything inside the house is served from that entry point. The router is the hallway that directs traffic from that entry point to each room.",
    "why_it_matters": "Knowing which box does what tells you where to plug in your router or Wi-Fi access point, what to check first when the internet is down, and whether your ISP has given you one combined unit or two separate ones.",
    "steps": [
        "The ISP's fiber cable reaches the home and plugs into the ONT.",
        "The ONT terminates the optical line and converts the optical signal into a connection the home equipment can use.",
        "An Ethernet link carries traffic from the ONT toward the router.",
        "The router shares that connection with the devices on the home network.",
        "With many ISPs, the ONT and the router are delivered as one single device, so steps 3 and 4 happen inside the same box.",
    ],
    "worked_example": "A fiber installer mounts a small box on the wall of a house and runs the outdoor fiber into it. The homeowner plugs a Wi-Fi router into that box and connects a laptop and a phone. Later, a different ISP supplies one combined unit, so the wall box and the router are now the same device: the homeowner just plugs it in and joins the Wi-Fi.",
    "diagram": "ISP fiber\n   |\n+-------------+\n|    ONT      |  <- terminates the optical line\n+-------------+\n   | Ethernet\n+-------------+\n|   Router    |  <- shares the connection with home devices\n+-------------+\n   |        |\n Wi-Fi    wired LAN",
    "misconception": "The ONT and the router are always two separate boxes you need to wire together. Correction: many ISPs deliver them as a single combined device, so the ONT function and the routing function can live in the same unit.",
    "exercise": "Sketch a simple home network showing the fiber coming from the street, the ONT, the router, and two client devices, then label which box would disappear if the ISP supplied a combined ONT-plus-router unit.",
    "knowledge_check": {
        "question": "What does ONT stand for, and where is it installed?",
        "answer": "ONT stands for Optical Network Terminal, and it is installed at your home.",
        "explanation": "The terminal is the endpoint of the ISP's optical line at the customer premises, so it sits at the home rather than out in the provider's network.",
    },
    "evidence": {
        "timestamps": [1913.31, 1917.84, 1944.6, 1947.73, 1953.61, 1959.32],
        "quotes": ["an ONT, Optical Network Terminal", "this is the terminal",
                   "the ONT that is installed at your home", "ONT plus router comes in a single device"],
    },
    "provenance": {
        "source": "video+ai+external",
        "video_supported": ["explain", "lesson_evidence"],
        "ai_explanation": ["intuition", "why_it_matters", "steps", "worked_example",
                           "diagram", "misconception", "exercise", "knowledge_check"],
        "external_supplement": [],
        "example_source": "ai-generated",
    },
    "teaching_language": "en",
}


def _map_with_concept(**extra):
    node = {
        "id": "c1", "text": "ONT and Router", "type": "concept", "start": 1913.31,
        "end": 1959.32, "summary": "", "explain": "Explanation unavailable: no verified transcript evidence or AI study aid.",
        "available": False, "generatedBy": "unavailable", "topic": TOPIC,
        "evidence": EVIDENCE, "children": [],
    }
    node.update(extra)
    return {"meta": {"videoId": "4-Ea5EI6LrM", "mode": "deep", "teachingLanguage": "en"},
            "root": {"id": "root", "text": "Networking", "type": "root",
                     "children": [{"id": "s1", "text": "01 ISP tiers", "type": "section",
                                   "children": [node]}]}}


def _write_cached_lesson(monkeypatch, lesson=None, topic=TOPIC, name="old-provider-key.json"):
    path = lesson_cache.cache_dir() / name
    path.write_text(json.dumps({"lesson": lesson or VALID_LESSON, "topic": topic}, ensure_ascii=False), encoding="utf-8")
    return path


def test_cached_lesson_is_found_even_when_provider_key_changed(monkeypatch):
    _write_cached_lesson(monkeypatch)
    found = lesson_cache.find_cached_lesson(TOPIC, EVIDENCE)
    assert found is not None
    assert found["explain"].startswith("ONT (Optical Network Terminal)")
    assert lesson_cache.find_cached_lesson("An unrelated topic", EVIDENCE) is None
    assert lesson_cache.find_cached_lesson(TOPIC, [{"start": 12.0, "end": 18.0, "text": "unrelated"}]) is None


def test_valid_cached_explanation_is_attached_not_copied_into_summary(monkeypatch):
    _write_cached_lesson(monkeypatch)
    mm = _map_with_concept()
    report = recover_explanations(mm)
    assert report["repaired"] == ["c1"], report
    node = mm["root"]["children"][0]["children"][0]
    assert node["explain"].startswith("ONT (Optical Network Terminal)")
    assert node["summary"] == "", "explanations live in `explain`, not the editable summary"
    assert node["lesson_available"] is True and node["available"] is True
    assert node["generatedBy"] == "llm-deep"
    assert node["steps"] and node["knowledge_check"]
    assert "unavailable" not in node["explain"].lower()


def test_invalid_cached_explanation_stays_rejected_with_reason(monkeypatch):
    bad = dict(VALID_LESSON)
    bad["explain"] = "Short filler."
    _write_cached_lesson(monkeypatch, bad)
    mm = _map_with_concept()
    report = recover_explanations(mm)
    assert report["repaired"] == []
    assert report["failed"] and report["failed"][0]["id"] == "c1"
    assert report["failed"][0]["reason"]
    node = mm["root"]["children"][0]["children"][0]
    assert node.get("lesson_available") is not True
    assert has_explanation(node) is False


def test_missing_explanation_has_honest_unavailable_state(monkeypatch):
    mm = _map_with_concept()
    report = recover_explanations(mm)
    assert report["failed"][0]["reason"] == "No validated cached lesson for this concept."
    node = mm["root"]["children"][0]["children"][0]
    assert has_explanation(node) is False
    assert node["explain"].lower().startswith("explanation unavailable")


def test_explanation_survives_serialization_and_old_maps_still_load():
    mm = _map_with_concept(explain="ONT is a fibre terminator at the customer premises with a useful teaching explanation.")
    encoded = json.dumps(mm, ensure_ascii=False)
    restored = json.loads(encoded)
    node = restored["root"]["children"][0]["children"][0]
    assert node["explain"] == "ONT is a fibre terminator at the customer premises with a useful teaching explanation."
    assert has_explanation(node) is True
    legacy = {"id": "x", "type": "concept", "text": "Legacy node", "summary": "old style"}
    assert has_explanation(legacy) is True, "older maps that used summary stay readable"


def test_recovery_targets_only_requested_concepts(monkeypatch):
    _write_cached_lesson(monkeypatch)
    node2 = {"id": "c2", "text": "Other", "type": "concept", "summary": "",
             "explain": "Explanation unavailable: no verified transcript evidence or AI study aid.",
             "topic": "Other", "evidence": [], "children": []}
    mm = _map_with_concept()
    mm["root"]["children"][0]["children"].append(node2)
    report = recover_explanations(mm, ["c2"])
    assert report["repaired"] == []
    assert report["attempted"] == 1
    assert report["failed"][0]["id"] == "c2"
