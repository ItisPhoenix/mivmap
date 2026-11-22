"""Study-readiness regressions: real terminology, honest provenance, usable recall."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline.builder import build_skeleton, rebuild_recall_from_lessons  # noqa: E402
from app.pipeline.lessons import validate_deep_teaching  # noqa: E402
from app.pipeline.text_utils import extract_json  # noqa: E402
from app.pipeline.terminology import (  # noqa: E402
    acronym_expansion, accuracy_notes_for, apply_accuracy_corrections,
    is_meaningful_label, normalize_label,
)
from app.quality import analyze_map  # noqa: E402


def test_llm_json_extraction_tolerates_reasoning_wrappers():
    assert extract_json('<think>reasoning with a lone { brace</think>\n{"a": 1, "b": {"c": 2}}') == \
        {"a": 1, "b": {"c": 2}}
    assert extract_json('Sure: ```json\n{"ok": true}\n```') == {"ok": True}


def test_fragments_become_canonical_terminology_or_are_rejected():
    assert normalize_label("Local Area", "A LAN is a local area network in one building") == "LAN"
    assert normalize_label("Local Area", "the fiber is extended into different localities") == \
        "Local access network (last mile)"
    assert normalize_label("Sign of Internet", "Different networks started forming") == \
        "Internet (network of networks)"
    assert normalize_label("Tower", "your mobile connects to your tower") == "Cell tower"
    assert normalize_label("Cables", "fiber cables are laid on the seabed") == "Submarine fibre cables"
    assert normalize_label("Waves", "Wi-Fi communication happens through radio waves") == "Radio waves (Wi-Fi)"
    assert normalize_label("Ont Plus Router", "the ONT plus router comes in a single device") == "ONT and router"
    assert normalize_label("Provider Network", "") == ""
    assert normalize_label("Geographical Categories", "") == ""
    for junk in ("Hard Work", "Motivated", "Researching", "Gained", "Relates", "X15 Room"):
        assert not is_meaningful_label(junk), junk
    assert is_meaningful_label("Medium access control")
    assert is_meaningful_label("Artificial neuron: weighted sum of inputs")


def test_accuracy_corrections_are_external_and_never_touch_evidence():
    evidence = [{"start": 10.0, "end": 12.0, "text": "this WAN is also a type of internet"}]
    lesson = {
        "explain": "WAN (Wide Area Network) is a type of internet because it spans countries.",
        "evidence": {"timestamps": [10.0], "quotes": ["this WAN is also a type of internet"]},
        "source": "this WAN is also a type of internet",
    }
    fixed = apply_accuracy_corrections(lesson, "WAN")
    assert "wan is a wide-area network" in fixed["explain"].lower()
    assert "a type of internet" not in fixed["explain"].lower()
    assert fixed["accuracy_notes"][0]["source"] == "external-supplement"
    assert fixed["accuracy_notes"][0]["supplemented"] is True
    assert fixed["evidence"] == lesson["evidence"]
    assert fixed["source"] == lesson["source"]

    arpa = apply_accuracy_corrections({"explain": "ARPANET was an early research network."}, "ARPANET")
    assert "Advanced Research Projects Agency Network" in arpa["explain"]
    assert acronym_expansion("ARPANET") == "Advanced Research Projects Agency Network"
    assert accuracy_notes_for("ARPANET")

    tower = apply_accuracy_corrections(
        {"explain": "The call reaches a tower located at the US-based provider.", "steps": []}, "Cell tower")
    assert "US-based provider" not in tower["explain"]
    assert "core network" in tower["explain"]

    wifi = apply_accuracy_corrections(
        {"explain": "Wi-Fi throughput is set by wavelength. The data speed is inversely proportional "
                    "to the wavelength, which corresponds to the range."}, "Wi-Fi throughput")
    assert "inversely proportional" not in wifi["explain"].lower()
    assert "modulation" in wifi["explain"].lower()
    assert wifi["accuracy_notes"][0]["required"] is True


def test_deep_teaching_requires_three_way_provenance_and_language():
    lesson = {
        "explain": "LAN stands for Local Area Network and connects devices in one building.",
        "intuition": "A LAN is like the internal mailroom of one office building.",
        "why_it_matters": "It lets devices share files before traffic reaches an ISP.",
        "worked_example": "A laptop and a printer on one switch form a LAN.",
        "misconception": "A LAN is not the same thing as the Internet.",
        "provenance": {
            "source": "video+ai+external",
            "video_supported": ["explain"],
            "ai_explanation": ["intuition", "worked_example"],
            "external_supplement": [],
        },
        "acronym_expansion": "Local Area Network",
        "teaching_language": "en",
    }
    ok, errors = validate_deep_teaching(lesson, "LAN", "en")
    assert ok, errors

    bad = dict(lesson)
    bad.pop("provenance")
    ok, errors = validate_deep_teaching(bad, "LAN", "en")
    assert not ok and "three-way provenance missing" in errors

    bad = dict(lesson)
    bad["acronym_expansion"] = "Local Area Net"
    ok, errors = validate_deep_teaching(bad, "LAN", "en")
    assert not ok and any("acronym" in e for e in errors)

    hi = dict(lesson)
    hi.update({
        "explain": "LAN का अर्थ है Local Area Network, जो एक इमारत के डिवाइसों को जोड़ता है।",
        "intuition": "LAN को एक इमारत की आंतरिक डाकघर की तरह समझें।",
        "why_it_matters": "इससे डिवाइस इंटरनेट तक जाने से पहले फाइलें साझा कर सकते हैं।",
        "worked_example": "एक लैपटॉप और प्रिंटर एक ही स्विच से जुड़कर LAN बनाते हैं।",
        "misconception": "LAN और इंटरनेट एक ही चीज़ नहीं हैं।",
    })
    ok, errors = validate_deep_teaching(hi, "LAN", "hi")
    assert ok, errors
    ok, errors = validate_deep_teaching(lesson, "LAN", "hi")  # English text, Hindi requested
    assert not ok and any("teaching language" in e for e in errors)


def test_quick_recall_is_compact_qa_not_raw_transcript():
    mm = {
        "root": {
            "children": [
                {"type": "section", "start": 0, "end": 100,
                 "children": [{"type": "concept", "id": "n1", "text": "LAN", "start": 1.0,
                               "source": "A LAN is a local area network.", "available": True,
                               "knowledge_check": {"question": "What does LAN stand for?",
                                                   "answer": "Local Area Network"}}]},
                {"type": "section", "recall": True, "start": 0, "end": 100, "summary": "Short revision",
                 "children": [{"type": "concept", "oneline": True, "text": "In one line: Short revision"},
                              {"type": "concept", "text": "raw transcript fragment"}]},
            ]
        }
    }
    rebuild_recall_from_lessons(mm)
    recall = mm["root"]["children"][1]
    qa = [c for c in recall["children"] if c.get("qa")]
    assert len(qa) == 1
    assert qa[0]["question"] == "What does LAN stand for?"
    assert qa[0]["answer"] == "Local Area Network"
    assert "Q:" in qa[0]["text"] and "A:" in qa[0]["text"]
    assert qa[0]["start"] == 1.0


def test_hindi_transcript_and_timestamps_are_preserved_in_export():
    transcript = [
        {"start": 0.0, "end": 6.0,
         "text": "आज हम नेटवर्क के बारे में बात करेंगे। LAN और WAN दोनों महत्वपूर्ण हैं।"},
        {"start": 6.0, "end": 12.0,
         "text": "LAN एक इमारत के अंदर के डिवाइसों को जोड़ता है, जबकि WAN शहरों को जोड़ता है।"},
        {"start": 12.0, "end": 18.0,
         "text": "राउटर और फाइबर केबल इंटरनेट को घर तक पहुंचाते हैं।"},
    ]
    mm = build_skeleton({"videoId": "HINDIHINDI01", "title": "नेटवर्क", "transcript": transcript,
                         "mode": "revision", "useLLM": False, "allowWhisper": False,
                         "teachingLanguage": "hi"})
    assert mm["meta"]["teachingLanguage"] == "hi"
    assert mm["meta"]["sourceLanguage"] in ("hi", "en")
    original = mm["meta"]["originalTranscript"]
    if original:  # NLLB model present: it translated, but the source must be kept
        assert [e["text"] for e in original] == [e["text"] for e in transcript]
        assert [(e["start"], e["end"]) for e in original] == [(e["start"], e["end"]) for e in transcript]
    # Working transcript still carries real timestamps.
    assert all(e["start"] in (0.0, 6.0, 12.0) for e in mm["transcript"])


def test_fragmentary_concept_label_is_flagged_by_the_gates():
    mm = {
        "meta": {"mode": "deep", "teachingLanguage": "en", "generationMs": 1},
        "root": {"type": "root", "text": "Test", "summary": "", "keywords": [], "children": [{
            "type": "section", "text": "Section", "start": 0, "end": 10, "children": [{
                "type": "concept", "id": "c1", "text": "Sign of Internet", "topic": "Sign of Internet",
                "start": 1, "end": 2, "source": "Some transcript evidence here.",
                "evidence": [{"start": 1, "end": 2, "text": "Some transcript evidence here."}],
            }],
        }]},
        "transcript": [],
    }
    report = analyze_map(mm)
    assert not report.passed
    assert any("concepts_not_fragmentary" in f for f in report.failures)
