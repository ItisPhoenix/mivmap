"""Regressions for PR #1 review findings: cache, evidence, multilingual, citations."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline import builder, labels, lesson_cache  # noqa: E402
from app.pipeline.lessons import overlap, validate_deep_teaching, validate_lesson  # noqa: E402
from app.pipeline.text_utils import (  # noqa: E402
    check_teaching_languages, field_language_ok, normalize_language, unicode_norm,
)
from app.pipeline.terminology import REFERENCE_URLS, accuracy_notes_for  # noqa: E402
from app.pipeline.transcript import TranscriptEntry  # noqa: E402

SCRIPTS = Path(__file__).parents[2] / "scripts"


# ---------------------------------------------------------------------------
# FIX 1: Unicode-aware, collision-free lesson cache keys
# ---------------------------------------------------------------------------
def test_hindi_evidence_with_same_timestamps_has_distinct_keys():
    evidence_a = [{"start": 10.0, "text": "राउटर पैकेट को आगे भेजता है।"}]
    evidence_b = [{"start": 10.0, "text": "फाइबर केबल समुद्र के नीचे जाती है।"}]
    a = lesson_cache.lesson_key("राउटर", evidence_a, "model", "https://provider", "hi")
    b = lesson_cache.lesson_key("राउटर", evidence_b, "model", "https://provider", "hi")
    assert a != b
    assert len(a) == 32
    assert lesson_cache.lesson_key("राउटर", evidence_a, "model", "https://provider", "hi") == a


def test_cache_key_includes_language_model_and_prompt_version():
    evidence = [{"start": 1.0, "text": "LAN connects devices."}]
    base = lesson_cache.lesson_key("LAN", evidence, "m1", "https://p", "en")
    assert base != lesson_cache.lesson_key("LAN", evidence, "m1", "https://p", "hi")
    assert base != lesson_cache.lesson_key("LAN", evidence, "m2", "https://p", "en")
    assert base != lesson_cache.lesson_key("LAN", evidence, "m1", "https://other", "en")
    assert "deep-lesson-v4" in lesson_cache.PROMPT_VERSION


def test_english_cache_behavior_remains_deterministic():
    evidence = [{"start": 1.0, "text": "LAN connects devices."}]
    assert lesson_cache.lesson_key("LAN", evidence, "m", "u", "en") == \
        lesson_cache.lesson_key("LAN", evidence, "m", "u", "en")


# ---------------------------------------------------------------------------
# FIX 2: strongest essential-topic evidence wins
# ---------------------------------------------------------------------------
def _mindmap_for_topics():
    return {
        "meta": {"mode": "deep"},
        "root": {"children": [{"type": "section", "start": 0, "end": 100, "children": []}]},
        "transcript": [],
    }


def test_early_weak_mention_loses_to_later_definition(monkeypatch):
    monkeypatch.setattr(builder, "active_topics", lambda mm: [("LAN", ["lan"])])
    mm = _mindmap_for_topics()
    weak = TranscriptEntry(0.0, 5.0, "I mentioned lan briefly before other topics.")
    strong = TranscriptEntry(50.0, 56.0,
                             "A LAN is a local area network that connects devices inside one building.")
    builder.ensure_essential_topics(mm, [weak, strong])
    concept = mm["root"]["children"][0]["children"][0]
    assert concept["start"] == 50.0
    assert "A LAN is a local area network" in concept["source"]


def test_equal_relevance_selects_earliest_timestamp(monkeypatch):
    monkeypatch.setattr(builder, "active_topics", lambda mm: [("LAN", ["lan"])])
    mm = _mindmap_for_topics()
    text = "The lan setup in this building uses a switch and a router for local devices."
    builder.ensure_essential_topics(mm, [TranscriptEntry(10.0, 15.0, text),
                                         TranscriptEntry(2.0, 7.0, text)])
    assert mm["root"]["children"][0]["children"][0]["start"] == 2.0


def test_no_matching_evidence_creates_no_concept(monkeypatch):
    monkeypatch.setattr(builder, "active_topics", lambda mm: [("LAN", ["lan"])])
    mm = _mindmap_for_topics()
    builder.ensure_essential_topics(mm, [TranscriptEntry(0.0, 5.0, "This caption has no relevant keyword here.")])
    assert mm["root"]["children"][0]["children"] == []


# ---------------------------------------------------------------------------
# FIX 3 + Hindi parsing audit: Unicode validation and clean_label
# ---------------------------------------------------------------------------
def _hindi_lesson() -> dict:
    return {
        "title": "LAN नेटवर्क",
        "acronym_expansion": "Local Area Network",
        "explain": "LAN यानी Local Area Network एक छोटे दायरे का नेटवर्क है जो एक इमारत के डिवाइसों को जोड़ता है।",
        "intuition": "सोचिए कि यह एक इमारत की आंतरिक डाकघर व्यवस्था है।",
        "why_it_matters": "इससे कंप्यूटर आपस में फाइलें साझा कर सकते हैं।",
        "steps": ["पहले लैन के डिवाइस पहचानें।", "फिर उन्हें स्विच से जोड़ें।", "अंत में स्थानीय पते दें।"],
        "worked_example": "एक कार्यालय में लैपटॉप और प्रिंटर एक ही WAN से जुड़े हैं।",
        "misconception": "गलत धारणा कि LAN और WAN एक ही हैं; दोनों के दायरे अलग हैं।",
        "exercise": "अपने घर का LAN नेटवर्क बनाएँ और उसके सभी डिवाइसों को चिह्नित करें।",
        "knowledge_check": {
            "question": "LAN का दायरा क्या है?",
            "answer": "LAN एक इमारत तक सीमित नेटवर्क है।",
            "explanation": "LAN स्थानीय नेटवर्क है, जबकि WAN कई शहरों को जोड़ता है।",
        },
        "diagram": "[device] -- [switch] -- [router]",
    }


def test_unicode_norm_preserves_devanagari_and_topic_matching():
    assert "राउटर" in unicode_norm("राउटर पैकेट को आगे भेजता है।")
    assert unicode_norm("WAN, TCP/IP!") == "wan tcp ip"
    assert overlap("राउटर पैकेट भेजता है", "राउटर पैकेट भेजता है") == 1.0


def test_all_hindi_fields_validated_and_english_fields_rejected():
    lesson = _hindi_lesson()
    assert check_teaching_languages(lesson, "hi-IN") == []
    english = dict(lesson)
    english["explain"] = "A LAN connects devices inside a single building."
    assert any("explain" in e for e in check_teaching_languages(english, "hi"))
    steps = dict(lesson)
    steps["steps"] = ["Identify devices.", "Connect them to a switch.", "Assign local addresses."]
    assert any("steps" in e for e in check_teaching_languages(steps, "hi"))
    answer = dict(lesson)
    answer["knowledge_check"] = dict(answer["knowledge_check"])
    answer["knowledge_check"]["answer"] = "It covers one building."
    assert any("knowledge_check.answer" in e for e in check_teaching_languages(answer, "hi"))


def test_mixed_technical_teaching_locale_variants_and_diagram_exemption():
    assert field_language_ok("राउटर WAN TCP/IP ONT से जुड़ता है।", "hi")
    assert field_language_ok("LAN (Local Area Network)", "hi")  # canonical technical-only
    assert field_language_ok("LAN connects devices.", "en-US")
    assert normalize_language("hi-IN") == "hi"
    assert normalize_language("en_US") == "en"
    assert check_teaching_languages({"diagram": "[a] -- [b]", "explain": "यह एक नेटवर्क है।"}, "hi") == []
    assert check_teaching_languages(_hindi_lesson(), "fr") == [
        "unsupported teaching language for validation: fr"]


def test_lesson_acceptance_uses_shared_language_validation():
    lesson = _hindi_lesson()
    evidence = [{"start": 1.0, "end": 2.0, "text": "LAN एक इमारत के अंदर डिवाइसों को जोड़ता है।"}]
    assert validate_lesson(lesson, "LAN एक नेटवर्क है।", "LAN", evidence, "hi-IN")[0]
    assert not validate_lesson(lesson, "LAN एक नेटवर्क है।", "LAN", evidence, "en")[0]
    assert validate_deep_teaching({"provenance": {
        "video_supported": [], "ai_explanation": ["explain"], "external_supplement": []},
        **lesson}, "LAN", "hi-IN")[1] == []


def test_clean_label_accepts_devanagari_and_rejects_caption_sentences():
    assert labels.clean_label("राउटर", 60, 8, "hi") == "राउटर"
    assert labels.clean_label("राउटर WAN", 60, 8, "hi-IN") == "राउटर WAN"
    assert labels.clean_label("LAN", 60, 8, "en") == "LAN"
    assert labels.clean_label("यह एक पूरा कैप्शन वाक्य है।", 110, 16, "hi") is None
    assert labels.clean_label("here is the definition", 110, 16, "en") is None


# ---------------------------------------------------------------------------
# FIX 5: verification script is self-contained and reports honestly
# ---------------------------------------------------------------------------
def _load_verification():
    spec = importlib.util.spec_from_file_location("generate_verification", SCRIPTS / "generate_verification.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_verification_input_validation():
    mod = _load_verification()
    assert mod.validate_input({
        "title": "Test", "teachingLanguage": "hi-IN",
        "transcript": [{"start": 0, "end": 5, "text": "नमस्ते"}]}) == []
    errors = mod.validate_input({
        "title": "", "teachingLanguage": "zz",
        "transcript": [{"start": 5, "end": 1, "text": ""}]})
    assert any("title" in e for e in errors)
    assert any("teachingLanguage" in e for e in errors)
    assert any("end" in e for e in errors)
    assert any("text" in e for e in errors)


def test_synthetic_cases_are_not_real_youtube_and_report_not_run():
    mod = _load_verification()
    labels_and_cases = mod._synthetic_cases()
    assert len(labels_and_cases) == 3
    assert all(label.startswith("synthetic-") for label, _, _ in labels_and_cases)
    assert mod.real_video_status()["status"] == "NOT RUN"
    assert "examples/verification" not in Path(mod.__file__).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# FIX 6: citation quality
# ---------------------------------------------------------------------------
def test_every_correction_has_verified_authoritative_citation():
    for topic in ("wan", "arpanet", "cell tower", "submarine fibre cables",
                  "osi model", "tcp/ip", "wi fi throughput", "lan", "pan", "man"):
        for note in accuracy_notes_for(topic):
            assert note["basis"] in REFERENCE_URLS
            title, url = REFERENCE_URLS[note["basis"]]
            assert note["title"] == title and note["url"] == url
            assert url.startswith("https://")
            assert "wikipedia" not in url.lower()
            assert note["source"] == "external-supplement"
            assert note["supplemented"] is True
