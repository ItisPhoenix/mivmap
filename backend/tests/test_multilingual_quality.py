"""Regressions for PR #1 final multilingual remediation (topics, language, lessons, citations)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline import builder, labels  # noqa: E402
from app.pipeline.lessons import ACTION_VERBS, _topic_terms, validate_lesson  # noqa: E402
from app.pipeline.terminology import (  # noqa: E402
    REFERENCE_URLS, TECHNICAL_CORRECTIONS, accuracy_notes_for, canonical_topic,
)
from app.pipeline.text_utils import (  # noqa: E402
    check_teaching_languages, field_language_ok, unicode_tokens,
)
from app.pipeline.topics import active_topics, detect_domain  # noqa: E402
from app.pipeline.transcript import TranscriptEntry  # noqa: E402

EN_BLOCKS = [
    {"text": "A LAN is a local area network inside one building, and a WAN connects cities.", "start": 0.0},
    {"text": "A router forwards packets between the LAN and the wide area network.", "start": 10.0},
    {"text": "ARPANET and TCP/IP laid the foundation for the internet and the OSI model.", "start": 20.0},
    {"text": "Fibre cables and ISPs carry traffic between content delivery networks.", "start": 30.0},
]
HI_BLOCKS = [
    {"text": "लैन एक इमारत के अंदर डिवाइसों को जोड़ता है और वान कई शहरों को जोड़ता है।", "start": 0.0},
    {"text": "राउटर पैकेट को एक नेटवर्क से दूसरे नेटवर्क तक ले जाता है।", "start": 10.0},
    {"text": "फाइबर केबल समुद्र के नीचे डेटा ले जाती है और इंटरनेट नेटवर्कों का नेटवर्क है।", "start": 20.0},
    {"text": "ओएसआई मॉडल सात परतों का ढांचा है और टीसीपी/आईपी प्रोटोकॉल है।", "start": 30.0},
]
HINGLISH_BLOCKS = [
    {"text": "आज हम networking पढ़ेंगे। LAN और WAN में क्या अंतर है?", "start": 0.0},
    {"text": "राउटर पैकेट को एक network से दूसरे network तक ले जाता है।", "start": 10.0},
    {"text": "फाइबर cable समुद्र के नीचे international traffic ले जाती है।", "start": 20.0},
    {"text": "इंटरनेट सभी networks को जोड़ने वाला network है।", "start": 30.0},
]


# ---------------------------------------------------------------------------
# PHASE 1 - English / Hindi / Hinglish topic detection
# ---------------------------------------------------------------------------
def test_english_domain_and_topics():
    domain, hits = detect_domain(EN_BLOCKS)
    assert domain == "networking"
    assert {"LAN", "WAN", "Router", "ARPANET", "TCP/IP", "OSI model"} <= {h.label for h in hits}


def test_pure_devanagari_domain_and_topics():
    domain, hits = detect_domain(HI_BLOCKS)
    assert domain == "networking"
    labels_found = {h.label for h in hits}
    for expected in ("LAN", "WAN", "Router", "Fibre infrastructure"):
        assert expected in labels_found, (expected, labels_found)


def test_hinglish_domain_and_topics():
    domain, hits = detect_domain(HINGLISH_BLOCKS)
    assert domain == "networking"
    assert {"LAN", "WAN", "Router", "Fibre infrastructure"} <= {h.label for h in hits}


def test_short_acronyms_do_not_false_positive():
    text = "The command and the multinational panel were discussed in detail today."
    domain, hits = detect_domain([{"text": text, "start": 0.0}])
    assert domain == "generic"
    assert not hits


def test_essential_topics_use_unicode_patterns(monkeypatch):
    monkeypatch.setattr(builder, "active_topics", lambda mm: active_topics(blocks=HI_BLOCKS))
    mm = {
        "meta": {"mode": "deep"},
        "root": {"children": [{"type": "section", "start": 0, "end": 40, "children": []}]},
        "transcript": [],
    }
    entries = [TranscriptEntry(b["start"], b["start"] + 9, b["text"]) for b in HI_BLOCKS]
    builder.ensure_essential_topics(mm, entries)
    topics = {c.get("topic") for c in mm["root"]["children"][0]["children"]}
    assert {"LAN", "WAN", "Router", "Fibre infrastructure"} <= topics
    for c in mm["root"]["children"][0]["children"]:
        assert c["evidence"] and c["start"] in {e.start for e in entries}


# ---------------------------------------------------------------------------
# PHASE 2 - robust language validation
# ---------------------------------------------------------------------------
def _lesson(**overrides):
    out = {
        "title": "लैन नेटवर्क",
        "explain": "लैन यानी Local Area Network एक इमारत के डिवाइसों को जोड़ने वाला नेटवर्क है।",
        "intuition": "सोचिए कि यह इमारत की अपनी डाकघर व्यवस्था है, जहाँ सब डिवाइस आपस में मिलते हैं।",
        "why_it_matters": "इससे कंप्यूटर एक दूसरे के साथ फाइलें और प्रिंटर साझा कर सकते हैं।",
        "steps": ["पहले लैन के सभी डिवाइस पहचानें।", "फिर उन्हें स्विच और राउटर से जोड़ें।",
                  "अंत में स्थानीय पते देकर आपस में संवाद शुरू करें।"],
        "worked_example": "एक कार्यालय में लैपटॉप और प्रिंटर लैन से जुड़कर एक फ़ोल्डर साझा करते हैं।",
        "misconception": "गलत धारणा कि लैन और वान एक ही हैं; लैन एक इमारत तक सीमित है।",
        "exercise": "अपने घर का लैन नेटवर्क बनाएँ और उसके सभी डिवाइसों की सूची बनाएँ।",
        "knowledge_check": {
            "question": "लैन का दायरा क्या है?",
            "answer": "लैन एक इमारत या परिसर तक सीमित नेटवर्क है।",
            "explanation": "लैन स्थानीय नेटवर्क है, जबकि वान कई शहरों को जोड़ता है।",
        },
        "diagram": "[device] -- [switch] -- [router]",
    }
    out.update(overrides)
    return out


def test_positive_language_cases():
    assert check_teaching_languages({
        "explain": "A LAN connects devices inside one building or campus.",
        "exercise": "Draw your home network and list the devices connected to it.",
    }, "en-US") == []
    assert check_teaching_languages(_lesson(), "hi-IN") == []
    hinglish = _lesson(explain="राउटर WAN और LAN को जोड़ता है और पैकेट को आगे भेजता है।")
    assert field_language_ok(hinglish["explain"], "hi")


def test_mostly_english_with_hindi_words_fails():
    fake = "A राउटर is a device that forwards packets between two different networks."
    assert not field_language_ok(fake, "hi")
    assert "explain" in [e.split()[0] for e in check_teaching_languages(_lesson(explain=fake), "hi")]
    english_lesson = _lesson(
        explain="A LAN is a local area network that connects devices inside one building.",
        intuition="Think of it as a private postal system for one office building.",
        why_it_matters="It matters because devices can share files and printers locally.",
        steps=["First, identify every device on the local network.",
               "Next, connect the devices to a local switch or router.",
               "Then, assign local addresses so the devices can communicate."],
        worked_example="For example, a laptop and a printer in one office share a folder over the LAN.",
        misconception="A common misconception is that a LAN is the same thing as the internet.",
        exercise="Draw your home network and label the devices connected to it.",
        knowledge_check={"question": "What does LAN stand for?",
                         "answer": "Local Area Network.",
                         "explanation": "The lesson defines LAN as a local area network."},
    )
    assert check_teaching_languages(english_lesson, "hi")


def test_all_fields_and_empty_fields():
    errors = check_teaching_languages(_lesson(), "hi")
    assert errors == []
    for field in ("title", "explain", "intuition", "why_it_matters", "worked_example",
                  "misconception", "exercise"):
        bad = _lesson(**{field: "A purely English sentence about the network."})
        assert any(field in e for e in check_teaching_languages(bad, "hi")), field
    bad_steps = _lesson(steps=["Identify the devices.", "Connect them to a switch.",
                               "Assign local addresses."])
    assert any("steps" in e for e in check_teaching_languages(bad_steps, "hi"))
    bad_check = _lesson(knowledge_check={"question": "What is a LAN?",
                                         "answer": "A local area network.",
                                         "explanation": "It covers one building."})
    assert any("knowledge_check" in e for e in check_teaching_languages(bad_check, "hi"))
    ok, errs = validate_lesson(_lesson(explain=""), "LAN एक नेटवर्क है।", "LAN", None, "hi")
    assert not ok and any("too short" in e for e in errs)


def test_unsupported_language_fails():
    assert check_teaching_languages(_lesson(), "fr") == [
        "unsupported teaching language for validation: fr"]
    assert not field_language_ok("A normal English sentence.", "fr")


# ---------------------------------------------------------------------------
# PHASE 3 - Hindi Deep lesson generation and diagnostics
# ---------------------------------------------------------------------------
def test_caption_like_topic_is_canonicalised():
    assert canonical_topic("राउटर पैकेट को एक नेटवर्क से दूसरे नेटवर्क तक ले जाता है। इंटरनेट दुनिया…") == "Router"
    assert canonical_topic("ऑप्टिकल फाइबर केबल समुद्र के नीचे से डेटा ले जाती है।") == "Fibre cable"
    assert canonical_topic("LAN") == "LAN"


def test_hindi_hints_added_for_multiword_topics_and_hindi_verbs():
    assert "फाइबर" in _topic_terms("Fibre cable")
    assert "फाइबर केबल" in _topic_terms("Fibre cable")
    assert "इंटरनेट" in _topic_terms("Internet (network of networks)")
    assert "राउटर" in _topic_terms("Router")
    for verb in ("कीजिए", "करें", "देखिए", "बनाएं", "लिखिए"):
        assert ACTION_VERBS.search(f"पहले_netवर्क_देखें, फिर {verb}।"), verb


def test_provider_response_shapes_are_parsed():
    lesson = {
        "explain": "यह एक नेटवर्क है।",
        "knowledge_check": {
            "question": "LAN क्या है?",
            "answer": "स्थानीय नेटवर्क।",
            "explanation": "यह पास के उपकरणों को जोड़ता है।",
        },
    }
    assert labels.parse_lesson_entries([lesson], "c1") == {"c1": lesson}
    assert labels.parse_lesson_entries([{"id": "c1", **lesson}], "other")["c1"]["explain"] == lesson["explain"]
    assert labels.parse_lesson_entries({"lesson1": lesson}, "c1") == {"lesson1": lesson}
    assert labels.parse_lesson_entries(lesson, "c1") == {"c1": lesson}
    assert labels.parse_lesson_entries({"id": "c1", "lesson": lesson}, "other") == {"c1": lesson}
    assert labels.parse_lesson_entries(json.dumps(lesson, ensure_ascii=False), "c1") == {"c1": lesson}
    assert labels.parse_lesson_entries(
        {"explain": {"c1": "यह एक नेटवर्क है।"},
         "knowledge_check": {"c1": lesson["knowledge_check"]}},
        "c1",
    ) == {"c1": {"explain": "यह एक नेटवर्क है।", "knowledge_check": lesson["knowledge_check"]}}


def test_rewritten_caption_sentences_are_not_accepted_as_concept_names():
    from app.pipeline.labels import clean_concept_label

    assert clean_concept_label("Yesterday we studied what a network is", "en") is None
    assert clean_concept_label("Whatever content I am speaking on the screen", "en") is None
    assert clean_concept_label("Study", "en") is None
    assert clean_concept_label("Local Area Network", "en") == "Local Area Network"
    assert clean_concept_label("राउटर", "hi") == "राउटर"


def test_unicode_tokens_keep_devanagari_marks_but_drop_danda():
    tokens = unicode_tokens("राउटरWAN TCP/IP फाइबर। यह एक नेटवर्क है।")
    assert "राउटरWAN" in tokens or "राउटरwan" in tokens
    assert any("फाइबर" == t for t in tokens)
    assert not any("।" in t for t in tokens)


# ---------------------------------------------------------------------------
# PHASE 4 - citation integrity
# ---------------------------------------------------------------------------
def test_every_external_note_has_verified_citation():
    for topic, config in TECHNICAL_CORRECTIONS.items():
        for note in config.get("notes", []):
            assert note["basis"] in REFERENCE_URLS, (topic, note["basis"])
            title, url = REFERENCE_URLS[note["basis"]]
            assert note["title"] == title and note["url"] == url
            assert url.startswith("https://")
            assert note["source"] == "external-supplement"
            assert note["supplemented"] is True


def test_no_blanket_rfc_for_geographic_classification():
    # LAN/PAN/MAN notes must not lean on RFC 1122 at all. WAN may keep one RFC
    # note for the separate "internet = network of networks" correction, but its
    # geographic classification must be supported by a dedicated source.
    for topic in ("lan", "pan", "man"):
        assert all(n["basis"] != "RFC 1122: internet is a network of networks"
                   for n in accuracy_notes_for(topic))
    wan_notes = accuracy_notes_for("wan")
    assert any(n["basis"] == "TechTarget: WAN (wide area network)" for n in wan_notes)
    bases = {n["basis"] for t in ("lan", "pan", "man", "wan") for n in accuracy_notes_for(t)}
    assert "IEEE 802 LAN/MAN Standards Committee" in bases
    assert "IEEE 802.15: Wireless Personal Area Networks" in bases
    assert "TechTarget: WAN (wide area network)" in bases


def test_submarine_and_osi_citations_match_claims():
    bases = {n["basis"] for n in accuracy_notes_for("submarine fibre cables")}
    assert "TeleGeography: Strait of Hormuz cable infrastructure" in bases
    assert "TeleGeography: submarine cable traffic fact-check" in bases
    osi = accuracy_notes_for("osi model")[0]
    assert osi["basis"] == "ITU-T X.200: OSI Basic Reference Model"
    assert "X.25" not in osi["correction"]


def test_hindi_language_gate_rejects_script_only_noun_piles():
    # Devanagari characters alone do not establish that learner-facing prose is Hindi.
    assert not field_language_ok("राउटर नेटवर्क पैकेट उपकरण डेटा प्रवाह व्यवस्था", "hi")
    assert field_language_ok("राउटर पैकेट को आगे भेजता है।", "hi")
    # A short technical title may be a noun phrase; prose fields may not use this exception.
    assert not field_language_ok("लैन नेटवर्क", "hi")
    assert field_language_ok("लैन नेटवर्क", "hi", allow_short_label=True)
    assert check_teaching_languages(_lesson(), "hi") == []


def test_deep_lesson_retry_includes_validation_feedback(monkeypatch):
    from app.pipeline import lesson_cache

    monkeypatch.setattr(lesson_cache, "load", lambda _key: None)
    monkeypatch.setattr(lesson_cache, "save", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(labels.time, "sleep", lambda _seconds: None)
    calls = []

    valid_lesson = {
        "title": "LAN",
        "explain": (
            "A LAN connects computers and nearby devices inside a limited area, such as a home or office. "
            "It lets connected devices communicate locally through switches or wireless access points."
        ),
        "intuition": (
            "Picture a LAN as the private corridors of one building: devices can pass messages directly "
            "without travelling out to the public street."
        ),
        "why_it_matters": (
            "This matters because nearby devices can share printers, files, and services quickly, "
            "even when the internet connection is unavailable."
        ),
        "steps": [
            "First, connect each device that belongs to the LAN to a switch or wireless access point.",
            "Next, configure each device with a valid local IP address so it can identify its peers.",
            "Finally, send a test message between two devices and confirm it stays on the local network.",
        ],
        "worked_example": (
            "In a small office, two laptops and a printer connect to the same switch. The laptops can "
            "print and share a folder over the LAN without sending local traffic to an outside service."
        ),
        "diagram": "[Laptop A] --- [Switch] --- [Printer]\n[Laptop B] ---/",
        "misconception": (
            "A common misconception is that a LAN must have internet access. Devices can still "
            "communicate with one another locally when the router's internet connection is down."
        ),
        "exercise": (
            "Draw a LAN for your home, label each connected device, and mark which devices can share "
            "a printer without internet access."
        ),
        "knowledge_check": {
            "question": "How do devices communicate on a LAN?",
            "answer": "Devices communicate locally.",
            "explanation": (
                "A LAN connects nearby devices through local equipment, allowing them to communicate "
                "locally without internet access."
            ),
        },
    }

    def fake_chat_json(_system, user, **_kwargs):
        request = json.loads(user)[0]
        calls.append(request)
        if len(calls) == 1:
            return {"c1": {"explain": "Too short."}}
        return {"c1": valid_lesson}

    monkeypatch.setattr(labels.llm, "chat_json", fake_chat_json)
    monkeypatch.setattr(
        labels.llm, "chat_json_diagnostics",
        lambda: {"provider": "test-provider", "status": 200, "retry_after": None,
                 "error": None, "error_kind": None},
    )
    node = {
        "id": "c1",
        "text": "LAN",
        "topic": "LAN",
        "start": 10.0,
        "end": 14.0,
        "source": "The office network links nearby computers and a printer.",
        "evidence": [{
            "start": 10.0,
            "end": 14.0,
            "text": "The office network links nearby computers and a printer.",
        }],
    }
    concepts = {"c1": labels.ConceptCand("c1", "s1", node)}
    accepted, failed, rejections = labels._llm_deep_lessons(
        {"meta": {"teachingLanguage": "en", "sourceLanguage": "en"}}, concepts
    )

    assert not failed, rejections
    assert "c1" in accepted
    assert len(calls) == 2
    assert "definition too short" in calls[1]["repairFeedback"]["previousErrors"]

