"""Small, dependency-free text helpers shared by every pipeline layer."""
from __future__ import annotations

import html
import json
import re
from typing import Any, Iterable

# Spoken-language filler that pollutes TF-IDF and node text.
FILLER_WORDS = {
    "um", "uh", "erm", "hmm", "mm", "mhm", "uhh", "umm", "ah", "oh", "okay", "ok", "yeah", "yep",
    "gonna", "wanna", "gotta", "kinda", "sorta", "like", "really", "actually", "basically",
    "literally", "right", "just", "so", "well", "thing", "things", "stuff", "lot", "lots",
    "going", "know", "mean", "said", "say", "says", "want", "get", "got", "let", "lets",
    "guys", "today", "video", "channel", "subscribe", "welcome", "thank", "thanks", "hey",
    "maybe", "probably", "pretty", "sure", "little", "bit", "way", "look", "looking", "make",
    "doing", "did", "does", "go", "goes", "come", "coming", "talk", "talking", "think", "kind",
    "sort", "able", "need", "use", "used", "using", "time", "good", "great", "new", "different",
    "gets", "getting", "makes", "means", "yes", "one", "two", "three", "first", "second",
    "instead", "example", "question", "point", "idea", "part", "number", "people", "everybody",
}

NOISE_RE = re.compile(
    r"\[(?:music|applause|laughter|laughs|inaudible|silence|__+|foreign|cheering)[^\]]*\]"
    r"|\((?:music|applause|laughs?|laughter|inaudible)\)",
    re.I,
)
SPEAKER_RE = re.compile(r"^\s*(?:>>|-\s+|\[?[A-Z][A-Za-z .]{1,24}\]?:\s)")
FILLER_RE = re.compile(r"\b(?:um+|uh+|erm|hmm+|you know,?|i mean,?)\b[,.]?\s*", re.I)
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


# --- light-weight noun-phrase heuristics (used when spaCy is not installed) ----------
_PURE_VERBS = """be have do say tell ask seem feel try leave call keep let begin show hear play run move live
believe hold bring happen write provide sit stand lose pay meet include continue set learn lead understand watch
follow stop create speak read allow add spend grow open walk win offer remember consider appear buy wait serve die
send expect build stay fall cut reach kill remain suggest raise pass sell require decide pull explain describe
destroy protect amplify interact determine explore rely collapse become give take find see come think know want
make get go put mean turn start help talk bring happen imagine realize notice figure apply compute solve prove
calculate store link cool represent contain involve depend enable produce reduce increase improve prevent cause
tells shows gives""".split()
_DUAL_VERBS = """search factor measure use work change need look answer test process design control plan report
record focus state claim model train question result value cost study share""".split()


def _verb_forms(base: str) -> set[str]:
    forms = {base}
    if base.endswith("y") and base[-2:-1] not in "aeiou":
        forms |= {base[:-1] + "ies", base[:-1] + "ied"}
    elif base.endswith(("s", "sh", "ch", "x", "z", "o")):
        forms |= {base + "es", base + "ed"}
    else:
        forms |= {base + "s", base + "d" if base.endswith("e") else base + "ed"}
    forms.add(base[:-1] + "ing" if base.endswith("e") and base != "be" else base + "ing")
    return forms


VERB_FORMS: set[str] = set()
for _v in _PURE_VERBS:
    VERB_FORMS |= _verb_forms(_v)
for _v in _DUAL_VERBS:
    VERB_FORMS |= _verb_forms(_v) - {_v}
VERB_FORMS |= {"is", "are", "was", "were", "been", "being", "has", "had", "does", "did", "done", "said", "told",
               "made", "took", "taken", "gave", "given", "found", "thought", "knew", "known", "went", "gone", "came",
               "saw", "seen", "got", "gotten", "led", "held", "kept", "left", "built", "sent", "spent", "lost", "won",
               "brought", "began", "begun", "wrote", "written", "ran", "stood", "can", "could", "would", "should",
               "will", "shall", "may", "might", "must", "called", "means"}

EDGE_ADJECTIVES = {"large", "small", "big", "hard", "easy", "fast", "slow", "important", "simple", "whole", "real",
                   "huge", "true", "possible", "able", "certain", "different", "same", "similar", "better", "best",
                   "bigger", "high", "low", "long", "short", "much", "many", "few", "several", "various", "entire",
                   "instantly", "exponentially", "faster", "slower", "separately", "famously", "near", "far"}

CLAUSE_SPLIT_RE = re.compile(r"[.,;:!?()\[\]\"“”…/]|\s[-–—]\s")
PHRASE_TOKEN_RE = re.compile(r"[a-z][a-z0-9+#\-]*(?:'s)?")


def candidate_phrases(text: str, stops: set[str], max_n: int = 3) -> list[str]:
    """
    Noun-phrase-like n-grams that never cross clause boundaries, never start/end with a
    stop word, never contain conjugated verbs and never end with a bare adjective/adverb.
    Usable directly as a scikit-learn `analyzer`.
    """
    out: list[str] = []
    for clause in CLAUSE_SPLIT_RE.split(text.lower()):
        toks = PHRASE_TOKEN_RE.findall(clause)
        for n in range(1, max_n + 1):
            for i in range(len(toks) - n + 1):
                gram = toks[i : i + n]
                first, last = gram[0], gram[-1]
                if first in stops or last in stops or last in EDGE_ADJECTIVES or first in EDGE_ADJECTIVES and n == 1:
                    continue
                if any(w in VERB_FORMS or w.endswith("ly") or len(w) < 2 for w in gram):
                    continue
                if any(w in stops and w != "of" for w in gram[1:-1]):
                    continue
                out.append(" ".join(gram))
    return out


def fmt_time(seconds: float | int | None) -> str:
    """83.2 -> '1:23', 3723 -> '1:02:03'."""
    if seconds is None:
        return ""
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def parse_time(value: str) -> float | None:
    """'1:02:03' / '12:30' -> seconds."""
    parts = value.strip().split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return None
    total = 0.0
    for n in nums:
        total = total * 60 + n
    return total


def clean_caption(text: str) -> str:
    text = html.unescape(html.unescape(text or ""))
    text = text.replace("\n", " ")
    text = NOISE_RE.sub(" ", text)
    text = re.sub(r"^\s*(?:>>|-)\s*", "", text)
    text = FILLER_RE.sub("", text)
    # Spoken punctuation read aloud by caption generators ("colon slash slash").
    text = re.sub(r"\bcolon\s+slash\s+slash\b", "://", text, flags=re.I)
    text = re.sub(r"\bcolon\s+slash\b", ":/", text, flags=re.I)
    text = re.sub(r"\bquestion\s+mark\b", "?", text, flags=re.I)
    text = re.sub(r"\bdot\s+com\b", ".com", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_speaker_change(raw_text: str) -> bool:
    return bool(SPEAKER_RE.match(html.unescape(raw_text or "")))


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in SENT_SPLIT_RE.split(text) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9+#\-']*[a-zA-Z0-9+#]|[a-zA-Z]", text.lower())


def truncate(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    space = cut.rfind(" ")
    if space > limit * 0.6:
        cut = cut[:space]
    return cut.rstrip(",;: ") + "…"


def smart_title(phrase: str) -> str:
    small = {"a", "an", "the", "of", "and", "or", "in", "on", "to", "for", "vs", "via", "with"}
    out = []
    for i, w in enumerate(phrase.split()):
        if w.isupper() and len(w) > 1:
            out.append(w)
        elif i and w.lower() in small:
            out.append(w.lower())
        else:
            out.append(w[:1].upper() + w[1:])
    return " ".join(out)


# ---------------------------------------------------------------------------
# Unicode-aware helpers shared by multilingual validation and similarity checks.
# ---------------------------------------------------------------------------
UNICODE_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
SUPPORTED_TEACHING_LANGUAGES = frozenset({"en", "hi"})

# Natural language is identified by its *grammar*, not by the fraction of
# letters in a script. A mostly-English explanation with a few Hindi nouns has
# English function words and no Hindi grammar; a natural Hinglish explanation
# keeps Hindi postpositions/auxiliaries while borrowing English technical nouns.
HINDI_GRAMMAR_WORDS = frozenset({
    # postpositions / auxiliaries / conjunctions
    "है", "हैं", "था", "थे", "थी", "हो", "होना", "होती", "होते", "हुआ", "हुई", "हुए",
    "से", "को", "में", "के", "की", "का", "पर", "तक", "और", "या", "कि", "क्योंकि",
    "लेकिन", "तो", "भी", "नहीं", "न", "यह", "ये", "वह", "वे", "यहाँ", "वहाँ",
    "जो", "जिस", "जिन", "क्या", "कैसे", "कब", "कहाँ", "जब", "तब", "फिर", "पहले",
    "एक", "दो", "कई", "बहुत", "सभी", "दूसरे", "दूसरा", "दोनों", "अपने", "अपनी",
    "रहा", "रही", "रहे", "लिए", "वाला", "वाली", "वाले", "जैसे", "यानी", "अर्थात",
    "मतलब", "ही", "तथा", "अंदर", "बाहर", "बीच", "कारण", "इसे", "उसे", "उन्हें",
    # common Hindi verb forms (strong grammatical signal even with English nouns)
    "कर", "करता", "करती", "करते", "करना", "करने", "किया", "जाता", "जाती", "जाते",
    "गया", "गई", "जोड़ता", "जोड़ती", "जोड़ने", "जुड़ता", "जुड़ती", "जुड़ने",
    "भेजता", "भेजती", "भेजने", "लेता",
    "देता", "मिलता", "चलता", "आता", "रखता", "दिखाता", "समझाता", "बताता", "पढ़ता",
    "लिखता", "बनाता", "सकता", "सकती", "चाहिए", "पहचानें", "समझें", "देखें",
    # Common imperative forms used in Hindi exercises and instructions.
    "बनाएँ", "बनाएं", "बनाइए", "खींचें", "खींचिए", "समझाएँ", "समझाएं", "समझाइए",
    "दिखाएँ", "दिखाएं", "दिखाइए", "पहचानिए", "जाँचें", "जांचें", "जाँचिए",
    "देखिए", "लिखें", "लिखिए", "आज़माएँ", "आजमाएं", "आज़माइए", "कीजिए",
    "कीजिये", "करें", "करिए", "बताइए", "बताएँ", "पढ़ें", "दें", "दीजिए",
})
ENGLISH_GRAMMAR_WORDS = frozenset({
    "a", "an", "the", "of", "to", "in", "on", "at", "for", "from", "by", "with",
    "and", "or", "but", "is", "are", "was", "were", "be", "been", "being", "am",
    "this", "that", "these", "those", "it", "its", "as", "into", "through",
    "between", "each", "every", "which", "who", "what", "when", "where", "how",
    "can", "will", "would", "should", "may", "might", "must", "do", "does", "did",
    "not", "no", "if", "then", "than", "so", "such", "because", "while", "about",
    "over", "under", "out", "up", "down", "more", "most", "other", "another",
    "own", "same", "each", "both", "few", "many", "much", "only", "also", "here",
    "there", "these", "those",
})


def has_devanagari(text: str) -> bool:
    return any("\u0900" <= ch <= "\u097f" for ch in str(text or ""))


def normalize_language(code: str | None) -> str:
    """Normalise BCP-47 codes to the base language (hi-IN -> hi, en-US -> en)."""
    base = str(code or "en").strip().lower().replace("_", "-").split("-")[0]
    return base or "en"


def unicode_tokens(text: str) -> list[str]:
    """Word tokens that preserve Devanagari marks and every other Unicode script."""
    # \w excludes combining marks (Devanagari matras), so include the Devanagari
    # block explicitly. lower() keeps scripts intact where casefold() decomposes them.
    # U+0964/U+0965 are the danda sentence markers; they live inside the
    # Devanagari block but are punctuation, not part of a word token.
    cleaned = re.sub(r"[^\w\u0900-\u0963\u0966-\u097f]+", " ", str(text or "").lower()).strip()
    return cleaned.split() if cleaned else []


def unicode_norm(text: str) -> str:
    """Normalised text for similarity/overlap checks; never strips non-Latin scripts."""
    return " ".join(unicode_tokens(text))


def script_counts(text: str) -> tuple[int, int, int]:
    """(Devanagari, Latin, other) letter counts."""
    dev = latin = other = 0
    for ch in str(text or ""):
        if not ch.isalpha():
            continue
        if "\u0900" <= ch <= "\u097f":
            dev += 1
        elif ch.isascii():
            latin += 1
        else:
            other += 1
    return dev, latin, other


def is_technical_only(text: str) -> bool:
    """True when a field is just acronyms/canonical terms (no prose to translate)."""
    from .terminology import ACRONYM_EXPANSIONS

    tokens = unicode_tokens(text)
    if not tokens:
        return False
    known = {"lan", "pan", "man", "wan", "isp", "isps", "cdn", "cdns", "osi", "arpanet",
             "tcp", "ip", "udp", "ont", "http", "https", "url", "uri", "wifi", "wi",
             "fi", "fibre", "fiber", "router", "internet", "network", "ethernet"}
    for expansion in ACRONYM_EXPANSIONS.values():
        known.update(unicode_tokens(expansion))
    return all(t.isdigit() or t in known for t in tokens)


def field_language_ok(text: str, language: str, *, allow_short_label: bool = False) -> bool:
    """Check one learner-facing field against the requested teaching language.

    A script-percentage threshold is not enough: "a राउटर is a device that
    forwards packets" is English with one Hindi noun, while "राउटर पैकेट को
    एक नेटवर्क से दूसरे नेटवर्क तक ले जाता है" is Hindi grammar carrying English
    technical vocabulary. We therefore require Hindi grammatical markers and
    require them to dominate English grammatical markers.
    """
    lang = normalize_language(language)
    if lang not in SUPPORTED_TEACHING_LANGUAGES:
        return False
    if is_technical_only(text):
        return True
    dev, latin, other = script_counts(text)
    letters = dev + latin + other
    if not letters:
        return False
    tokens = unicode_tokens(text)
    if not tokens:
        return False
    if lang == "hi":
        if dev == 0:
            return False
        hindi_markers = sum(t in HINDI_GRAMMAR_WORDS for t in tokens)
        english_markers = sum(t in ENGLISH_GRAMMAR_WORDS for t in tokens)
        ratio = dev / letters
        # Prose must show Hindi grammar. Script alone is not proof of Hindi:
        # random Devanagari words and noun piles are not accepted as explanations.
        if hindi_markers >= 2 and hindi_markers >= english_markers and ratio >= 0.15:
            return True
        # Short labels may be noun phrases, but this exception is enabled only
        # for the title field by check_teaching_languages().
        if (allow_short_label and len(tokens) <= 4 and len(str(text).strip()) <= 60
                and ratio >= 0.55 and english_markers == 0):
            return True
        return False
    return latin > 0 and latin / letters >= 0.8


def check_teaching_languages(lesson: dict[str, Any], language: str) -> list[str]:
    """Validate every learner-facing teaching field with one shared rule set.

    Diagrams, provenance, evidence, quotations and technical acronyms are exempt
    from ordinary-language script requirements.
    """
    lang = normalize_language(language)
    if lang not in SUPPORTED_TEACHING_LANGUAGES:
        return [f"unsupported teaching language for validation: {language}"]
    errors: list[str] = []
    fields: list[tuple[str, Any]] = [
        (name, lesson.get(name)) for name in
        ("title", "explain", "intuition", "why_it_matters", "worked_example", "misconception", "exercise")
    ]
    steps = lesson.get("steps")
    if isinstance(steps, list):
        fields.extend((f"steps[{i}]", step) for i, step in enumerate(steps))
    check = lesson.get("knowledge_check")
    if isinstance(check, dict):
        fields.extend((f"knowledge_check.{key}", check.get(key))
                      for key in ("question", "answer", "explanation"))
    for name, value in fields:
        if not isinstance(value, str) or not value.strip():
            continue
        if not field_language_ok(value, lang, allow_short_label=(name == "title")):
            errors.append(f"{name} not written in the teaching language")
    return errors


def extract_json(text: str) -> Any:
    """Parse the first JSON object/array in an LLM response (tolerates ```json fences)."""
    if not text:
        raise ValueError("empty LLM response")
    text = text.strip()
    # Some gateways wrap the answer in a reasoning block; it is not part of the JSON.
    text = re.sub(r"<(?:think|thinking|reasoning)>.*?</(?:think|thinking|reasoning)>",
                  "", text, flags=re.S | re.I).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try every candidate opener: a reasoning sentence before the answer can
    # contain a lone "{" that would otherwise abort the scan with unbalanced JSON.
    starts = sorted(i for i in (text.find("{"), text.find("[")) if i >= 0)
    starts += [m.start() for m in re.finditer(r"[\[{]", text) if m.start() not in starts]
    if not starts:
        raise ValueError("no JSON found in LLM response")
    for start in starts:
        opener = text[start]
        closer = "}" if opener == "{" else "]"
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError("unbalanced JSON in LLM response")


def dedupe_keep_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for item in items:
        key = item.lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def parse_description_chapters(description: str, duration: float | None = None) -> list[dict]:
    """YouTube chapters are '0:00 Intro' lines in the description (the first must be 0:00)."""
    chapters = []
    pattern = re.compile(r"^\s*[\(\[]?((?:\d{1,2}:)?\d{1,2}:\d{2})[\)\]]?\s*[-–—:|]?\s*(.+?)\s*$")
    for line in (description or "").splitlines():
        m = pattern.match(line)
        if m:
            t = parse_time(m.group(1))
            if t is not None:
                chapters.append({"title": m.group(2).strip(" -–—"), "start": t})
    if len(chapters) < 2 or chapters[0]["start"] > 1:
        return []
    chapters.sort(key=lambda c: c["start"])
    for i, ch in enumerate(chapters):
        ch["end"] = chapters[i + 1]["start"] if i + 1 < len(chapters) else (duration or ch["start"] + 60)
    return chapters
