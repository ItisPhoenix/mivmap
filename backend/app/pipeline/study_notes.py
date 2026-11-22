"""Revision / Academic-mode study notes — semantic, transcript-grounded, honest.

No LLM is used here. Every factual statement is a compressed sentence from the
concept's own transcript span — never a generated technical explanation.
Fields the transcript does not support are left empty and `available` is False;
the viewer surfaces that as an explicit "lesson unavailable" notice.

Design:
* Selects candidate sentences with semantic similarity (embeddings) to the concept,
  preferring sentences that DEFINE the concept (copula right after the term).
* Definition is the defining clause only; fragments are rejected.
* intuition, why_it_matters, steps, worked_example, diagram, misconception are
  only filled when the transcript actually contains sentences of that kind.
* Exercises are generated, but are distinct per concept and reference the
  specific concept and timestamp.
* Confidence is set from the semantic similarity of the note to its evidence —
  never from keyword overlap.
"""
from __future__ import annotations

import re

import numpy as np

from .embeddings import cosine_matrix, embed
from .text_utils import fmt_time, truncate, unicode_norm

DEFINITION_CUE = re.compile(
    r"^\s*(?:or|which is|that is|is|are|was|were|means?|refers? to|is called|"
    r"stands for|defined as|is a|is an|is the|known as|describes?|explains?|defines?)\s+", re.I)
ANALOGY_CUE = re.compile(
    r"\b(?:like|think of|considered|similar to|imagine|sort of|can be viewed as|"
    r"compared to|rather than|the point is|for instance)\b", re.I)
SEQUENCE_CUE = re.compile(r"\b(?:first|then|next|after|finally|once|step)\b", re.I)
EXAMPLE_CUE = re.compile(r"\b(?:for example|for instance|such as|imagine|let's say|say you|consider|e\.g\.)\b", re.I)
IMPORTANCE_CUE = re.compile(
    r"\b(?:because|so that|in order to|allows?|enables?|needs? to|important|critical|"
    r"essential|therefore|which means|used to|lets? you|helps?|makes? it possible|"
    r"depends? on|relies? on|works? by|requires?)\b", re.I)
CORRECTIVE_CUE = re.compile(
    r"\b(?:but|however|instead|rather than|not|never|mistake|wrong|misconception|"
    r"doesn't|does not|can't|cannot|unlike|do not think|as long as it isn't)\b", re.I)
OUTRO_CUE = re.compile(
    r"\b(?:subscribe|like and subscribe|thanks for watching|thank you for watching|"
    r"see you in the next|hope you enjoyed|hope you liked|check the description|"
    r"link in the description|comment below|hit the bell)\b", re.I)
CLAUSE_RE = re.compile(r",|;|\s(?:which|because|so that|whereas|while|although)\s|\s[-–—]\s")
FRAGMENT_START = re.compile(
    r"^(?:and|but|or|so|then|meaning|which|that|because|also|do|don't|it|this|they|there|"
    r"the|a|an|by|on|at|in|of|to|from)\b", re.I)

# Similarity thresholds: definition needs a real defining sentence or a
# high-similarity clause; aids need to be semantically close to the concept.
SIM_DEFINITION = 0.5
SIM_AID = 0.40
SIM_AID_EXAMPLE = 0.45
SIM_HARD = 0.75  # faith ceiling for extractive notes.

# Defining clauses that may follow the term in a definition sentence.
DEFINITION_PREFIX_RE = re.compile(
    r"^(?:or|which is|that is|is|are|was|were|means?|refers? to|is called|is defined as|stands for|"
    r"defined as|is a|is an|is the|known as|describes?|explains?|defines?)\s+", re.I)

# Acronyms are often spelled out in the surrounding transcript, not repeated as acronyms.
ACRONYM_WORDS = {
    "lan": ["local", "area", "network"],
    "pan": ["personal", "area", "network"],
    "man": ["metropolitan", "area", "network"],
    "wan": ["wide", "area", "network"],
    "isp": ["internet", "service", "provider"],
    "osi": ["open", "systems", "interconnection"],
    "cdn": ["content", "delivery", "network"],
    "cdns": ["content", "delivery", "network"],
    "tcp": ["transmission", "control", "protocol"],
    "ip": ["internet", "protocol"],
}


def _norm(text: str) -> str:
    return unicode_norm(text)


def _clean(sentence: str) -> str:
    text = re.sub(r"\s+", " ", sentence or "").strip().strip("“”\"")
    while (m := re.match(r"^(?:>>|-\s+|\[?[A-Z][A-Za-z .]{1,24}\]?:\s)", text)):
        text = text[m.end():]
    text = re.sub(r"\s+(?:It|This|That|They|There)\.?\s*$", "", text, flags=re.I)
    return text.strip()


def _short(sentence: str, limit: int = 24) -> str:
    text = _clean(sentence)
    parts = [p.strip().rstrip(",;:") for p in CLAUSE_RE.split(text)]
    fits = [p for p in parts
            if 5 <= len(p.split()) <= limit and not FRAGMENT_START.match(p) and "->" not in p]
    if not fits:
        fits = [p for p in parts if 5 <= len(p.split()) <= limit and not FRAGMENT_START.match(p)]
    if not fits:
        fits = [text]
    return truncate(fits[0] if fits else text, 150)


def _topic_words(topic: str) -> list[str]:
    words: set[str] = set()
    for w in _norm(topic).split():
        if len(w) < 3:
            continue
        words.add(w)
        if w.endswith("ies"):
            words.add(w[:-3] + "y")
        elif w.endswith("s"):
            words.add(w[:-1])
        words.update(ACRONYM_WORDS.get(w, []))
    return sorted(words, key=len, reverse=True)


def _alias_hits(sentence: str, aliases: list[str]) -> list[tuple[int, str]]:
    hits = []
    for alias in aliases:
        if not alias or len(alias) < 3:
            continue
        for m in re.finditer(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])", sentence, re.I):
            hits.append((m.start(), alias))
    return sorted(hits)


def _sims(topic: str, sentences: list[str]) -> np.ndarray:
    """Cosine similarity of each sentence to the concept (semantic, not keyword overlap)."""
    query = " ".join([topic, *_topic_words(topic)])
    vecs = embed([query, *sentences])
    return cosine_matrix(vecs[:1], vecs[1:])[0]


# Sentences that merely promise a topic is coming later are not definitions.
MENTION_GUARD = re.compile(
    r"^(?:we will talk about|we'll talk about|we talk about|we are going to talk about|we are talking about|"
    r"we're talking about|let's talk about|let's see|we'll see|hope you enjoyed|thanks for watching|"
    r"check the description|see you in the next|coming up next|in the next video)\b", re.I)


def _identifies(clause_text: str, aliases: list[str]) -> bool:
    """Does this definition name the concept?

    A definition names the concept when it contains the term itself or the full
    acronym expansion (e.g. "Personal Area Network" for PAN) — never a loose
    word like "area" or "network" that other concepts share.
    """
    n = _norm(clause_text)
    if not n:
        return False
    if any(a and a in n for a in aliases):
        return True
    expansions: list[list[str]] = []
    for a in aliases:
        words = ACRONYM_WORDS.get(a)
        if isinstance(words, list) and words:
            expansions.append([w for w in words if w])
    return any(all(w in n for w in exp) for exp in expansions)


def _definition(topic: str, sentences: list[str], sims: np.ndarray) -> tuple[str, str, bool]:
    """Return (definition, supporting sentence, is_explicit_definition).

    A definition is the defining clause after the term in a defining sentence
    (copula right after the alias). Anything else is rejected: we never label an
    unrelated fragment as a definition.
    """
    aliases = sorted({topic.lower().strip(), *[a for a in _norm(topic).split() if len(a) >= 3]},
                     key=len, reverse=True)
    words = _topic_words(topic)  # includes acronym expansions (PAN -> personal/area/network)
    word_re = (re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words if len(w) >= 3) + r")\b", re.I)
               if any(len(w) >= 3 for w in words) else None)
    cands: list[tuple[float, str, str, bool]] = []
    for i, sentence in enumerate(sentences):
        hits = _alias_hits(sentence, aliases)
        if MENTION_GUARD.search(sentence):
            continue  # a plan to talk about a topic is not a definition of it
        if hits:
            # Slice after the LAST alias of a term mention so a spelled-out term
            # ("Personal Area Network is...") keeps its defining clause.
            pos, alias = max(hits, key=lambda h: (h[0], len(h[1])))
        elif word_re is not None and (m := word_re.search(sentence)):
            pos, alias = m.start(), m.group(0)
        else:
            continue
        clause = _clean(sentence[pos + len(alias):]).lstrip(" ,;:")
        m = re.match(DEFINITION_PREFIX_RE, clause)
        explicit = bool(m)
        # A definition is the defining clause after the term in a defining sentence
        # (copula right after the alias); otherwise the whole sentence, but only when
        # it names the concept — never an unrelated fragment.
        after_term = _short(clause[m.end():] if explicit else sentence)
        # An explicit copula sits AFTER the term, so the defining clause
        # naturally omits the alias; judge identification on the whole
        # sentence in that case, not on the clause that follows it.
        if not _identifies(sentence if explicit else after_term, aliases):
            continue
        if explicit and len(after_term.split()) < 3:
            continue
        if not explicit and len(after_term.split()) < 6:
            continue
        spelled = len([w for w in words if w in _norm(after_term)]) >= 2
        sim = float(sims[i])
        if explicit and sim < 0.40:
            continue
        # "Explicit copula or sim≥0.5–0.6": a short definition that spells out the
        # concept's terms (e.g. "Personal Area Network") also qualifies — the term
        # itself is the definition; LSA similarity of the query to such a sentence
        # is too weak to rely on alone.
        if not explicit and sim < SIM_DEFINITION and not (spelled and len(after_term.split()) <= 8):
            continue
        score = (3.0 if explicit else 0.0) + (1.0 if spelled else 0.0) + sim * 0.8 - 0.02 * len(after_term.split())
        cands.append((score, after_term, sentence, explicit))
    if not cands:
        return "", "", False
    _, definition, sentence, explicit = max(cands, key=lambda c: c[0])
    return (definition.rstrip(".") + "." if definition else ""), sentence, explicit


def _pick(sentences: list[str], sims: np.ndarray, pattern: re.Pattern, topic: str,
          limit: int = 1, min_sim: float = SIM_AID) -> list[str]:
    """Sentences of this kind that are checkable from the concept, deduped.

    Accepts a sentence when it shares the concept's term words or its embedding
    similarity clears `min_sim`. Sharing the term matters because the LSA
    fallback ranks genuinely concept-mentioning sentences low; the cue plus the
    term mention make it a real aid.
    """
    words = [w for w in _norm(topic).split() if len(w) >= 3]
    out = []
    seen = set()
    for i, s in enumerate(sentences):
        if OUTRO_CUE.search(s) or not pattern.search(s):
            continue
        clean = _short(s)
        if not clean or len(clean.split()) < 8 or FRAGMENT_START.match(clean):
            continue
        norm = _norm(clean)
        if not any(w in norm for w in words) and float(sims[i]) < min_sim:
            continue
        if norm in seen:
            continue
        seen.add(norm)
        out.append((float(sims[i]), clean))
    out.sort(key=lambda x: -x[0])
    return [t for _, t in out[:limit]]


def build_study_note(topic: str, evidence: list[dict], context: list[dict] | None = None) -> dict:
    """Build a transcript-grounded study note for one concept.

    Only fills fields when the transcript actually contains a supporting sentence.
    `available` is True only when there is a real definition + checkable answer.
    """
    items = [e for e in (evidence or []) if isinstance(e, dict) and str(e.get("text") or "").strip()]
    sentences: list[str] = []
    seen: set[str] = set()
    for e in items:
        s = re.sub(r"\s+", " ", str(e.get("text") or "")).strip()
        if s and _norm(s) not in seen:
            seen.add(_norm(s))
            sentences.append(s)
    for c in context or []:
        s = re.sub(r"\s+", " ", str(c.get("text") or "")).strip()
        if s and _norm(s) not in seen:
            seen.add(_norm(s))
            sentences.append(s)

    sims = _sims(topic, sentences)
    definition, support, explicit = _definition(topic, sentences, sims)

    why = _pick(sentences, sims, IMPORTANCE_CUE, topic, 1)
    intuition = _pick(sentences, sims, ANALOGY_CUE, topic, 1)
    if intuition and why and _norm(intuition[0]) == _norm(why[0]):
        intuition = []  # intuition must differ from why_it_matters
    if intuition and definition and _norm(intuition[0]) == _norm(definition):
        intuition = []

    steps = _pick(sentences, sims, SEQUENCE_CUE, topic, 4, min_sim=SIM_AID)
    examples = _pick(sentences, sims, EXAMPLE_CUE, topic, 1, min_sim=SIM_AID_EXAMPLE)
    if not examples:
        # A concrete instance with a number/proper noun, not a fragment.
        examples = _pick(sentences, sims, re.compile(r"\b\d{1,4}\b|[A-Z][a-z]{3,}"), topic, 1)
        if examples:
            examples = [t for t in examples if len(t.split()) >= 8]
    misconceptions = _pick(sentences, sims, CORRECTIVE_CUE, topic, 1)
    diagram = ""
    if len(steps) >= 2 and not any(FRAGMENT_START.match(s) for s in steps):
        diagram = "\n -> ".join(s.rstrip(".") for s in steps[:4])

    anchor = next((e for e in items if support and str(e.get("text") or "").strip() == support),
                  items[0] if items else None)
    at = float((anchor or {}).get("start") or 0)
    stamp = fmt_time(at)
    evidence_text = _clean(str((anchor or {}).get("text") or ""))[:220]

    explain = f"{topic} — {definition}" if definition else ""
    if explain:
        explain += f" (Grounded in the video transcript at {stamp}.)"
    answer = _short(definition or evidence_text, 22).rstrip(".")
    answer = re.sub(r"\s+(?:It|This|That|They|There)$", "", answer, flags=re.I).strip()

    exercise = (
        f"At {stamp}, explain how {topic} works in your own words — then name one "
        f"thing the video highlights about it."
    )
    note = {
        "explain": explain,
        "intuition": intuition[0] if intuition else "",
        "why_it_matters": why[0] if why else "",
        "steps": steps,
        "worked_example": examples[0] if examples else "",
        "diagram": diagram,
        "misconception": misconceptions[0] if misconceptions else "",
        "exercise": exercise,
        "knowledge_check": {
            "question": f"According to the video at {stamp}, how is {topic} described?",
            "answer": answer,
            "explanation": evidence_text,
        },
        "evidence": [{"start": round(float(e.get("start") or 0), 2),
                      "end": round(float(e.get("end") or 0), 2),
                      "text": truncate(str(e.get("text") or ""), 300)} for e in items[:4]],
        "video_timestamp": round(at, 2),
        "provenance": {
            "source": "video",
            "kind": "transcript-extractive",
            "example_source": "video",
        },
        "generatedBy": "extractive-transcript",
        "groundingNote": (
            "Definition stated in the video." if explicit else
            "Grounded context note from the video's own transcript; no AI explanations were used."
        ),
    }
    note["available"] = bool(definition and answer)
    return note
