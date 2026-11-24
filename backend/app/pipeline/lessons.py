"""Domain-independent validation for Deep Mode concept lessons.

A lesson is only accepted when every teaching field is substantive, grounded in the
concept's own transcript evidence, and clearly separates video claims from supplemental
teaching. Nothing here assumes a particular subject.
"""
from __future__ import annotations

import difflib
import re
from typing import Any

from .text_utils import check_teaching_languages, field_language_ok, unicode_norm

LESSON_FIELDS = (
    "explain",          # definition / plain-English meaning
    "intuition",        # mental model in everyday language
    "why_it_matters",   # relevance to a beginner
    "steps",            # ordered mechanism (list[str])
    "worked_example",   # concrete, specific, supplemental
    "diagram",          # meaningful multi-line text diagram
    "misconception",    # common wrong idea + correction
    "exercise",         # practical task
    "knowledge_check",  # {"question": ..., "answer": ..., "explanation": ...}
)

GENERIC_PHRASES = re.compile(
    r"\b(?:a concept covered|key idea introduced|available information|"
    r"component that can be part of|important to understand|this is a topic|"
    r"in this section|the speaker|as mentioned earlier)\b",
    re.I,
)

ACTION_VERBS = re.compile(
    r"\b(?:write|draw|explain|predict|trace|identify|list|compare|build|try|which|what|would|should|choose|select|"
    r"implement|sketch|describe|calculate|check|test|run|create)\b"
    # Hindi teaching: common imperative forms used in a practical exercise.
    r"|(?:बनाएँ|बनाएं|बनाइए|खींचें|खींचिए|समझाएँ|समझाएं|समझाइए|दिखाएँ|दिखाएं|दिखाइए|"
    r"पहचानें|पहचानिए|सूची|तुलना|गणना|वर्णन|जाँचें|जांचें|जाँचिए|देखें|देखिए|लिखें|लिखिए|"
    r"चित्र|आज़माएँ|आजमाएं|आज़माइए|निर्धारित|पता\s*लगाएँ|पता\s*लगाएं|कीजिए|कीजिये|"
    r"करें|करिए|बताइए|बताएँ|पढ़ें|नाम\s*दें|उदाहरण\s*दें)",
    re.I,
)


def _norm(text: str) -> str:
    return unicode_norm(text)


def _topic_terms(topic: str) -> set[str]:
    """Concept-name variants a lesson may legitimately use (plural + acronym expansion)."""
    terms: set[str] = set()
    for w in _norm(topic).split():
        if len(w) < 3:
            continue
        terms.add(w)
        if w.endswith("ies"):
            terms.add(w[:-3] + "y")
        elif w.endswith("s"):
            terms.add(w[:-1])
    from .study_notes import ACRONYM_WORDS
    from .terminology import HINDI_TERM_HINTS
    for w in list(terms):
        terms.update(ACRONYM_WORDS.get(w, []))
    # Hindi teaching keeps English acronyms but may also use the spoken Hindi
    # technical term. Compare whole-topic and multi-word canonical phrases too:
    # "Fibre cable" -> "फाइबर", "Internet (network of networks)" -> "इंटरनेट".
    whole = unicode_norm(topic)
    normalized_terms = {unicode_norm(t) for t in terms}
    whole_parts = set(whole.split())
    for hint, canonical in HINDI_TERM_HINTS.items():
        cnorm = unicode_norm(canonical)
        cparts = set(cnorm.split())
        if (cnorm == whole or cnorm in normalized_terms or (cparts and cparts <= whole_parts)):
            terms.add(hint.lower())
    return terms


def overlap(text: str, evidence: str) -> float:
    """Similarity to the source passage; high overlap means a copied sentence."""
    if not text or not evidence:
        return 0.0
    return difflib.SequenceMatcher(None, _norm(text), _norm(evidence)).ratio()


def _script_ok(text: str, language: str) -> bool:
    """Shared script check; technical-only fields are exempt by design."""
    return field_language_ok(text, language)


def validate_deep_teaching(lesson: dict[str, Any], topic: str,
                           teaching_language: str = "en") -> tuple[bool, list[str]]:
    """Check the study-readiness fields that `validate_lesson` does not cover.

    These are stamped deterministically by the pipeline so source transparency is
    never left to the model. A lesson that cannot state where its claims come
    from is not study-ready.
    """
    from .terminology import acronym_expansion, correction_required

    errors: list[str] = []
    if not isinstance(lesson, dict):
        return False, ["lesson is not an object"]
    prov = lesson.get("provenance")
    if not isinstance(prov, dict):
        errors.append("three-way provenance missing")
    else:
        for key in ("video_supported", "ai_explanation", "external_supplement"):
            if not isinstance(prov.get(key), list):
                errors.append(f"provenance.{key} missing")
        if not prov.get("ai_explanation") and not prov.get("video_supported"):
            errors.append("provenance has no sources")
    expected = acronym_expansion(topic)
    if expected and str(lesson.get("acronym_expansion") or "").strip().lower() != expected.lower():
        errors.append(f"acronym expansion wrong or missing (expected {expected})")
    if correction_required(topic) and not lesson.get("accuracy_notes"):
        errors.append("externally supplemented accuracy note missing")
    errors.extend(check_teaching_languages(lesson, teaching_language))
    return (not errors), errors


def stamp_lesson(lesson: dict[str, Any], topic: str,
                 evidence: list[dict] | None = None,
                 teaching_language: str = "en",
                 expansion: str = "") -> dict[str, Any]:
    """Stamp deterministic source attribution onto a validated lesson.

    Shared by fresh generation and cache recovery so a recovered lesson carries
    exactly the same provenance, accuracy corrections and grounding metadata as
    a newly generated one.
    """
    from .terminology import apply_accuracy_corrections

    out = apply_accuracy_corrections(dict(lesson), topic)
    ev = out.get("evidence")
    if not isinstance(ev, dict) or not (ev.get("timestamps") or ev.get("starts")):
        stamps = sorted({round(float(e.get("start") or 0), 2)
                         for e in (evidence or []) if e.get("start") is not None})
        out["evidence"] = {"timestamps": stamps, "quotes": []}
    external = ["accuracy_notes"] if out.get("accuracy_notes") else []
    out["provenance"] = {
        "source": "video+ai+external" if external else "video+ai",
        "video_supported": ["explain", "lesson_evidence"],
        "ai_explanation": ["intuition", "why_it_matters", "steps", "worked_example",
                           "diagram", "misconception", "exercise", "knowledge_check"],
        "external_supplement": external,
        "example_source": "ai-generated",
    }
    if expansion:
        out["acronym_expansion"] = expansion
    out["teaching_language"] = teaching_language
    return out


KNOWLEDGE_CHECK_ERRORS = frozenset({
    "knowledge check missing",
    "knowledge check too short",
    "knowledge check explanation missing",
    "knowledge check answer not supported by the lesson",
})


def knowledge_check_only(errors: list[str] | tuple[str, ...]) -> bool:
    """True when every validation error concerns only the knowledge_check field.

    This is the precondition for a bounded, targeted repair of that one field:
    an otherwise fully validated lesson must not be discarded over a single
    retrieval-phrasing mistake, but a lesson with any other defect is still
    rejected in full.
    """
    return bool(errors) and all(str(e) in KNOWLEDGE_CHECK_ERRORS for e in errors)


def validate_lesson(lesson: dict[str, Any], evidence_text: str, topic: str,
                    evidence: list[dict] | None = None,
                    teaching_language: str | None = None) -> tuple[bool, list[str]]:
    """Return (accepted, errors). No subject-specific facts are consulted here.

    The optional `evidence` list enables timestamp/quote checks against the concept's
    own transcript spans; every pedagogical gate is domain independent.
    """
    errors: list[str] = []
    if not isinstance(lesson, dict):
        return False, ["lesson is not an object"]

    def field(name: str) -> Any:
        return lesson.get(name)

    def text_value(name: str) -> str:
        value = field(name)
        return value.strip() if isinstance(value, str) else ""

    explain = text_value("explain")
    intuition = text_value("intuition")
    why = text_value("why_it_matters")
    steps = field("steps")
    example = text_value("worked_example")
    diagram = text_value("diagram")
    misconception = text_value("misconception")
    exercise = text_value("exercise")
    check = field("knowledge_check")

    if len(explain) < 50:
        errors.append("definition too short")
    if GENERIC_PHRASES.search(explain + " " + why):
        errors.append("generic placeholder wording")
    if topic and not any(w in _norm(explain) for w in _topic_terms(topic)):
        errors.append("definition does not name the concept")
    if overlap(explain, evidence_text) > 0.78:
        errors.append("definition copies the transcript")

    if len(why) < 40:
        errors.append("why_it_matters too short")
    if not isinstance(steps, list) or len(steps) < 3:
        errors.append("fewer than three mechanism steps")
    else:
        for i, step in enumerate(steps):
            if not isinstance(step, str) or len(step.strip()) < 18:
                errors.append(f"step {i + 1} too short")

    if len(example) < 50:
        errors.append("worked example too short")
    if len(diagram) < 20 or not re.search(r"->|<->|--|\[|\]|\|", diagram):
        errors.append("diagram is not a meaningful multi-line flow")

    if len(misconception) < 40:
        errors.append("misconception too short")
    if len(exercise) < 30:
        errors.append("exercise too short")
    if not isinstance(check, dict):
        errors.append("knowledge check missing")
    else:
        question = str(check.get("question") or "").strip()
        answer = str(check.get("answer") or "").strip()
        if len(question) < 12 or len(answer) < 8:
            errors.append("knowledge check too short")
        check_explanation = str(check.get("explanation") or "").strip()
        if len(check_explanation) < 20:
            errors.append("knowledge check explanation missing")
        # A learner must be able to answer using only the supplied lesson text.
        content = _norm(" ".join([explain, intuition, why,
                                  " ".join(steps) if isinstance(steps, list) else "", example]))
        answer_words = [w for w in _norm(answer).split() if len(w) > 3]
        if answer_words and not check_explanation:
            errors.append("knowledge check answer not supported by the lesson")
        elif answer_words and sum(1 for w in answer_words if w in content) / len(answer_words) < 0.6:
            errors.append("knowledge check answer not supported by the lesson")

    if len(intuition) < 40:
        errors.append("intuition too short")
    topic_terms = _topic_terms(topic)
    step_text = _norm(" ".join(steps) if isinstance(steps, list) else "")
    if topic_terms and not any(w in step_text for w in topic_terms):
        errors.append("steps are not specific to the concept")
    if topic_terms and not any(w in _norm(exercise) for w in topic_terms):
        errors.append("exercise does not name the concept")
    if not ACTION_VERBS.search(exercise):
        errors.append("exercise has no actionable verb")

    # Repetitions / polished-but-empty lessons: near-identical teaching fields.
    bodies = [b for b in (explain, intuition, why, example, misconception, exercise) if b]
    for i in range(len(bodies)):
        for j in range(i + 1, len(bodies)):
            if overlap(bodies[i], bodies[j]) > 0.85:
                errors.append(f"repeated teaching text ({i + 1}/{j + 1})")

    # Distinctness: the exercise must not duplicate the worked example, the
    # intuition must not duplicate the why, and the exercise must not be the
    # same task as the intuition or the definition.
    if example and exercise and overlap(exercise, example) > 0.75:
        errors.append("exercise repeats the worked example")
    if intuition and exercise and overlap(exercise, intuition) > 0.8:
        errors.append("exercise repeats the intuition")
    if intuition and why and overlap(intuition, why) > 0.75:
        errors.append("intuition repeats why_it_matters")
    if explain and exercise and overlap(exercise, explain) > 0.85:
        errors.append("exercise repeats the definition")

    if evidence:
        # Timestamps must land on real transcript positions: any time within an
        # evidence span (its start or end, with tolerance) is acceptable. The
        # model often reports the end of a passage or a neighbouring entry.
        spans: list[tuple[float, float]] = []
        for e in evidence:
            if not isinstance(e, dict):
                continue
            try:
                start = float(e.get("start") or 0)
            except (TypeError, ValueError):
                start = 0.0
            try:
                end = float(e.get("end")) if e.get("end") is not None else start + 1.0
            except (TypeError, ValueError):
                end = start + 1.0
            spans.append((start, max(end, start)))
        supplied = _norm(" ".join(str(e.get("text") or "") for e in evidence if isinstance(e, dict)))
        ev_field = lesson.get("evidence") or {}
        if isinstance(ev_field, dict):
            stamps = ev_field.get("timestamps") or ev_field.get("starts") or []
            quotes = ev_field.get("quotes") or []
            for at in stamps if isinstance(stamps, list) else []:
                try:
                    at_f = float(at)
                except (TypeError, ValueError):
                    errors.append("invalid evidence timestamp")
                    continue
                if not any(s - 2.0 <= at_f <= e + 2.0 for s, e in spans):
                    errors.append("evidence timestamp not in transcript span")
            for q in quotes if isinstance(quotes, list) else []:
                if len(str(q)) >= 12 and _norm(q) not in supplied:
                    errors.append("evidence quote not in transcript")

    if teaching_language:
        errors.extend(check_teaching_languages(lesson, teaching_language))
    return (not errors), errors
