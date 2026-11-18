"""Domain-independent validation for Deep Mode concept lessons.

A lesson is only accepted when every teaching field is substantive, grounded in the
concept's own transcript evidence, and clearly separates video claims from supplemental
teaching. Nothing here assumes a particular subject.
"""
from __future__ import annotations

import difflib
import re
from typing import Any

LESSON_FIELDS = (
    "explain",          # definition / plain-English meaning
    "why_it_matters",   # relevance to a beginner
    "steps",            # ordered mechanism (list[str])
    "worked_example",   # concrete, specific, supplemental
    "diagram",          # meaningful multi-line text diagram
    "misconception",    # common wrong idea + correction
    "exercise",         # practical task
    "knowledge_check",  # {"question": ..., "answer": ...}
)

GENERIC_PHRASES = re.compile(
    r"\b(?:a concept covered|key idea introduced|available information|"
    r"component that can be part of|important to understand|this is a topic|"
    r"in this section|the speaker|as mentioned earlier)\b",
    re.I,
)

ACTION_VERBS = re.compile(
    r"\b(?:write|draw|explain|predict|trace|identify|list|compare|build|try|which|what|would|should|choose|select|"
    r"implement|sketch|describe|calculate|check|test|run|create)\b",
    re.I,
)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def overlap(text: str, evidence: str) -> float:
    """Similarity to the source passage; high overlap means a copied sentence."""
    if not text or not evidence:
        return 0.0
    return difflib.SequenceMatcher(None, _norm(text), _norm(evidence)).ratio()


def validate_lesson(lesson: dict[str, Any], evidence_text: str, topic: str) -> tuple[bool, list[str]]:
    """Return (accepted, errors). No subject-specific facts are consulted here."""
    errors: list[str] = []
    if not isinstance(lesson, dict):
        return False, ["lesson is not an object"]

    def field(name: str) -> Any:
        return lesson.get(name)

    def text_value(name: str) -> str:
        value = field(name)
        return value.strip() if isinstance(value, str) else ""

    explain = text_value("explain")
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
    if topic and not any(w in _norm(explain) for w in _norm(topic).split()):
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

    # Provenance is stamped by the pipeline; the model often omits it.
    return (not errors), errors
