"""Fail-closed recovery of concept explanations for already-built maps.

A map can be cached as a skeleton before any provider was configured, leaving
deep concepts with a literal "Explanation unavailable" placeholder even though
a validated lesson for that exact concept may already exist on disk under an
older provider-keyed cache name. This module repairs those maps by matching
cached lessons on topic + grounded transcript timestamps and re-running the
full quality gate. It never invents teaching content and never changes provider
configuration.
"""
from __future__ import annotations

import re
from typing import Any

from ..config import settings
from ..pipeline import llm
from ..pipeline.lesson_cache import find_cached_lesson
from ..pipeline.lessons import stamp_lesson, validate_deep_teaching, validate_lesson
from ..pipeline.terminology import acronym_expansion, canonical_topic
from .treeutil import walk


def is_teachable(node: dict) -> bool:
    """Recall items and structural nodes are not concept lessons."""
    return (node.get("type") == "concept"
            and not node.get("recall")
            and not node.get("oneline")
            and not node.get("qa")
            and not node.get("conceptUnavailable")
            and bool(node.get("id")))

PLACEHOLDER = re.compile(
    r"^(?:explanation unavailable|unavailable|no verified transcript evidence|n/?a)\b",
    re.I,
)


def has_explanation(node: dict) -> bool:
    """True only for substantive teaching text, not an honest placeholder."""
    for key in ("explain", "summary"):
        text = str(node.get(key) or "").strip()
        if text and not PLACEHOLDER.match(text):
            return True
    return False


def _evidence(node: dict) -> list[dict]:
    raw = node.get("evidence") or [{"start": node.get("start"), "end": node.get("end"),
                                    "text": node.get("source", "")}]
    return [
        {"start": e.get("start") or 0, "end": e.get("end"),
         "text": (e.get("text") or "")[:320]}
        for e in raw[:4]
    ]


def _patch(node: dict, lesson: dict) -> None:
    """Apply the same teaching fields the normal label pipeline would write."""
    node.update({
        "explain": lesson.get("explain", ""),
        "intuition": lesson.get("intuition", ""),
        "why_it_matters": lesson.get("why_it_matters", ""),
        "steps": lesson.get("steps", []),
        "worked_example": lesson.get("worked_example", ""),
        "diagram": lesson.get("diagram", ""),
        "misconception": lesson.get("misconception", ""),
        "exercise": lesson.get("exercise", ""),
        "knowledge_check": lesson.get("knowledge_check", {}),
        "lesson_evidence": lesson.get("evidence", {}),
        "provenance": lesson.get("provenance", {"example_source": "supplemental"}),
        "groundingNote": (
            "Factual claims about the video are grounded in the timestamped transcript; "
            "intuition, examples, diagrams and exercises are AI-generated teaching aids; "
            "accuracy_notes are externally supplemented corrections, not video claims."
        ),
        "accuracy_notes": lesson.get("accuracy_notes", []),
        "acronym_expansion": lesson.get("acronym_expansion", ""),
        "teaching_language": lesson.get("teaching_language", ""),
        "lesson_available": True,
        "available": True,
        "conceptUnavailable": False,
        "unavailableReason": "",
        "generatedBy": "llm-deep",
    })


def recover_explanations(mindmap: dict, concept_ids: list[str] | None = None) -> dict[str, Any]:
    """Recover validated lessons from cache for concepts that lack an explanation.

    Returns a report. Failures keep their reason so the UI can show an honest
    unavailable state and offer a safe retry; nothing is marked successful
    unless validate_lesson and validate_deep_teaching both pass.
    """
    meta = mindmap.get("meta") or {}
    language = meta.get("teachingLanguage") or "en"
    wanted = set(concept_ids or [])
    repaired: list[str] = []
    failed: list[dict[str, str]] = []
    skipped: list[str] = []

    root = mindmap.get("root")
    if not isinstance(root, dict):
        return {"repaired": repaired, "failed": failed, "skipped": skipped,
                "attempted": 0, "error": "map has no root"}

    for node, _parent, _depth in walk(root):
        if not is_teachable(node):
            continue
        cid = str(node["id"])
        if wanted and cid not in wanted:
            continue
        if has_explanation(node):
            skipped.append(cid)
            continue

        topic = canonical_topic(str(node.get("topic") or node.get("text") or ""))
        evidence = _evidence(node)
        cached = find_cached_lesson(topic, evidence, language)
        if not cached:
            failed.append({"id": cid, "label": str(node.get("text") or topic),
                           "reason": "No validated cached lesson for this concept."})
            continue

        stamped = stamp_lesson(cached, topic, evidence, language, acronym_expansion(topic))
        evidence_text = " ".join(e.get("text", "") for e in evidence)
        ok, errors = validate_lesson(stamped, evidence_text, topic, evidence, language)
        if ok:
            ok, errors = validate_deep_teaching(stamped, topic, language)
        if not ok:
            failed.append({"id": cid, "label": str(node.get("text") or topic),
                           "reason": "; ".join(errors[:6]) or "Lesson failed validation."})
            continue

        _patch(node, stamped)
        repaired.append(cid)

    mindmap.setdefault("meta", {})["explanationsRecovered"] = True
    return {
        "repaired": repaired,
        "failed": failed,
        "skipped": skipped,
        "attempted": len(repaired) + len(failed),
    }


def describe_generation_failure(rejection: dict) -> str:
    """Translate a provider diagnostic into an honest, specific UI reason.

    Keeps the fail-closed contract: a failed call never becomes a lesson. The
    point is only to tell the user *why* — missing provider, unreachable local
    server, unloaded model, bad configuration, invalid JSON or failing quality
    gate — instead of one generic "unavailable" string.
    """
    kind = str(rejection.get("providerErrorKind") or "")
    failure = str(rejection.get("failureKind") or "")
    status = rejection.get("providerStatus")
    error = str(rejection.get("providerError") or "").strip()

    if kind == "not_configured":
        return "No LLM provider is configured."
    if status == 404:
        return ("The configured model is not loaded at the endpoint (HTTP 404). "
                + (error[:160] or "Check LLM_MODEL."))
    if status in (401, 403):
        return (f"The endpoint rejected authentication (HTTP {status}). "
                + (error[:160] or "Check the API key."))
    if status == 400:
        return ("The endpoint rejected the request (HTTP 400): invalid configuration "
                "or an unsupported parameter. " + error[:160])
    if isinstance(status, int) and status >= 500:
        return f"The inference server returned HTTP {status}. " + error[:160]
    lowered = error.lower()
    if any(m in lowered for m in ("connecterror", "connecttimeout", "timed out", "timeout", "refused", "unreachable")):
        return "The configured inference server is unreachable. " + error[:180]
    if failure == "validation":
        return "The provider returned a lesson that did not pass quality validation; nothing was changed."
    if failure == "lesson_parse" or kind == "response_parse":
        return "The provider response was not valid lesson JSON; nothing was changed."
    if kind == "empty_response":
        return "The provider returned an empty response; nothing was changed."
    errors = rejection.get("errors") or []
    if errors:
        return "; ".join(str(e) for e in errors[:3])[:240]
    return (error[:240] or "Generation failed; nothing was changed.")


def generate_explanations(mindmap: dict, concept_ids: list[str] | None = None) -> dict[str, Any]:
    """Generate fresh validated lessons for teachable concepts.

    Uses the existing `_llm_deep_lessons` generator, which validates, stamps,
    checkpoints and revalidates cached candidates. Nothing is written to a node
    unless both quality gates pass. Never called without a configured provider.
    """
    from ..pipeline.labels import ConceptCand, _llm_deep_lessons

    wanted = set(concept_ids or [])
    root = mindmap.get("root")
    generated: list[str] = []
    failed: list[dict[str, str]] = []
    skipped: list[str] = []
    rejections: list[dict] = []

    if not isinstance(root, dict):
        return {"generated": generated, "failed": [{"id": "", "reason": "map has no root"}],
                "skipped": skipped, "attempted": 0, "rejections": rejections}

    candidates: dict[str, ConceptCand] = {}
    for node, _parent, _depth in walk(root):
        if not is_teachable(node):
            continue
        cid = str(node["id"])
        if wanted and cid not in wanted:
            continue
        if has_explanation(node):
            skipped.append(cid)
            continue
        candidates[cid] = ConceptCand(cid=cid, sid="", node=node)

    if not candidates:
        reason = ("No eligible concepts need generation." if not wanted
                  else "This node already has an explanation or is not a teachable concept.")
        return {"generated": generated, "failed": [{"id": (list(wanted)[0] if wanted else ""), "reason": reason}],
                "skipped": skipped, "attempted": 0, "rejections": rejections}

    if not llm.generation_available():
        if settings.llm_base_url and not settings.llm_is_local and not settings.llm_api_key:
            reason = "A remote endpoint is configured without an API key, so fresh generation is unavailable."
        elif settings.llm_base_url and not settings.llm_model:
            reason = "The local endpoint is configured but no model identifier is set."
        else:
            reason = "Fresh generation is unavailable: no LLM provider is configured."
        failed = [{"id": cid, "label": str(n.node.get("text") or ""), "reason": reason}
                  for cid, n in candidates.items()]
        return {"generated": generated, "failed": failed, "skipped": skipped,
                "attempted": len(candidates), "rejections": rejections,
                "provider": llm.provider_name()}

    accepted, failures, rejections = _llm_deep_lessons(mindmap, candidates)
    for cid, lesson in accepted.items():
        node = candidates[cid].node
        _patch(node, lesson)
        generated.append(cid)
    for label in failures:
        cid = next((c for c, n in candidates.items() if str(n.node.get("text") or "") == label), "")
        rejection = next((r for r in rejections if r.get("label") == label), {})
        reason = describe_generation_failure(rejection) if rejection else "Lesson failed validation."
        failed.append({"id": cid, "label": label, "reason": reason})

    mindmap.setdefault("meta", {})["explanationsGenerated"] = True
    return {
        "generated": generated,
        "failed": failed,
        "skipped": skipped,
        "attempted": len(candidates),
        "rejections": rejections,
        "provider": llm.provider_name(),
    }
