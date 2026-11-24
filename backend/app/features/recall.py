"""Meaningful Quick Recall pairs grounded in validated lesson material.

Legacy maps store extractive transcript fragments as recall "points". This
module replaces only those generated fragments with question/answer pairs,
preserving user-authored recall items and honest unavailability when no
validated lesson material exists.
"""
from __future__ import annotations

import difflib
import re
from typing import Any

from ..pipeline import llm
from ..pipeline.builder import _norm, compress, make_node
from ..pipeline.lessons import overlap
from ..pipeline.text_utils import truncate
from .treeutil import walk

WH = re.compile(r"\b(what|why|how|which|when|where|who|difference|explain|purpose)\b", re.I)
GENERIC = re.compile(r"^covers\b", re.I)


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def valid_pair(question: str, answer: str, evidence: str = "") -> bool:
    """Reject repetition, weak prompts and unsupported answers."""
    q, a = str(question or "").strip(), str(answer or "").strip()
    if not q or not a or not (WH.search(q) or "?" in q):
        return False
    nq, na = normalize(q), normalize(a)
    if not nq or not na or nq == na or difflib.SequenceMatcher(None, nq, na).ratio() > 0.72:
        return False
    if len(a.split()) < 5 or len(a.split()) <= len(q.split()):
        return False
    if GENERIC.match(q) or GENERIC.match(a):
        return False
    if evidence and overlap(a, evidence) < 0.12:
        return False
    return True


def _concepts(mindmap: dict) -> list[dict]:
    out = []
    sections = [s for s in (mindmap.get("root") or {}).get("children", []) if not s.get("recall")]
    for section in sections:
        for node in section.get("children", []):
            if node.get("type") != "concept" or node.get("recall"):
                continue
            if not (node.get("lesson_available") or node.get("available")) or node.get("conceptUnavailable"):
                continue
            out.append(node)
    out.sort(key=lambda n: (n.get("start") if n.get("start") is not None else 1e9))
    return out


def _llm_pairs(concepts: list[dict]) -> list[dict]:
    """Use the existing LLM pipeline for Q/A when a provider is configured."""
    if not llm.fast_available() or not concepts:
        return []
    listing = [{
        "id": str(c["id"]),
        "topic": str(c.get("topic") or c.get("text") or ""),
        "evidence": " ".join(str(e.get("text") or "") for e in (c.get("evidence") or [])[:3])[:600],
    } for c in concepts[:6]]
    system = (
        "Create revision questions from the supplied video evidence. Return JSON only: "
        '{"pairs":[{"id":"...","question":"...","answer":"..."}]}. '
        "Questions must ask for an explanation or application. Answers must add information "
        "beyond the question and must only use the supplied evidence."
    )
    try:
        data = llm.chat_json(system, llm.compact(listing), max_tokens=1800, temperature=0.2)
    except Exception:  # provider failures stay honest: fall back to no new pairs
        return []
    if not isinstance(data, dict):
        return []
    by_id = {str(c["id"]): c for c in concepts}
    pairs = []
    for item in data.get("pairs") or []:
        if not isinstance(item, dict):
            continue
        node = by_id.get(str(item.get("id") or ""))
        if not node:
            continue
        evidence = " ".join(str(e.get("text") or "") for e in (node.get("evidence") or [])[:3])
        q, a = str(item.get("question") or ""), str(item.get("answer") or "")
        if valid_pair(q, a, evidence):
            pairs.append({"question": q, "answer": a, "node": node})
    return pairs


def build_recall_pairs(mindmap: dict, use_llm: bool = True) -> list[dict]:
    """Pairs from validated knowledge checks, optionally topped up by the LLM."""
    concepts = _concepts(mindmap)
    pairs: list[dict] = []
    seen: set[str] = set()
    for node in concepts:
        check = node.get("knowledge_check") or {}
        q = str(check.get("question") or "").strip()
        a = str(check.get("answer") or "").strip()
        evidence = " ".join(str(e.get("text") or "") for e in (node.get("evidence") or [])[:3])
        if not valid_pair(q, a, evidence):
            continue
        key = normalize(q)
        if key in seen or any(difflib.SequenceMatcher(None, key, old).ratio() > 0.6 for old in seen):
            continue
        seen.add(key)
        pairs.append({"question": q, "answer": a, "node": node})
    if len(pairs) < 3 and use_llm:
        for extra in _llm_pairs(concepts):
            key = normalize(extra["question"])
            if key in seen or any(difflib.SequenceMatcher(None, key, old).ratio() > 0.6 for old in seen):
                continue
            seen.add(key)
            pairs.append(extra)
    pairs.sort(key=lambda p: (p["node"].get("start") if p["node"].get("start") is not None else 1e9))
    if len(pairs) > 6:
        pairs = [pairs[round(i * (len(pairs) - 1) / 5)] for i in range(6)]
    return pairs


def _is_generated_fragment(child: dict) -> bool:
    """Legacy extractive point: text is just a compressed copy of its source."""
    if not child.get("recall") or child.get("qa") or child.get("oneline"):
        return False
    source = str(child.get("source") or "")
    text = str(child.get("text") or "")
    return bool(source) and text == compress(source, 14)


def _is_user_item(child: dict) -> bool:
    return bool(child.get("notes") or child.get("links") or child.get("image")
                or child.get("userAuthored") or child.get("origin") == "user"
                or not _is_generated_fragment(child))


def improve_recall(mindmap: dict, use_llm: bool = True) -> dict[str, Any]:
    """Replace only generated recall fragments; keep user items and evidence."""
    root = mindmap.get("root") or {}
    recall = next((s for s in root.get("children", []) if s.get("recall")), None)
    if not isinstance(recall, dict):
        return {"pairs": 0, "preserved": 0, "removed": 0, "unavailable": "No Quick recall section."}

    existing = recall.get("children") or []
    preserved = [c for c in existing if _is_user_item(c)]
    removed = len(existing) - len(preserved)
    pairs = build_recall_pairs(mindmap, use_llm=use_llm)

    children = [c for c in preserved if c.get("oneline")]
    for pair in pairs:
        node = pair["node"]
        children.append(make_node(
            f"Q: {pair['question']}  A: {pair['answer']}", "concept", 2,
            node.get("start"), node.get("end"),
            source=truncate(str(node.get("source") or ""), 300),
            question=pair["question"], answer=pair["answer"],
            recall=True, qa=True, conceptId=str(node["id"]),
            generatedBy=node.get("generatedBy") or "recall-qa",
        ))
    children.extend(c for c in preserved if not c.get("oneline"))
    recall["children"] = children
    recall["qaCount"] = len(pairs)
    if not pairs:
        recall["recallUnavailable"] = "No validated lesson material is available for recall questions."
    else:
        recall.pop("recallUnavailable", None)

    summary = str(recall.get("summary") or "")
    if not summary or GENERIC.match(summary):
        recall["summary"] = " · ".join(str(c.get("text") or "").strip() for c in _concepts(mindmap)[:6])
    return {"pairs": len(pairs), "preserved": len(preserved), "removed": removed,
            "unavailable": recall.get("recallUnavailable")}
