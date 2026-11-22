"""
Select-then-rewrite: ONE LLM call turns an already built map into exam-ready English.

The pipeline has already SELECTED everything (sections, concepts, tagged level-3 items, their
timestamps, source sentences and the summary retriever's best sentences). This module sends a
compact numbered candidate list and asks for rewritten text back, one item per line:

    R|Topic of the whole video
    M|One-line summary of the whole video
    S|s1|Section title|What the section teaches
    C|c1|Term: one-line meaning
    D|d1|Eg|Short example
    X|c2>c5|leads to          (optional, at most 4)

Anything missing, malformed, non-English or duplicated keeps its heuristic (English) text,
so a timeout, a 429 or a garbled answer still yields a complete map. Results are cached per
section (keyed by a hash of that section's candidate list), so re-runs cost 0 tokens.

Input is built from the map JSON alone, so the rewrite can run later (e.g. on a cached
skeleton) and returns ops for the same `update` / `edge:add` path the extension and the
collaboration server already use. The LLM never writes the tree structure or timestamps.
"""
from __future__ import annotations

import hashlib
import json
import time
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from ..collab.ops import apply_op
from ..config import settings
from . import llm
from .builder import TAGS, make_node, new_id, score_faithfulness
from .lessons import LESSON_FIELDS, overlap, validate_deep_teaching, validate_lesson
from .terminology import (acronym_expansion, apply_accuracy_corrections, norm as term_norm,
                         TECHNICAL_CORRECTIONS)
from .topics import ESSENTIAL_LABELS, HTTP_TOPICS
from .embeddings import Embedder
from .text_utils import fmt_time, truncate, unicode_norm
from .translate import needs_translation

log = logging.getLogger("mivmap.labels")

PROMPT_VERSION = "4"
MAX_CONCEPTS_PER_CALL = 60
MAX_PROMPT_TOKENS = 6000
MAX_CHUNKS = 3
MAX_LINKS = 4
MAX_DETAILS_PER_CONCEPT = 1  # level-3 items rewritten per concept (the rest keep their heuristic text)
SENTENCE_WORDS = 25
# The answer has to fit the deadline: ~350 output tokens/s on the fast model, after ~0.3 s to the
# first token. D lines are only asked for when the whole answer fits; otherwise the (already tagged)
# heuristic level-3 lines stay. Estimated answer tokens per line kind:
OUTPUT_TOKENS_PER_SECOND = 350
LINE_TOKENS = {"R": 8, "M": 22, "S": 18, "C": 13, "A": 26, "E": 22, "G": 22, "Q": 22, "D": 14, "X": 8}

SYSTEM = """You improve a Mivmap mind map into a beginner-friendly learning sketchbook.
The tree and timestamps are fixed. Rewrite only the text you are given.

Reply with lines only, nothing else, in this order:
R|topic of the whole video (2-6 words)
M|one-line summary of the whole video (max 20 words)
S|<section id>|section title (2-6 words)|what the section teaches (max 12 words)
C|<concept id>|Term: one-line meaning (max 12 words)
A|<concept id>|beginner explanation in plain English (max 25 words)
E|<concept id>|one concrete HTTP example (max 20 words)
G|<section id>|tiny ASCII diagram (one line, max 50 chars)
Q|<section id>|one practical exercise (max 20 words)
D|<detail id>|<Def, Eg, Formula, Tip or Watch-out>|the point (max 14 words)
X|<concept id>><concept id>|relation (1-3 words)

Grounding rules (important):
- Definitions (C/S/D/R/M lines) must come only from the supplied transcript sentences. Do not invent facts.
- A/E/G/Q lines are GENERATED study aids. They may explain, illustrate and exercise the topic, but never say
  "the speaker says", "the video states" or imply that the exact wording or example was in the video.
- English only. Fix obvious caption typos and garbled technical terms. Self-contained lines; never write "this section" or "the video".
- One line for every id you are given. Keep ids exactly. Fixed titles: repeat unchanged.
- No timestamps, no markdown, no duplicate lines. X lines optional, at most 4, between different sections."""


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------
@dataclass
class SectionCand:
    cid: str
    node: dict
    concepts: list["ConceptCand"] = field(default_factory=list)
    fixed: bool = False
    key: str = ""
    details: list["DetailCand"] = field(default_factory=list)
    recall: bool = False


@dataclass
class ConceptCand:
    cid: str
    sid: str
    node: dict


@dataclass
class DetailCand:
    cid: str  # d1, d2…
    concept: str  # c-id of its concept
    node: dict


def collect(mindmap: dict) -> list[SectionCand]:
    sections: list[SectionCand] = []
    c = d = 0
    for i, sec in enumerate(n for n in mindmap["root"].get("children", []) if n.get("type") == "section"):
        recall = bool(sec.get("recall"))
        cand = SectionCand(f"s{i + 1}", sec, fixed=bool(sec.get("chapter")) or recall, recall=recall)
        for node in sec.get("children", []):
            if node.get("type") != "concept" or node.get("oneline"):
                continue
            c += 1
            concept = ConceptCand(f"c{c}", cand.cid, node)
            cand.concepts.append(concept)
            for child in [ch for ch in node.get("children", []) if ch.get("type") == "detail" and ch.get("tag")][:MAX_DETAILS_PER_CONCEPT]:
                d += 1
                cand.details.append(DetailCand(f"d{d}", concept.cid, child))
        cand.key = _section_key(mindmap, cand)
        sections.append(cand)
    return sections


def _words(text: str, n: int = SENTENCE_WORDS) -> str:
    words = re.sub(r"\s+", " ", text or "").strip().split(" ")
    return " ".join(words[:n]) + ("…" if len(words) > n else "")


def _sentence(node: dict) -> str:
    return _words(node.get("source") or node.get("summary") or "")


def _untagged(node: dict) -> str:
    text = node.get("text") or ""
    tag = node.get("tag")
    return text[len(tag) + 1:].strip() if tag and text.startswith(f"{tag}:") else text


def _section_line(s: SectionCand) -> str:
    node = s.node
    if s.recall:
        return f"{s.cid} | Quick recall (fixed title; its concepts are must-remember points)"
    span = f"{fmt_time(node.get('start'))}-{fmt_time(node.get('end'))}"
    head = f"chapter: {node['text']} (fixed title)" if s.fixed else f"keywords: {', '.join((node.get('keywords') or [])[:5])}"
    return f'{s.cid} {span} | {head} | gist: "{_words(node.get("summary") or "", 30)}"'


def _concept_line(c: ConceptCand) -> str:
    return f'{c.cid} {c.sid} {fmt_time(c.node.get("start"))} | {c.node["text"]} | "{_sentence(c.node)}"'


def _detail_line(d: DetailCand) -> str:
    return f'{d.cid} {d.concept} {d.node.get("tag")} | {_untagged(d.node)} | "{_sentence(d.node)}"'


def _section_key(mindmap: dict, s: SectionCand) -> str:
    body = "\n".join(
        [PROMPT_VERSION, _section_line(s).split(" ", 1)[1]]
        + [_concept_line(c).split(" ", 2)[2] for c in s.concepts]
        + [_detail_line(d).split(" ", 2)[2] for d in s.details]
    )
    return hashlib.sha1(body.encode()).hexdigest()[:20]


def _root_key(mindmap: dict, sections: list[SectionCand]) -> str:
    body = "|".join([PROMPT_VERSION, mindmap["meta"].get("title", "")] + [s.key for s in sections])
    return hashlib.sha1(body.encode()).hexdigest()[:20]


def build_prompt(title: str, sections: list[SectionCand], want_root: bool, outline: list[str] | None = None, with_details: bool = True) -> str:
    # what a student reads first comes first: if the deadline cuts the answer, the tail is lost
    sections = [s for s in sections if s.recall] + [s for s in sections if not s.recall]
    details = [_detail_line(d) for s in sections for d in s.details] if with_details else []
    lines = [f"Video: {title}"]
    if want_root and outline:
        lines.append("Outline of the whole video: " + "; ".join(outline))
    todo = ("one S, G and Q line per section, one C, A and E line per concept"
            + (", one D line per detail" if details else "") + ", then up to 4 X lines")
    lines.append(f"Answer with: the R line, the M line, {todo}." if want_root else f"Answer with: {todo}. No R or M line.")
    lines.append("Sections:")
    lines += [_section_line(s) for s in sections]
    lines.append("Concepts:")
    lines += [_concept_line(c) for s in sections for c in s.concepts]
    if details:
        lines.append("Details:")
        lines += details
    return "\n".join(lines)


def answer_tokens(sections: list[SectionCand], want_root: bool, with_details: bool) -> int:
    """Estimated size of a complete answer (to decide whether D lines fit the deadline)."""
    n = (LINE_TOKENS["R"] + LINE_TOKENS["M"] if want_root else 0) + MAX_LINKS * LINE_TOKENS["X"]
    for s in sections:
        n += LINE_TOKENS["S"] + LINE_TOKENS["C"] * len(s.concepts) + (LINE_TOKENS["D"] * len(s.details) if with_details else 0)
    return n


def max_tokens(n_lines: int) -> int:
    """~13 tokens per answer line plus headroom for a reasoning model's (low-effort) thinking."""
    return min(1100, 150 + 13 * n_lines)


def plan_chunks(sections: list[SectionCand], title: str) -> list[list[SectionCand]]:
    """Whole video in one prompt; split into ≤3 contiguous chunks only when it is too big."""
    total = sum(len(s.concepts) for s in sections)
    tokens = llm.estimate_tokens(SYSTEM + build_prompt(title, sections, True))
    n = 1
    while n < MAX_CHUNKS and (total / n > MAX_CONCEPTS_PER_CALL or tokens / n > MAX_PROMPT_TOKENS):
        n += 1
    if n == 1:
        return [sections]
    per = total / n
    chunks: list[list[SectionCand]] = [[]]
    count = 0
    for s in sections:
        if chunks[-1] and count + len(s.concepts) / 2 > per * len(chunks) and len(chunks) < n:
            chunks.append([])
        chunks[-1].append(s)
        count += len(s.concepts)
    return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# Tolerant line parser
# ---------------------------------------------------------------------------
@dataclass
class Parsed:
    root: str | None = None
    summary: str | None = None  # M line
    sections: dict[str, str] = field(default_factory=dict)
    section_summaries: dict[str, str] = field(default_factory=dict)
    concepts: dict[str, str] = field(default_factory=dict)
    details: dict[str, tuple[str | None, str]] = field(default_factory=dict)  # id -> (tag or None, text)
    links: list[tuple[str, str, str]] = field(default_factory=list)
    explanations: dict[str, str] = field(default_factory=dict)
    examples: dict[str, str] = field(default_factory=dict)
    diagrams: dict[str, str] = field(default_factory=dict)
    exercises: dict[str, str] = field(default_factory=dict)
    malformed: int = 0


LINE_RE = re.compile(r"^\s*(?:[-*•>]+|\d+[.)])?\s*(?:\*\*)?\s*([RSCXAEGQMD])\s*(?:\*\*)?\s*\|(.*)$", re.I)
BARE_ID_RE = re.compile(r"^\s*(?:[-*•]+\s*)?\[?([scd]\d+)\]?\s*\|(.*)$", re.I)
ID_RE = re.compile(r"^\[?([scd]\d+)\]?$", re.I)
LINK_RE = re.compile(r"^\[?(c\d+)\]?\s*(?:->|>|→|,|\s)\s*\[?(c\d+)\]?$", re.I)
BARE_LINK_RE = re.compile(r"^\s*(c\d+)\s+(c\d+)\s*$", re.I)
TIME_RE = re.compile(r"\[?\(?\b\d{1,2}:\d{2}(?::\d{2})?\b\)?\]?")
META_RE = re.compile(
    r"\b(?:the speaker|this section|this video|the video|speaker (?:mentions|says|talks)|"
    r"here is|here are|let's|we will|we're going to|as i mentioned|i think)\b", re.I)
TAG_ALIASES = {
    "def": "Def", "definition": "Def", "defn": "Def", "meaning": "Def",
    "eg": "Eg", "e.g": "Eg", "e.g.": "Eg", "example": "Eg", "ex": "Eg",
    "formula": "Formula", "equation": "Formula", "rule": "Formula",
    "tip": "Tip", "remember": "Tip", "note": "Tip", "key point": "Tip",
    "watch-out": "Watch-out", "watch out": "Watch-out", "watchout": "Watch-out", "caution": "Watch-out", "warning": "Watch-out", "pitfall": "Watch-out",
}


def clean_label(text: str, max_chars: int, max_words: int,
                teaching_language: str = "en") -> str | None:
    text = TIME_RE.sub(" ", text.replace("**", "").replace("`", ""))
    text = re.sub(r"\s+", " ", text).strip().strip("\"'“”‘’").rstrip(".;,").strip()
    text = re.sub(r"\s+:", ":", text)
    hindi_ok = teaching_language.lower().split("-")[0] == "hi"
    has_content = any(ch.isalnum() for ch in text)  # Unicode-aware, keeps Devanagari
    if (not text or META_RE.search(text) or not has_content or "।" in text
            or (not hindi_ok and needs_translation(text))):
        return None
    if len(text.split()) > max_words or len(text) > max_chars:
        return None
    return text


def _tag(text: str) -> str | None:
    return TAG_ALIASES.get(text.strip().strip("[]()*").lower())


def parse_labels(text: str, teaching_language: str = "en") -> Parsed:
    """Parse the pipe format. Never raises; anything unusable is counted as malformed and skipped."""
    lang = teaching_language
    out = Parsed()
    last_concept: str | None = None
    last_section: str | None = None
    seen_pipe = False
    plain_lines = 0
    for raw in (text or "").replace("｜", "|").splitlines():
        if not raw.strip() or raw.strip().startswith("```"):
            continue
        if "|" in raw:
            seen_pipe = True
        if "|" not in raw and not seen_pipe and plain_lines < 2:
            plain_lines += 1
            stripped = raw.strip().strip("*# ")
            if re.search(r"here are|labels:|sure[!,]", stripped, re.I):
                out.malformed += 1
                continue
            if stripped and not out.root:
                label = clean_label(stripped, 60, 8, lang)
                if label: out.root = label
                continue
            if stripped and not out.summary:
                label = clean_label(stripped, 170, 24, lang)
                if label: out.summary = label
                continue
        if not raw.strip() or raw.strip().startswith("```"):
            continue
        m = LINE_RE.match(raw)
        if m:
            kind, rest = m.group(1).upper(), m.group(2)
        elif bm := BARE_ID_RE.match(raw):  # "c3|Label" without the C prefix
            kind, rest = bm.group(1)[0].upper(), f"{bm.group(1)}|{bm.group(2)}"
        else:
            out.malformed += 1
            continue
        parts = [p.strip() for p in rest.split("|")]
        if kind in "RM":
            label = clean_label(" ".join(p for p in parts if p), 60 if kind == "R" else 170, 8 if kind == "R" else 24, lang)
            current = out.root if kind == "R" else out.summary
            if label and current is None:
                if kind == "R":
                    out.root = label
                else:
                    out.summary = label
            elif not label:
                out.malformed += 1
            continue
        if kind in "SCD":
            ident = ID_RE.match(parts[0]) if parts else None
            id_kind = ident.group(1)[0].upper() if ident else ""
            if not ident or len(parts) < 2 or not (id_kind == kind or kind == "D" and id_kind == "C"):
                out.malformed += 1
                continue
            key = ident.group(1).lower()
            if kind == "S":
                last_section = key
                title = clean_label(parts[1], 60, 8, lang)
                summary = clean_label(". ".join(p for p in parts[2:] if p), 140, 20, lang) if len(parts) > 2 else None
                if title and key not in out.sections:
                    out.sections[key] = title
                if summary and key not in out.section_summaries:
                    out.section_summaries[key] = summary
                if not title and not summary:
                    out.malformed += 1
                continue
            if kind == "C":
                last_concept = key
                # The fast model often emits C|Term: meaning|source-sentence.
                label_text = parts[1] if len(parts) >= 3 and parts[1] and ":" in parts[1] else ": ".join(p for p in parts[1:] if p)
                label = clean_label(label_text, 110, 16, lang)
                if label and key not in out.concepts:
                    out.concepts[key] = label
                elif not label:
                    out.malformed += 1
                continue
            # D|d1|Eg|text  ·  D|d1|text  ·  D|c1|Def|text (detail addressed through its concept)
            tag = _tag(parts[1]) if len(parts) > 2 else None
            body = " ".join(p for p in parts[2:] if p) if tag else ": ".join(p for p in parts[1:] if p)
            inline = re.match(r"^(def|definition|eg|e\.g\.?|example|formula|equation|tip|remember|note|watch[- ]?out|caution|warning)\s*:\s*", body, re.I)
            if inline:  # "Watch out: …" written inside the text: that is the tag
                tag, body = tag or _tag(inline.group(1)), body[inline.end():]
            label = clean_label(body, 110, 16, lang)
            target = f"{key}#{tag or ''}" if id_kind == "C" else key
            if label and target not in out.details:
                out.details[target] = (tag, label)
            elif not label:
                out.malformed += 1
            continue
        if kind in "AEGQ":
            ident = ID_RE.match(parts[0]) if parts else None
            if ident:
                key = ident.group(1).lower()
                body_parts = parts[1:]
            else:
                key = (last_concept if kind in "AE" else last_section) or ""
                body_parts = parts
            if not key or not body_parts:
                out.malformed += 1
                continue
            body = clean_label(": ".join(p for p in body_parts if p), 180, 28, lang)
            if not body:
                out.malformed += 1
                continue
            target = {
                "A": out.explanations, "E": out.examples, "G": out.diagrams, "Q": out.exercises,
            }[kind]
            fingerprint = unicode_norm(body)
            if any(unicode_norm(v) == fingerprint for v in target.values()):
                out.malformed += 1
                continue
            target.setdefault(key, body)
            continue
        # X
        link = LINK_RE.match(parts[0]) if parts else None
        bare = BARE_LINK_RE.match(parts[0]) if parts else None
        label = clean_label(parts[1], 32, 4, lang) if len(parts) > 1 else None
        if bare and label and len(out.links) < MAX_LINKS:
            out.links.append((bare.group(1).lower(), bare.group(2).lower(), label))
        elif link and label and len(out.links) < MAX_LINKS:
            out.links.append((link.group(1).lower(), link.group(2).lower(), label))
        elif not (link and label):
            out.malformed += 1
    return out


def _resolve_details(parsed: Parsed, details: dict[str, DetailCand]) -> dict[str, tuple[str | None, str]]:
    """Map 'c1#Def'-style answers onto the concept's detail with that tag (or its first one)."""
    out = {k: v for k, v in parsed.details.items() if k in details}
    for key, value in parsed.details.items():
        if "#" not in key:
            continue
        concept, tag = key.split("#", 1)
        owned = [d for d in details.values() if d.concept == concept]
        pick = next((d for d in owned if tag and d.node.get("tag") == tag), owned[0] if owned else None)
        if pick and pick.cid not in out:
            out[pick.cid] = value
    return out


# ---------------------------------------------------------------------------
# Cache (per section, keyed by its candidate list) — re-runs cost 0 tokens
# ---------------------------------------------------------------------------
def _cache_path(key: str):
    path = settings.data_dir / "cache" / "labels"
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{key}.json"


def _cache_read(key: str) -> dict | None:
    try:
        return json.loads(_cache_path(key).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _cache_write(key: str, value: dict) -> None:
    try:
        _cache_path(key).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        log.warning("label cache write failed: %s", exc)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def _llm_enrich(mindmap: dict, concepts: dict, details: dict) -> dict:
    """LLM teaching aids for non-deep modes; deterministic notes remain the fallback."""
    items = []
    for c in concepts.values():
        node = c.node
        evidence = node.get("evidence") or [{"start": node.get("start"), "end": node.get("end"), "text": node.get("source", "")}]
        items.append({
            "id": c.cid,
            "title": node.get("text", ""),
            "topic": node.get("topic", ""),
            "evidence": [{"at": round(float(e.get("start") or 0), 1), "text": (e.get("text") or "")[:260]}
                         for e in evidence[:3]],
        })
    if not items:
        return {}
    system = (
        "You are an expert teacher writing beginner study aids. Return a JSON object keyed by the given id. "
        "For each id return: definition (1-2 specific sentences that name the concept), intuition (plain-English "
        "mental model), why_it_matters, steps (3 ordered mechanism steps specific to this concept), "
        "worked_example (one concrete supplemental example), diagram (ASCII flow), "
        "misconception (wrong idea and correction), exercise (a task a learner can actually perform), "
        "knowledge_check {question, answer, explanation}, provenance {example_source:'supplemental'}. "
        "Use ONLY supplied evidence for factual claims about the video. Examples, diagrams and exercises are "
        "your own teaching aids: keep them supplemental and never attribute them to the video. If the evidence "
        "is insufficient for a concept, return an empty object for that id. Never say 'the speaker says'."
    )
    result: dict[str, dict] = {
        "explanations": {}, "intuitions": {}, "why_it_matters": {}, "mechanisms": {},
        "examples": {}, "exercises": {}, "titles": {}, "diagrams": {},
        "misconceptions": {}, "knowledge_checks": {}, "available": {}, "generatedBy": {},
    }
    wanted = {it["id"] for it in items}
    for item in items:
        data = llm.chat_json(system, json.dumps({item["id"]: item}, ensure_ascii=False),
                             max_tokens=1200, temperature=0.3)
        if not isinstance(data, dict):
            continue
        entries = {str(k).lower(): v for k, v in data.items()
                   if isinstance(v, dict) and str(k).lower() in wanted}
        if not entries:  # flat maps: {"explanations": {"c1": ...}, ...}
            for key, values in data.items():
                if isinstance(values, dict):
                    for cid, value in values.items():
                        if str(cid).lower() in wanted:
                            entries.setdefault(str(cid).lower(), {})[key] = value
        for cid, obj in entries.items():
            ev = next((it for it in items if it["id"] == cid), None)
            ev_text = " ".join(e.get("text", "") for e in (ev or {}).get("evidence", []))
            definition = str(obj.get("definition") or "").strip()
            if len(definition) < 40 or overlap(definition, ev_text) > 0.78:
                continue  # reject transcript copies / too-short definitions
            result["explanations"][cid] = definition
            for source_key, target_key in (
                ("intuition", "intuitions"), ("why_it_matters", "why_it_matters"),
                ("steps", "mechanisms"), ("worked_example", "examples"),
                ("exercise", "exercises"), ("diagram", "diagrams"),
                ("misconception", "misconceptions"), ("knowledge_check", "knowledge_checks"),
            ):
                value = obj.get(source_key)
                if isinstance(value, str) and value.strip():
                    result[target_key][cid] = value.strip()
                elif value:
                    result[target_key][cid] = value  # steps: list, knowledge_check: dict
            result["available"][cid] = True
            result["generatedBy"][cid] = "llm"
    for item in items:
        result["generatedBy"].setdefault(item["id"], "extractive-transcript")
    return result

def _llm_deep_lessons(mindmap: dict, concepts: dict) -> tuple[dict, list[str]]:
    """Generate, validate and checkpoint one study-ready lesson per concept.

    Returns (accepted_lessons, failed_concept_labels). Every accepted lesson is
    stamped with three-way provenance, the correct acronym expansion (when the
    concept is an acronym) and externally sourced accuracy notes. Cache hits
    are reused; only missing concepts call the LLM, with bounded backoff.
    """
    from .lesson_cache import lesson_key, load as load_lesson, save as save_lesson

    meta = mindmap.get("meta") or {}
    teaching_language = meta.get("teachingLanguage") or "en"
    source_language = meta.get("sourceLanguage") or "en"
    items = []
    for c in concepts.values():
        node = c.node
        evidence = node.get("evidence") or [{"start": node.get("start"), "end": node.get("end"), "text": node.get("source", "")}]
        ev = [{"start": e.get("start") or 0, "end": e.get("end"), "text": (e.get("text") or "")[:320]} for e in evidence[:4]]
        topic = node.get("topic") or node.get("text", "")
        model = settings.llm_model or settings.groq_model
        base = settings.llm_base_url or "configured"
        key = lesson_key(topic, ev, model, base, teaching_language)
        cfg = TECHNICAL_CORRECTIONS.get(term_norm(topic), {})
        pitfalls = [{"video_claim": n["video_claim"], "correction": n["correction"]}
                    for n in cfg.get("notes", [])]
        items.append({
            "id": c.cid, "topic": topic, "evidence": ev, "key": key,
            "label": node.get("text", topic),
            "teachingLanguage": teaching_language, "sourceLanguage": source_language,
            "acronymExpansion": acronym_expansion(topic),
            "knownPitfalls": pitfalls,
        })
    if not items:
        return {}, []

    language_rule = (
        "Write EVERY teaching field in English. Keep standard technical acronyms "
        "(LAN, WAN, TCP/IP, OSI, ARPANET) in their conventional English form."
        if teaching_language.lower().split("-")[0] == "en" else
        f"Write EVERY teaching field in {teaching_language} so a {teaching_language} speaker can "
        "learn without replaying the video. Keep standard technical acronyms and proper nouns "
        "(LAN, WAN, TCP/IP, OSI, ARPANET, BGP, ONT) in English, but explain them in "
        f"{teaching_language}. Do not translate the supplied evidence - it is for grounding only."
    )
    if teaching_language.lower().split("-")[0] != "en":
        # The cloud provider is markedly less reliable with long, deeply nested
        # prompts for non-English output. Keep the same required fields but use
        # a compact instruction; all provenance/accuracy stamping is done below.
        system = (
            "Return JSON only, keyed by id. Fields per id: title, explain (2 sentences), intuition, "
            "why_it_matters, steps (3), worked_example, diagram (ASCII), misconception, exercise, "
            "knowledge_check {question, answer, explanation}, "
            "evidence {timestamps:[numbers from evidence], quotes:[short exact phrases]}. "
            + language_rule + " "
            "Ground video claims in the supplied evidence. Examples, diagrams and exercises are your own "
            "teaching aids. If evidence is insufficient, return an empty object for that id."
        )
    else:
        system = (
            "You are an expert teacher. Return JSON only, an object keyed by the given id. Per id return: "
            "title, explain (2+ sentences, begins with the concept name), intuition, why_it_matters, "
            "steps (3-6), worked_example, diagram (multi-line ASCII), misconception, exercise, "
            "knowledge_check {question, answer, explanation}, "
            "evidence {timestamps:[numbers from the supplied evidence], quotes:[short exact phrases from it]}. "
            + language_rule + " "
            "Use only the supplied evidence for video claims; examples, diagrams and exercises are your own teaching "
            "aids - never say 'the speaker says'. If knownPitfalls is given, correct that video claim, do not repeat it. "
            "Each field teaches something new. Insufficient evidence: return an empty object for that id."
        )

    def finish(lesson: dict, item: dict) -> dict:
        """Stamp honest, deterministic source attribution onto a validated lesson."""
        out = apply_accuracy_corrections(lesson, item["topic"])
        # The model sometimes omits the evidence object. Fall back to the
        # verified transcript positions already attached to this concept;
        # never invent a timestamp or a quote.
        ev = out.get("evidence")
        if not isinstance(ev, dict) or not (ev.get("timestamps") or ev.get("starts")):
            stamps = sorted({round(float(e.get("start") or 0), 2)
                             for e in item["evidence"] if e.get("start") is not None})
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
        expansion = item.get("acronymExpansion") or acronym_expansion(item["topic"])
        if expansion:
            out["acronym_expansion"] = expansion
        out["teaching_language"] = teaching_language
        return out

    accepted: dict[str, dict] = {}
    failed: list[str] = []
    max_attempts = 4
    base_delay = 2.0
    max_delay = 45.0

    for item in items:
        hit = load_lesson(item["key"])
        if hit and isinstance(hit.get("lesson"), dict):
            stamped = finish(hit["lesson"], item)
            ev_text = " ".join(e.get("text", "") for e in item["evidence"])
            ok, _ = validate_lesson(stamped, ev_text, item["topic"], item["evidence"], teaching_language)
            ok2, _ = validate_deep_teaching(stamped, item["topic"], teaching_language)
            if ok and ok2:
                accepted[item["id"]] = stamped
                continue

        lesson = None
        for attempt in range(max_attempts):
            data = None
            try:
                data = llm.chat_json(
                    system,
                    json.dumps([{k: item[k] for k in (
                        ("id", "topic", "evidence", "teachingLanguage", "sourceLanguage")
                        if teaching_language.lower().split("-")[0] != "en" else
                        ("id", "topic", "evidence", "teachingLanguage", "sourceLanguage",
                         "acronymExpansion", "knownPitfalls"))}],
                               ensure_ascii=False),
                    max_tokens=2400, temperature=0.25,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("Deep lesson call failed for %s: %s", item["id"], exc)
            entries: dict = {}
            if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
                entries = {str(item["id"]).lower(): data[0]}
            if isinstance(data, dict):
                entries = {k.lower(): v for k, v in data.items() if isinstance(v, dict) and k.lower().startswith("c")}
                if not entries and isinstance(data.get("id"), str):
                    entries = {data["id"].lower(): data}
                if not entries:
                    for field in LESSON_FIELDS + ("provenance",):
                        flat = data.get(field) or {}
                        if isinstance(flat, dict):
                            for k, v in flat.items():
                                entries.setdefault(str(k).lower(), {})[field] = v
            ev_text = " ".join(e.get("text", "") for e in item["evidence"])
            for cid, obj in entries.items():
                ok, errors = validate_lesson(obj, ev_text, item["topic"], item["evidence"], teaching_language)
                if ok:
                    stamped = finish(obj, item)
                    ok2, errors2 = validate_deep_teaching(stamped, item["topic"], teaching_language)
                    if ok2:
                        lesson = stamped
                        break
                    errors = errors2
                log.info("deep lesson rejected %s: %s", cid, "; ".join(errors[:3]))
            if lesson is not None:
                break
            if attempt < max_attempts - 1:
                retry_after = getattr(llm, "LAST_RETRY_AFTER", None)
                status = getattr(llm, "LAST_STATUS", None)
                delay = float(retry_after) if (status == 429 and retry_after) else min(max_delay, base_delay * (2 ** attempt))
                log.info("retrying lesson %s in %.1fs (status=%s)", item["id"], delay, status)
                time.sleep(delay)

        if lesson is None:
            failed.append(item["label"])
            continue
        accepted[item["id"]] = lesson
        try:
            save_lesson(item["key"], {"lesson": lesson, "topic": item["topic"]})
        except Exception as exc:  # noqa: BLE001
            log.warning("lesson checkpoint failed for %s: %s", item["id"], exc)
    return accepted, failed



def label_map(mindmap: dict, timeout: float | None = None) -> dict:
    """
    Return {"ops": [...], "meta": {...}, "stats": {...}} for the one-call rewrite.
    Never raises; with no LLM, a failure or a timeout, `ops` only contains what arrived.
    """
    started = time.perf_counter()
    # The faithfulness encoder does not depend on the LLM answer: fit it while the call runs.
    texts = [e["text"] for e in mindmap.get("transcript") or []]
    encoder_pool = ThreadPoolExecutor(max_workers=1)
    encoder_future = encoder_pool.submit(Embedder, texts) if texts else None
    title = mindmap.get("meta", {}).get("title") or mindmap["root"]["text"]
    sections = collect(mindmap)
    by_cid = {s.cid: s for s in sections}
    concepts = {c.cid: c for s in sections for c in s.concepts}
    details = {d.cid: d for s in sections for d in s.details}
    stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "llm_ms": 0, "cached_sections": 0, "malformed": 0, "errors": []}
    merged = Parsed()
    provider = llm.fast_provider()
    remote_llm = bool(provider and "llm.test" not in provider[0])
    is_deep = (mindmap.get("meta") or {}).get("mode") == "deep"
    teaching_language = (mindmap.get("meta") or {}).get("teachingLanguage") or "en"

    # 1. cache lookup ----------------------------------------------------------------
    todo: list[SectionCand] = []
    for s in sections:
        hit = _cache_read(s.key)
        if hit is None:
            todo.append(s)
            continue
        stats["cached_sections"] += 1
        if hit.get("title") and not s.fixed:
            merged.sections[s.cid] = hit["title"]
        if hit.get("summary"):
            merged.section_summaries[s.cid] = hit["summary"]
        for c, label in zip(s.concepts, hit.get("concepts") or []):
            if label:
                merged.concepts[c.cid] = label
        for d, item in zip(s.details, hit.get("details") or []):
            if item:
                merged.details[d.cid] = (item[0], item[1])
        for key, target in (("explanations", merged.explanations), ("examples", merged.examples),
                            ("diagrams", merged.diagrams), ("exercises", merged.exercises)):
            for cid, value in (hit.get(key) or {}).items():
                target[cid] = value
    root_key = _root_key(mindmap, sections)
    root_hit = _cache_read(root_key)
    if root_hit:
        merged.root = root_hit.get("root")
        merged.summary = root_hit.get("summary")
        for a, b, label in root_hit.get("links") or []:
            merged.links.append((a, b, label))

    # 2. one call (≤3 parallel chunks for very long videos) ---------------------------
    want_root = root_hit is None
    if (todo or want_root) and llm.fast_available() and sections and not is_deep:
        chunks = plan_chunks(todo, title) if todo else [[]]
        outline = [s.node["text"] for s in sections] if (len(chunks) > 1 or not todo or len(todo) < len(sections)) else None

        deadline = timeout or (max(settings.label_timeout, 45.0) if remote_llm else settings.label_timeout)
        budget = max(200, (deadline - 0.3) * OUTPUT_TOKENS_PER_SECOND)

        def run(i_chunk: tuple[int, list[SectionCand]]):
            i, chunk = i_chunk
            root_here = want_root and i == 0
            # Always keep the tagged D lines when the original D-line budget allowed them;
            # the new study-aid lines are short and must not crowd out existing detail rewrites.
            with_details = answer_tokens(chunk, root_here, False) <= budget
            prompt = build_prompt(title, chunk, root_here, outline, with_details=with_details)
            n_lines = 2 + 3 * len(chunk) + sum(3 * len(s.concepts) + (len(s.details) if with_details else 0) for s in chunk) + MAX_LINKS
            mt = max_tokens(n_lines)
            if remote_llm:
                mt = min(2600, 200 + 26 * n_lines)
            return chunk, llm.complete(SYSTEM, prompt, max_tokens=mt, temperature=0.2, timeout=deadline)

        with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
            results = list(pool.map(run, enumerate(chunks)))

        fresh_links: list[tuple[str, str, str]] = []
        for chunk, comp in results:
            stats["calls"] += 1 if comp.sent else 0
            stats["prompt_tokens"] += comp.prompt_tokens
            stats["completion_tokens"] += comp.completion_tokens
            stats["llm_ms"] = max(stats["llm_ms"], round(comp.ms))
            if comp.error:
                stats["errors"].append(comp.error)
            text = comp.text
            if not comp.ok and not text.endswith("\n"):  # cut off mid-line: drop the unfinished label
                text = text.rsplit("\n", 1)[0] if "\n" in text else ""
            parsed = parse_labels(text, teaching_language)
            parsed.details = _resolve_details(parsed, details)
            stats["malformed"] += parsed.malformed
            if want_root and merged.root is None and parsed.root:
                merged.root = parsed.root
            if want_root and merged.summary is None and parsed.summary:
                merged.summary = parsed.summary
            for sid, label in parsed.sections.items():
                if sid in by_cid and not by_cid[sid].fixed:
                    merged.sections.setdefault(sid, label)
            for sid, summary in parsed.section_summaries.items():
                if sid in by_cid:
                    merged.section_summaries.setdefault(sid, summary)
            for cid, label in parsed.concepts.items():
                if cid in concepts:
                    merged.concepts.setdefault(cid, label)
            for did, item in parsed.details.items():
                merged.details.setdefault(did, item)
            for src, dst in ((parsed.explanations, merged.explanations), (parsed.examples, merged.examples),
                             (parsed.diagrams, merged.diagrams), (parsed.exercises, merged.exercises)):
                for key, value in src.items():
                    dst.setdefault(key, value)
            fresh_links += parsed.links
            for s in chunk:
                # a complete answer caches every section; a cut-off one only the sections that fully
                # arrived, so the next open asks only for the rest
                complete = (s.fixed or s.cid in parsed.sections) and all(c.cid in parsed.concepts for c in s.concepts)
                if comp.ok or (complete and s.concepts):
                    _cache_write(s.key, {
                        "title": parsed.sections.get(s.cid),
                        "summary": parsed.section_summaries.get(s.cid),
                        "concepts": [parsed.concepts.get(c.cid) for c in s.concepts],
                        "details": [list(parsed.details[d.cid]) if d.cid in parsed.details else None for d in s.details],
                        "explanations": {c.cid: parsed.explanations[c.cid] for c in s.concepts if c.cid in parsed.explanations},
                        "examples": {c.cid: parsed.examples[c.cid] for c in s.concepts if c.cid in parsed.examples},
                        "diagrams": {s.cid: parsed.diagrams[s.cid]} if s.cid in parsed.diagrams else {},
                        "exercises": {s.cid: parsed.exercises[s.cid]} if s.cid in parsed.exercises else {},
                    })
                    stats.setdefault("sections_cached", 0)
                    stats["sections_cached"] += 1
        merged.links += fresh_links
        if want_root and merged.root and (results[0][1].ok or merged.summary):
            _cache_write(root_key, {"root": merged.root, "summary": merged.summary, "links": [list(link) for link in fresh_links]})

    test_provider = bool(provider and "llm.test" in provider[0])
    deep_status = "n/a"
    if is_deep and concepts and llm.fast_available() and not test_provider:
        try:
            lessons, failed_lessons = _llm_deep_lessons(mindmap, concepts)
            if not hasattr(merged, "lessons"):
                merged.lessons = {}
            if not hasattr(merged, "available"):
                merged.available = {}
            if not hasattr(merged, "generatedBy"):
                merged.generatedBy = {}
            if not hasattr(merged, "unavailable"):
                merged.unavailable = {}
            for cid, lesson in lessons.items():
                merged.lessons[cid] = lesson
                merged.explanations[cid] = lesson.get("explain", "")
                merged.available[cid] = True
                merged.generatedBy[cid] = "llm-deep"
            for cid in concepts:
                if cid not in lessons:
                    merged.available[cid] = False  # fail closed: no fabricated teaching
                    merged.unavailable[cid] = "Lesson generation failed; concept kept with explicit unavailable status."
            stats["lesson_failures"] = failed_lessons
            stats["lessonsAccepted"] = len(lessons)
            stats["lessonsRejected"] = len(failed_lessons)
            stats["lessonsAttempted"] = len(lessons) + len(failed_lessons)
            stats["provider"] = llm.provider_name()
            stats["teachingLanguage"] = (mindmap.get("meta") or {}).get("teachingLanguage") or "en"
            deep_status = "ok" if not failed_lessons else f"incomplete: {len(failed_lessons)} concepts rejected"
            if failed_lessons:
                log.warning("Deep lessons failed for: %s", ", ".join(failed_lessons))
            stats["calls"] += 1
        except Exception as exc:  # noqa: BLE001
            deep_status = "failed: provider or validation error"
            log.warning("Deep lesson generation failed: %s", exc)
    if is_deep and deep_status == "n/a":
        deep_status = ("unavailable: no LLM provider configured" if not llm.fast_available()
                       else "unavailable: no concepts with evidence")
    if concepts and llm.fast_available() and not test_provider and not is_deep:
        try:
            enriched = _llm_enrich(mindmap, concepts, details)
            for key, target in (("explanations", merged.explanations), ("examples", merged.examples),
                                ("exercises", merged.exercises)):
                for cid, value in (enriched.get(key) or {}).items():
                    target[cid] = value
            if not hasattr(merged, "mechanisms"):
                merged.mechanisms = {}
                merged.titles = {}
                merged.available = {}
            for cid, value in (enriched.get("mechanisms") or {}).items():
                merged.mechanisms[cid] = value
            for cid, value in (enriched.get("titles") or {}).items():
                merged.titles[cid] = value
            for cid, value in (enriched.get("available") or {}).items():
                merged.available[cid] = value
            if not hasattr(merged, "diagrams"):
                merged.diagrams = {}
            for cid, value in (enriched.get("diagrams") or {}).items():
                merged.diagrams[cid] = value
            if not hasattr(merged, "generatedBy"):
                merged.generatedBy = {}
            for cid, value in (enriched.get("generatedBy") or {}).items():
                merged.generatedBy[cid] = value
            for key in ("intuitions", "why_it_matters", "misconceptions", "knowledge_checks"):
                if not hasattr(merged, key):
                    setattr(merged, key, {})
                getattr(merged, key).update(enriched.get(key) or {})
            stats["calls"] += 1
        except Exception as exc:  # noqa: BLE001
            log.warning("NVIDIA study-aid enrichment failed: %s", exc)

    # 3. labels -> ops ---------------------------------------------------------------------
    apply_started = time.perf_counter()
    ops, changed = _to_ops(mindmap, sections, concepts, details, merged, encoder_future.result() if encoder_future else None)
    encoder_pool.shutdown(wait=False)
    stats["apply_ms"] = round((time.perf_counter() - apply_started) * 1000)
    stats["applied"] = len([o for o in ops if o["type"] == "update"])
    stats["candidates"] = 2 + sum(1 for s in sections if not s.fixed) + len(concepts) + len(details)
    stats["ms"] = round((time.perf_counter() - started) * 1000)
    # an answer cut by the deadline is accepted as it is (no re-call on every reopen);
    # hard failures (429, 5xx, network) leave the map unlabelled so a later open can retry
    labelled = bool(ops) and not [e for e in stats["errors"] if e != "deadline"]
    log.info(
        "labels %s llm_calls=%d prompt_tokens=%d completion_tokens=%d llm_ms=%d applied=%d/%d cached_sections=%d/%d malformed=%d errors=%s apply_ms=%d total_ms=%d",
        mindmap.get("meta", {}).get("videoId"), stats["calls"], stats["prompt_tokens"], stats["completion_tokens"], stats["llm_ms"],
        stats["applied"], stats["candidates"], stats["cached_sections"], len(sections), stats["malformed"], stats["errors"] or "none", stats["apply_ms"], stats["ms"],
    )
    meta = {"labelled": labelled or (not todo and root_hit is not None)}
    if is_deep:
        meta["deepStatus"] = deep_status
        meta["provider"] = llm.provider_name()
    if stats["calls"] or stats["cached_sections"]:
        meta["llm"] = llm.fast_name()
    return {"ops": ops, "meta": meta, "stats": stats}




def _norm(text: str) -> str:
    return unicode_norm(text)


def _missing_topic_ops(mindmap: dict) -> list[dict]:
    """Add an evidence-backed concept for any essential topic the extractor missed."""
    ops = []
    concepts = [n for sec in mindmap["root"].get("children", []) for n in sec.get("children", [])
                if n.get("type") == "concept"]
    present = " ".join((n.get("topic", "") + " " + n.get("text", "")).lower() for n in concepts)
    transcript = mindmap.get("transcript") or []
    for label, aliases in HTTP_TOPICS:
        if label.lower() in present:
            continue
        candidates = []
        for e in transcript:
            text = (e.get("text") or "").strip()
            if len(text) < 35:
                continue
            low = text.lower()
            hits = [a for a in aliases if len(a) >= 4 and re.search(r"(?<![a-z])" + re.escape(a) + r"(?![a-z])", low)]
            if hits:
                candidates.append((len(hits), len(text), e))
        if not candidates:
            continue
        entry = max(candidates, key=lambda x: (x[0], x[1]))[2]
        section = next((s for s in mindmap["root"].get("children", []) if not s.get("recall")), None)
        if not section:
            continue
        start = float(entry.get("start", 0) or 0)
        node = make_node(
            label, "concept", 2, start, float(entry.get("end", start + 1) or start + 1),
            source=(entry.get("text") or "")[:300],
            topic=label,
            evidence=[{"start": start, "end": float(entry.get("end", start + 1) or start + 1),
                       "text": (entry.get("text") or "")[:300]}],
            explain="Verified transcript evidence: " + (entry.get("text") or "")[:240],
            mechanism="Evidence: " + (entry.get("text") or "")[:240],
            example="Observed at " + str(round(start, 1)) + "s: " + (entry.get("text") or "")[:180],
            exercise="Using the evidence at " + str(round(start, 1)) + "s, explain " + label + " in your own words.",
            available=True,
            generatedBy="evidence",
            faith=1.0,
        )
        ops.append({"type": "add", "parentId": section["id"], "index": len(section.get("children", [])), "node": node})
    return ops


def _to_ops(mindmap: dict, sections: list[SectionCand], concepts: dict[str, ConceptCand], details: dict[str, DetailCand],
            parsed: Parsed, encoder: Embedder | None = None) -> tuple[list[dict], list[dict]]:
    root = mindmap["root"]
    by_cid = {s.cid: s for s in sections}
    patches: dict[str, dict] = {}  # node id -> patch (insertion order = display order)
    nodes: dict[str, dict] = {}

    def propose(node: dict, **patch) -> None:
        patch = {k: v for k, v in patch.items() if v is not None and v != node.get(k)}
        if patch:
            nodes[node["id"]] = node
            patches.setdefault(node["id"], {}).update(patch)

    new_titles = [parsed.sections.get(s.cid, s.node["text"]) for s in sections if not s.recall]
    if parsed.root:
        propose(root, text=parsed.root, summary=parsed.summary or " · ".join(new_titles[:6]))
    elif parsed.summary:
        propose(root, summary=parsed.summary)
    for s in sections:
        title = truncate(parsed.sections[s.cid], 60) if s.cid in parsed.sections and not s.fixed else None
        summary = parsed.section_summaries.get(s.cid)
        if s.recall:
            summary = parsed.summary
            line = next((c for c in s.node.get("children", []) if c.get("oneline")), None)
            if line is not None and parsed.summary:
                propose(line, text=f"In one line: {parsed.summary}")
        propose(s.node, text=title, summary=summary)
    for cid, label in parsed.concepts.items():
        node = concepts[cid].node
        patch = {"text": truncate(label, 48)}
        if cid in getattr(parsed, "titles", {}):
            patch["text"] = truncate(parsed.titles[cid].split(":", 1)[0].strip(), 48)
        if cid in parsed.explanations:
            patch.update(explain=parsed.explanations[cid], generatedBy="llm")
        if cid in getattr(parsed, "intuitions", {}):
            patch["intuition"] = parsed.intuitions[cid]
        if cid in getattr(parsed, "why_it_matters", {}):
            patch["why_it_matters"] = parsed.why_it_matters[cid]
        if cid in getattr(parsed, "misconceptions", {}):
            patch["misconception"] = parsed.misconceptions[cid]
        if cid in getattr(parsed, "knowledge_checks", {}):
            patch["knowledge_check"] = parsed.knowledge_checks[cid]
        if cid in getattr(parsed, "mechanisms", {}):
            patch["mechanism"] = parsed.mechanisms[cid]
        if cid in getattr(parsed, "diagrams", {}):
            patch["diagram"] = parsed.diagrams[cid]
        if cid in getattr(parsed, "diagrams", {}):
            patch["diagram"] = parsed.diagrams[cid]
        if cid in parsed.examples:
            patch["example"] = parsed.examples[cid]
        if cid in parsed.exercises:
            patch["exercise"] = parsed.exercises[cid]
        if cid in getattr(parsed, "available", {}):
            patch["available"] = bool(parsed.available[cid])
        if getattr(parsed, "unavailable", {}).get(cid):
            patch.update({
                "conceptUnavailable": True, "available": False, "lesson_available": False,
                "explain": "", "steps": [], "worked_example": "", "exercise": "",
                "diagram": "", "misconception": "", "knowledge_check": {},
                "generatedBy": "unavailable",
                "unavailableReason": str(parsed.unavailable[cid])[:200],
            })
        if cid in getattr(parsed, "lessons", {}):
            lesson = parsed.lessons[cid]
            patch.update({
                "text": truncate(str(lesson.get("title") or node.get("text", "")).split(":", 1)[0].strip(), 48),
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
                "generatedBy": "llm-deep",
            })
        propose(node, **patch)
    for cid, c in concepts.items():
        if cid in parsed.concepts:
            continue
        node = c.node
        patch = {}
        if cid in parsed.explanations:
            patch["explain"] = parsed.explanations[cid]
        if cid in getattr(parsed, "intuitions", {}):
            patch["intuition"] = parsed.intuitions[cid]
        if cid in getattr(parsed, "why_it_matters", {}):
            patch["why_it_matters"] = parsed.why_it_matters[cid]
        if cid in getattr(parsed, "misconceptions", {}):
            patch["misconception"] = parsed.misconceptions[cid]
        if cid in getattr(parsed, "knowledge_checks", {}):
            patch["knowledge_check"] = parsed.knowledge_checks[cid]
        if cid in getattr(parsed, "mechanisms", {}):
            patch["mechanism"] = parsed.mechanisms[cid]
        if cid in getattr(parsed, "diagrams", {}):
            patch["diagram"] = parsed.diagrams[cid]
        if cid in parsed.examples:
            patch["example"] = parsed.examples[cid]
        if cid in parsed.exercises:
            patch["exercise"] = parsed.exercises[cid]
        if cid in getattr(parsed, "titles", {}):
            patch["text"] = truncate(parsed.titles[cid].split(":", 1)[0].strip(), 48)
        if cid in getattr(parsed, "available", {}):
            patch["available"] = bool(parsed.available[cid])
        if getattr(parsed, "unavailable", {}).get(cid):
            patch.update({
                "conceptUnavailable": True, "available": False, "lesson_available": False,
                "explain": "", "steps": [], "worked_example": "", "exercise": "",
                "diagram": "", "misconception": "", "knowledge_check": {},
                "generatedBy": "unavailable",
                "unavailableReason": str(parsed.unavailable[cid])[:200],
            })
        if cid in getattr(parsed, "lessons", {}):
            lesson = parsed.lessons[cid]
            patch.update({
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
                "generatedBy": "llm-deep",
            })
        if patch:
            patch["generatedBy"] = getattr(parsed, "generatedBy", {}).get(cid, "llm")
            propose(node, **patch)
    for sid, text in parsed.diagrams.items():
        if sid in by_cid:
            propose(by_cid[sid].node, diagram=text, generatedBy="llm")
    for sid, text in parsed.exercises.items():
        if sid in by_cid:
            propose(by_cid[sid].node, exercise=text, generatedBy="llm")
    for did, (tag, text) in parsed.details.items():
        d = details[did]
        tag = tag if tag in TAGS else d.node.get("tag")
        propose(d.node, text=truncate(f"{tag}: {text}" if tag else text, 120), tag=tag)

    # no duplicate labels: a rewrite that repeats another node's text keeps its heuristic text
    seen: dict[str, str] = {}
    for node in [root, *(s.node for s in sections), *(c.node for c in concepts.values()), *(d.node for d in details.values())]:
        text = patches.get(node["id"], {}).get("text", node["text"])
        key = _norm(text)
        if key in seen and "text" in patches.get(node["id"], {}):
            patches[node["id"]].pop("text")
        else:
            seen.setdefault(key, node["id"])

    ops: list[dict] = []
    changed: list[dict] = []
    for nid, patch in patches.items():
        if not patch:
            continue
        ops.append({"type": "update", "id": nid, "patch": patch})
        if "text" in patch:
            changed.append({**nodes[nid], **patch})

    # faithfulness of the new labels against their own transcript spans
    if changed and encoder is not None:
        for node in changed:
            if node.get("type") == "concept":
                node["_faith_text"] = " ".join(str(node.get(k) or "") for k in ("explain", "mechanism", "example", "text"))[:800]
        entries = mindmap.get("transcript") or []
        spans = {}
        for n in changed:
            if n.get("start") is None or n.get("end") is None:
                continue
            if n.get("type") == "section":
                spans[n["id"]] = " ".join(e["text"] for e in entries if n["start"] - 0.5 <= e["start"] <= n["end"])[:6000]
            elif n.get("type") == "concept":
                spans[n["id"]] = (n.get("source") or "") + " " + " ".join(
                    (e.get("text") or "") for e in (n.get("evidence") or [])[:3])[:6000]
        score_faithfulness(changed, encoder, spans)
        faith = {n["id"]: n.get("faith") for n in changed}
        for op in ops:
            if op.get("id") and faith.get(op["id"]) is not None:
                op["patch"]["faith"] = faith[op["id"]]

    # cross-links: LLM relation labels between concepts of different sections


    edges = mindmap.get("edges") or []
    for a, b, label in parsed.links[:MAX_LINKS]:
        ca, cb = concepts.get(a), concepts.get(b)
        if not ca or not cb or ca.sid == cb.sid or by_recall(ca, cb, sections):
            continue
        existing = next((e for e in edges if {e["source"], e["target"]} == {ca.node["id"], cb.node["id"]}), None)
        if existing and existing.get("label") != "related to":
            continue
        if existing:
            ops.append({"type": "edge:remove", "id": existing["id"]})
        ops.append({"type": "edge:add", "edge": {"id": new_id("e"), "source": ca.node["id"], "target": cb.node["id"], "label": label, "origin": "llm"}})
    return ops, changed


def by_recall(a: ConceptCand, b: ConceptCand, sections: list[SectionCand]) -> bool:
    recall = {s.cid for s in sections if s.recall}
    return a.sid in recall or b.sid in recall


def apply_outcome(mindmap: dict, outcome: dict) -> dict:
    """Apply label ops + meta to a map in place (server-side cache / CLI)."""
    for op in outcome.get("ops", []):
        apply_op(mindmap, op)
    mindmap.setdefault("meta", {}).update(outcome.get("meta", {}))
    stats = outcome.get("stats") or {}
    mindmap["meta"]["labelStats"] = {
        k: stats.get(k) for k in (
            "calls", "prompt_tokens", "completion_tokens", "llm_ms", "applied", "candidates",
            "lessonsAttempted", "lessonsAccepted", "lessonsRejected", "lesson_failures",
            "provider", "teachingLanguage",
        )
    }
    mindmap["meta"]["generationMs"] = stats.get("ms")
    return mindmap
