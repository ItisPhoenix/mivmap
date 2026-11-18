"""
Layer 4 + 5 glue — turns segments + concepts into the hierarchical mindmap JSON.

The map is a last-minute revision sketchbook:
    layer 0  root        topic of the video (one-line summary underneath)
    layer 1  section     numbered sections in video order, each with a time range and a
                         1–2 sentence summary picked by the extractive summary retriever
    layer 2  concept     key concepts as "Term: one-line meaning"
    layer 3  detail      lazy "+" items tagged Def / Eg / Formula / Tip / Watch-out
    layer 4  transcript  verbatim grounding quotes (kept in the JSON, not shown in the viewer)
    + a final "Quick recall" section: a one-line summary and 3–6 must-remember points

Non-English transcripts are translated to English ONCE (NLLB-200, see `translate.py`) right
after the transcript fetch, so every later stage — and every node — works on English text.

The whole tree is built from the transcript WITHOUT an LLM (`build_skeleton`, well under
a second). Every node carries a real transcript span (`start`/`end`), the exact
sentence it came from (`source`) and a faithfulness score (`faith`). An LLM, if
configured, is then called ONCE to rewrite the labels (see `labels.py`) — it never writes
the tree, timestamps or transcript leaves.
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
import uuid
from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import llm, translate
from .concepts import VAGUE_LABELS, ConceptGraph, extract_concepts
from .embeddings import Embedder, backend_name, cosine_matrix
from .segmentation import Segment, segment_transcript
from .topics import ESSENTIAL_LABELS, HTTP_TOPICS, supplement_concepts
from .text_utils import FILLER_WORDS, VERB_FORMS, fmt_time, parse_description_chapters, smart_title, truncate
from .tone import detect_tone, merge_tone
from .transcript import Transcript, TranscriptEntry, get_transcript, sentences_with_time

log = logging.getLogger("mivmap.builder")
ProgressFn = Callable[[str, float], None]

# Legacy schema identifier: kept verbatim so existing exported maps and saved documents still load in Mivmap.
SCHEMA = "tubemind/1"


@dataclass(frozen=True)
class ModeConfig:
    concepts: int
    details: int
    leaves: int
    detail_words: int
    expanded: bool = False  # everything shown: no collapsed concepts, no lazy "+", untagged supporting lines too


MODES = {
    # Short (default): exam-revision density — few, punchy nodes; level 3 behind "+N"
    "revision": ModeConfig(concepts=5, details=3, leaves=1, detail_words=14),
    # Standard: more concepts and tagged items; level 3 behind "+N"
    "academic": ModeConfig(concepts=7, details=4, leaves=2, detail_words=16),
    # Detailed: the full map, nothing hidden
    "deep": ModeConfig(concepts=3, details=2, leaves=0, detail_words=20, expanded=True),
}
DEFAULT_MODE = "revision"
SUMMARY_POINTS = 3  # extractive summary retriever: representative sentences per section
TAGS = ("Def", "Formula", "Eg", "Tip", "Watch-out")  # level-3 type tags, in display order
TRANSCRIPT_MARKERS = re.compile(
    r"\b(?:revisit|number of reasons|first being|let's|we're going to|welcome back|"
    r"take a look|going to talk|today we|so here|now let's)\b", re.I)
VAGUE_LABELS = {"revisit", "revisit http", "foundations", "foundation", "additional", "information",
                "message", "messages", "overview", "topic", "topics", "part", "parts", "web server",
                "server", "general", "other", "others", "something", "example", "examples"}


def new_id(prefix: str = "n") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def make_node(text: str, type_: str, layer: int, start: float | None = None, end: float | None = None, **extra) -> dict:
    node = {
        "id": new_id(),
        "text": text,
        "type": type_,
        "layer": layer,
        "start": None if start is None else round(float(start), 2),
        "end": None if end is None else round(float(end), 2),
        "summary": "",
        "tone": [],
        "keywords": [],
        "notes": "",
        "links": [],
        "image": None,
        "collapsed": False,
        "children": [],
    }
    node.update(extra)
    return node


# ---------------------------------------------------------------------------
def prune_weak_concepts(mindmap: dict) -> None:
    """Reject concepts whose teaching text is not grounded in their transcript span."""
    essential = {e.lower() for e in ESSENTIAL_LABELS}
    is_deep = (mindmap.get("meta") or {}).get("mode") == "deep"
    for sec in mindmap.get("root", {}).get("children", []):
        for c in sec.get("children", []):
            if c.get("type") != "concept" or not c.get("evidence"):
                continue
            if is_deep and not c.get("lesson_available"):
                continue
            ev = c["evidence"][0]
            c["start"] = round(float(ev.get("start") or 0), 2)
            c["end"] = round(float(ev.get("end") or c["start"] + 1), 2)
            c["source"] = ev.get("text", c.get("source", ""))[:300]
            topic = _norm(c.get("topic") or "")
            evidence = _norm(c.get("source") or "")
            teaching = _norm(" ".join(str(c.get(k) or "") for k in ("text", "explain", "mechanism", "example", "exercise")))
            if (topic and topic not in essential
                    and not any(w in teaching for w in topic.split())):
                c["topic"] = c.get("text") or c.get("topic")
            for child in c.get("children", []):
                if child.get("start") is None:
                    continue
                if child["start"] < c["start"]:
                    # Keep the detail but anchor it to the concept's verified passage.
                    child["start"] = c["start"]
                    child["end"] = c["end"]
        kept = []
        for c in sec.get("children", []):
            if c.get("type") != "concept":
                kept.append(c)
                continue
            if c.get("available") is False:
                continue  # reject concepts without verified teaching content
            if is_deep and not c.get("lesson_available"):
                continue  # fail closed: no validated lesson, no concept
            if c.get("faith", 1.0) >= 0.30:
                kept.append(c)
                continue
            if (c.get("topic") or "").lower() in essential and c.get("evidence"):
                c["source"] = c["evidence"][0].get("text", c.get("source", ""))[:300]
                c["faith"] = 1.0
                kept.append(c)
        sec["children"] = kept


def ensure_essential_topics(mindmap: dict, entries: list[TranscriptEntry]) -> None:
    """Add domain key topics that were not extracted, using strong transcript evidence only."""
    if not any("http" in e.text.lower() for e in entries[:80]):
        return
    sections = [s for s in mindmap["root"].get("children", []) if not s.get("recall")]
    if not sections:
        return
    present = {_norm(c.get("topic") or c.get("text") or "") for s in sections for c in s.get("children", [])
               if c.get("type") == "concept"}
    for label, _patterns in HTTP_TOPICS:
        aliases = [w for w in re.findall(r"[a-z]{4,}", label.lower()) if w not in {"and", "the", "for"}]
        best = None
        for e in entries:
            text = e.text.strip()
            if len(text) < 35:
                continue
            hits = sum(1 for a in aliases if re.search(r"(?<![a-z])" + re.escape(a) + r"(?![a-z])", text, re.I))
            score = hits * 3 + (1 if COPULA_RE.search(text) else 0) + min(len(text) // 60, 3)
            if hits and (best is None or score > best[0]):
                best = (score, e)
        if not best:
            continue
        e = best[1]
        section = min(sections, key=lambda s: abs((s.get("start") or 0) - e.start))
        if _norm(label) in present:
            # repair an existing concept whose evidence does not support its topic
            for sec in sections:
                for node in sec.get("children", []):
                    if node.get("type") != "concept" or _norm(node.get("topic") or node.get("text") or "") != _norm(label):
                        continue
                    evidence = _norm((node.get("evidence") or [{}])[0].get("text", "") or node.get("source", ""))
                    if any(w in evidence for w in _norm(label).split()):
                        break
                    node["start"] = round(e.start, 2)
                    node["end"] = round(e.end, 2)
                    node["source"] = truncate(e.text, 300)
                    node["evidence"] = [{"start": round(e.start, 2), "end": round(e.end, 2), "text": truncate(e.text, 300)}]
                    node["faith"] = 1.0
                    node.update({"explain": "", "mechanism": "", "example": "", "exercise": "", "diagram": "",
                                 "available": False, "generatedBy": "unavailable"})
            continue
        node = make_node(
            smart_title(label), "concept", 2, e.start, e.end,
            source=truncate(e.text, 300),
            topic=label,
            evidence=[{"start": round(e.start, 2), "end": round(e.end, 2), "text": truncate(e.text, 300)}],
            explain="", mechanism="", example="", exercise="", diagram="",
            available=False, generatedBy="unavailable",
        )
        section["children"].append(node)


def dedupe_concepts(mindmap: dict) -> None:
    """Remove semantically equivalent concepts across the whole map."""
    import difflib

    all_nodes = [(sec, node) for sec in mindmap.get("root", {}).get("children", [])
                 for node in sec.get("children", [])]
    seen: list[tuple[dict, dict]] = []
    for sec, node in all_nodes:
        if node.get("type") != "concept":
            continue
        key = _norm(node.get("topic") or node.get("text") or "")
        title = _norm(node.get("text") or "")
        duplicate = None
        for other_sec, other in seen:
            okey = _norm(other.get("topic") or other.get("text") or "")
            otitle = _norm(other.get("text") or "")
            if key and key == okey:
                duplicate = (other_sec, other)
                break
            if title and otitle and difflib.SequenceMatcher(None, title, otitle).ratio() >= 0.92:
                duplicate = (other_sec, other)
                break
        if duplicate is None:
            seen.append((sec, node))
            continue
        other_sec, other = duplicate
        better = (bool(node.get("available")) and not bool(other.get("available"))) or \
                 (node.get("faith", 0) > other.get("faith", 0))
        if better:
            other_sec["children"] = [c for c in other_sec.get("children", []) if c is not other]
            seen[seen.index(duplicate)] = (sec, node)
        else:
            sec["children"] = [c for c in sec.get("children", []) if c is not node]



def finalize_concepts(mindmap: dict) -> None:
    """Resolve identity, evidence, timestamp and section ownership together.

    Transcript leaves keep their own exact ranges; only concept anchors are aligned to
    the passage that actually supports the concept. Faithfulness is measured, never
    assumed, and weak or duplicated concepts are discarded.
    """
    import difflib

    root = mindmap.get("root") or {}
    sections = [s for s in root.get("children", []) if not s.get("recall")]
    if not sections:
        return
    is_deep = (mindmap.get("meta") or {}).get("mode") == "deep"

    def owning_section(t: float) -> dict:
        for sec in sections:
            st, en = sec.get("start"), sec.get("end")
            if st is not None and en is not None and st - 2 <= t <= en + 2:
                return sec
        return min(sections, key=lambda sec: abs((sec.get("start") or 0) - t))

    def topic_words(topic: str) -> list[str]:
        return [w for w in _norm(topic).split() if len(w) >= 3]

    # 1) Repair every invalid range first (end must never precede start).
    for node in _walk(root):
        st, en = node.get("start"), node.get("end")
        if st is None or en is None:
            continue
        en = float(en)
        if en < float(st):
            node["end"] = round(float(st), 2)

    # 2) Evidence-based identity + timestamp + ownership for concepts.
    concepts = [(sec, node) for sec in sections for node in sec.get("children", [])
                if node.get("type") == "concept"]
    for sec, node in concepts:
        evidence = node.get("evidence") or []
        if not evidence and node.get("source"):
            evidence = [{"start": node.get("start"), "end": node.get("end"), "text": node.get("source", "")}]
        if not evidence:
            sec["children"] = [c for c in sec.get("children", []) if c is not node]
            continue
        topic = node.get("topic") or node.get("text") or ""
        words = topic_words(topic)

        def evidence_score(e: dict) -> float:
            text = _norm(e.get("text") or "")
            hit = sum(1 for w in words if w in text) / max(1, len(words))
            return hit + min(0.5, len(text) / 400)

        best = max(evidence, key=evidence_score)
        start = float(best.get("start") or 0)
        end = float(best.get("end") or (start + 1))
        if end < start:
            end = start + 1
        node["evidence"] = [best] + [e for e in evidence if e is not best]
        node["start"] = round(start, 2)
        node["end"] = round(end, 2)
        node["source"] = (best.get("text") or "")[:300]
        owner = owning_section(start)
        if owner is not None and owner is not sec:
            sec["children"] = [c for c in sec.get("children", []) if c is not node]
            owner["children"].append(node)
        # Measure grounding: topic presence in evidence + similarity to the source.
        teaching = _norm(" ".join(str(node.get(k) or "") for k in ("text", "explain", "steps", "worked_example")))
        evidence_text = _norm(best.get("text") or "")
        topic_hit = sum(1 for w in words if w in evidence_text) / max(1, len(words))
        similarity = difflib.SequenceMatcher(None, teaching, evidence_text).ratio()
        node["faith"] = round(min(0.95, max(0.0, 0.6 * topic_hit + 0.4 * min(1.0, similarity * 2))), 2)

    # 3) Fail closed: in Deep Mode, keep only concepts with a validated lesson.
    for sec in sections:
        sec["children"] = [c for c in sec.get("children", [])
                           if c.get("type") != "concept"
                           or (c.get("evidence") and (not is_deep or c.get("lesson_available")))]

    # 3b) Detail timestamps must fall inside their concept's verified span.
    for sec in sections:
        for node in sec.get("children", []):
            if node.get("type") != "concept":
                continue
            for child in node.get("children", []):
                if child.get("start") is None:
                    continue
                if child["start"] < node.get("start", 0) or child["start"] > node.get("end", 0):
                    child["start"] = node["start"]
                    child["end"] = node["end"]

    # 4) Semantic deduplication across sections.
    seen: list[tuple[dict, dict]] = []
    for sec in sections:
        for node in list(sec.get("children", [])):
            if node.get("type") != "concept":
                continue
            key = _norm(node.get("topic") or node.get("text") or "")
            title = _norm(node.get("text") or "")
            teaching = _norm(" ".join(str(node.get(k) or "") for k in ("explain", "mechanism", "steps", "worked_example")))
            twin = None
            for other_sec, other in seen:
                okey = _norm(other.get("topic") or other.get("text") or "")
                otitle = _norm(other.get("text") or "")
                oteaching = _norm(" ".join(str(other.get(k) or "") for k in ("explain", "mechanism", "steps", "worked_example")))
                if key and key == okey:
                    twin = (other_sec, other); break
                if title and otitle and difflib.SequenceMatcher(None, title, otitle).ratio() >= 0.90:
                    twin = (other_sec, other); break
                if teaching and oteaching and difflib.SequenceMatcher(None, teaching, oteaching).ratio() >= 0.86:
                    twin = (other_sec, other); break
            if twin is None:
                seen.append((sec, node)); continue
            other_sec, other = twin
            better = len(teaching) > len(_norm(" ".join(str(other.get(k) or "") for k in ("explain", "mechanism", "steps", "worked_example"))))
            if better:
                other_sec["children"] = [c for c in other_sec.get("children", []) if c is not other]
                seen[seen.index(twin)] = (sec, node)
            else:
                sec["children"] = [c for c in sec.get("children", []) if c is not node]



def generate_mindmap(request: dict, progress: ProgressFn | None = None) -> dict:
    """Skeleton + (when enabled) the single label call applied in place. Used by the CLI and tests."""
    from . import labels

    mindmap = build_skeleton(request, progress)
    if request.get("useLLM", True) and llm.fast_available():
        (progress or (lambda m, p: None))("Polishing", 0.9)
        labels.apply_outcome(mindmap, labels.label_map(mindmap))
        finalize_concepts(mindmap)
    else:
        finalize_concepts(mindmap)
    return mindmap


def build_skeleton(request: dict, progress: ProgressFn | None = None) -> dict:
    """Video → Transcript → Segments → Concepts → complete, usable mindmap. No LLM."""
    progress = progress or (lambda msg, pct: None)
    t_start = tick = time.perf_counter()
    timings: dict[str, int] = {}

    def lap(name: str) -> None:
        nonlocal tick
        now = time.perf_counter()
        timings[name] = round((now - tick) * 1000)
        tick = now

    video_id = request["videoId"]
    mode = request.get("mode") if request.get("mode") in MODES else DEFAULT_MODE
    cfg = MODES[mode]

    # 1. Transcript -----------------------------------------------------------
    progress("Fetching transcript", 0.05)
    transcript = get_transcript(
        video_id,
        provided=request.get("transcript"),
        languages=request.get("languages") or None,
        allow_whisper=request.get("allowWhisper", True),
        progress=lambda m, p: progress(m, 0.05 + p * 0.2),
    )
    duration = float(request.get("duration") or transcript.duration)
    lap("transcript")

    # 1b. English only: detect the language once; translate once (cached) BEFORE every other stage
    title = request.get("title") or "YouTube video"
    chapters = [dict(c) for c in (request.get("chapters") or parse_description_chapters(request.get("description", ""), duration))]
    tinfo = translate.to_english(
        video_id, transcript, _language_hint(request, transcript), [title] + [str(c.get("title") or "") for c in chapters],
        min_coverage=float(request.get("minCoverage") or 1.0),  # a click may start from an evenly spread part
    )
    progress("Reading the video", 0.25)
    if tinfo.lang != "en" and tinfo.extras:
        title = tinfo.extras[0] or title
        for chapter, name in zip(chapters, tinfo.extras[1:]):
            chapter["title"] = name or chapter.get("title")
    lap("translate")
    timings["detect"] = round(tinfo.detect_ms)
    timings["translate"] = max(0, timings["translate"] - timings["detect"])

    # 2. Segmentation -----------------------------------------------------------
    progress(f"Segmenting {fmt_time(duration)} of transcript ({transcript.source})", 0.3)
    segments = segment_transcript(transcript, chapters, mode)
    lap("segment")

    # 3. Concepts + knowledge graph ------------------------------------------------
    progress(f"Extracting concepts from {len(segments)} sections", 0.5)
    graph = extract_concepts(segments, max_concepts=max(30, len(segments) * cfg.concepts * 2))
    all_blocks = [block for seg in segments for block in seg.blocks]
    http_domain = any("http" in (b.text or "").lower() for b in all_blocks[:40])
    if http_domain:
        graph.concepts = supplement_concepts(graph.concepts, segments, all_blocks)
    lap("concepts")

    # 4. Tree assembly: one embedding pass over all sentences, batched grounding ------
    progress("Assembling nodes", 0.7)
    index = SentenceIndex(sentences_with_time(transcript.entries))
    clock = {"summary": 0.0}
    used_terms: set[str] = set()
    def _concepts_for(seg: Segment):
        picked = graph.for_segment(seg.id, limit=cfg.concepts * 2)
        if http_domain:
            for c in graph.concepts:
                if c.label in ESSENTIAL_LABELS and seg.id in c.segments and c not in picked:
                    picked.append(c)
        return picked

    sections = [_assemble_section(seg, _concepts_for(seg), index, cfg, clock, used_terms) for seg in segments]
    lap("assemble")
    timings["summary"] = round(clock["summary"])

    root = make_node(
        _short_title(title),
        "root",
        0,
        0,
        duration,
        summary=_heuristic_overview(sections),
        explain=_heuristic_overview(sections),
        generatedBy="extractive",
        tone=merge_tone(detect_tone(transcript.full_text), None),
        keywords=[c.label for c in sorted(graph.concepts, key=lambda c: -c.centrality)[:8]],
    )
    for i, sec in enumerate(sections):
        sec["color"] = i % 5
    recall = _recall_branch(sections, duration)
    root["children"] = sections + ([recall] if recall else [])

    # 5. Cross links (knowledge graph + embedding similarity) + faithfulness ---------
    edges = _cross_edges(root, graph, index)
    score_faithfulness([n for sec in root["children"] for n in _walk(sec)], index.encoder, {sec["id"]: seg.text[:6000] for sec, seg in zip(sections, segments)})
    for sec in sections:
        keep = [c for c in sec["children"]
                if c.get("type") != "concept" or c.get("faith", 1.0) >= 0.20
                or (c.get("text") or "").lower() in {e.lower() for e in ESSENTIAL_LABELS}]
        if len(keep) != len(sec["children"]):
            log.info("pruned %d weak concepts from %s", len(sec["children"]) - len(keep), sec["text"])
            sec["children"] = keep
    lap("edges")

    # 6. Final guard: every node label/summary is English (re-translates stragglers only)
    guarded = translate.guard_map({"root": root}, tinfo.lang) if tinfo.lang != "en" else 0
    lap("guard")
    timings["total"] = round((time.perf_counter() - t_start) * 1000)
    timings["skeleton"] = timings["total"] - timings["transcript"] - timings["detect"] - timings["translate"]

    n_concepts = sum(1 for sec in sections for c in sec["children"] if c["type"] == "concept")
    log.info(
        "skeleton %s mode=%s lang=%s sections=%d concepts=%d | transcript=%dms detect=%dms translate=%dms (%s) segment=%dms concepts=%dms "
        "assemble=%dms summary=%dms edges=%dms guard=%dms(%d fixed) skeleton=%dms total=%dms",
        video_id, mode, tinfo.lang, len(sections), n_concepts, timings["transcript"], timings["detect"], timings["translate"], tinfo.status, timings["segment"],
        timings["concepts"], timings["assemble"], timings["summary"], timings["edges"], timings["guard"], guarded, timings["skeleton"], timings["total"],
    )

    mindmap = {
        "schema": SCHEMA,
        "id": hashlib.sha1(f"{video_id}:{mode}:{time.time()}".encode()).hexdigest()[:16],
        "version": 1,
        "meta": {
            "videoId": video_id,
            "title": title,
            "channel": request.get("channel", ""),
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "duration": duration,
            "mode": mode,
            "language": transcript.language,
            "sourceLanguage": tinfo.lang,
            "translation": tinfo.to_dict(),
            "originalTitle": request.get("title") or "",
            "transcriptSource": transcript.source,
            "llm": "heuristic",
            "labelled": False,
            "embeddings": backend_name(),
            "createdAt": int(time.time() * 1000),
            "buildSeconds": round(timings["total"] / 1000, 2),
            "timings": timings,
            "storyboardSpec": request.get("storyboardSpec"),
        },
        "root": root,
        "edges": edges,
        "segments": [s.to_dict() for s in segments],
        "graph": graph.to_dict(),
        "transcript": [{"start": round(e.start, 2), "end": round(e.end, 2), "text": e.text} for e in transcript.entries],
    }
    ensure_essential_topics(mindmap, transcript.entries)
    progress("Map ready", 0.85)
    return mindmap


# ---------------------------------------------------------------------------
# Sentence index: ONE encoder + ONE encode pass for every transcript sentence
# ---------------------------------------------------------------------------
class SentenceIndex:
    def __init__(self, sents: list[TranscriptEntry]):
        self.sents = sents
        self.texts = [s.text for s in sents]
        self.encoder = Embedder(self.texts or ["empty"])
        self.vecs = self.encoder.encode(self.texts) if sents else np.zeros((0, 1), dtype=np.float32)
        self.starts = np.array([s.start for s in sents]) if sents else np.zeros(0)

    def span(self, start: float, end: float) -> np.ndarray:
        """Indices of sentences starting inside [start, end] (half-second tolerance)."""
        return np.nonzero((self.starts >= start - 0.5) & (self.starts < end + 0.5))[0]


class _Grounder:
    """Picks real transcript sentences for nodes. Queries are encoded in one batch per section."""

    def __init__(self, sents: list[TranscriptEntry], vecs: np.ndarray, encoder: Embedder):
        self.sents, self.vecs, self.encoder = sents, vecs, encoder
        self.texts = [s.text for s in sents]

    def sims(self, queries: list[str]) -> np.ndarray:
        if not self.sents or not queries:
            return np.zeros((len(queries), len(self.sents)))
        return cosine_matrix(self.encoder.encode(queries), self.vecs)

    def locate(self, sims: np.ndarray, hint: float | None, seg: Segment, taken: set[int] = frozenset()) -> int:
        """Most similar sentence not already used by a sibling, pulled towards `hint` seconds when given."""
        score = np.array(sims, dtype=float)
        if isinstance(hint, (int, float)) and seg.start - 5 <= hint <= seg.end + 5:
            dist = np.array([abs(s.start - hint) for s in self.sents])
            score = score - dist / max(seg.end - seg.start, 1.0) * 0.5
        if len(taken) < len(score):
            score[list(taken)] = -np.inf
        return int(np.argmax(score))

    def best(self, sims: np.ndarray, k: int, exclude: set[int]) -> list[int]:
        order = [int(i) for i in np.argsort(-sims) if int(i) not in exclude and len(self.texts[int(i)]) > 25]
        return order[:k]


def _language_hint(request: dict, transcript: Transcript) -> str | None:
    """Caption-track language from YouTube, when the transcript came with one."""
    if transcript.source == "extension":
        return (request.get("languages") or [None])[0]
    if transcript.source == "youtube-captions":
        return transcript.language
    return None  # speech-to-text: let the text decide


def _assemble_section(seg: Segment, concepts, index: SentenceIndex, cfg: ModeConfig, clock: dict, used_terms: set[str]) -> dict:
    idx = index.span(seg.start, seg.end)
    if len(idx):
        grounding = _Grounder([index.sents[i] for i in idx], index.vecs[idx], index.encoder)
    else:  # tiny segments without sentence starts: fall back to its blocks
        grounding = _Grounder(seg.blocks, index.encoder.encode([b.text for b in seg.blocks]), index.encoder)
    sents = grounding.sents

    if seg.from_chapter:
        title = seg.title
    else:  # name the section after the concepts that are most specific to it
        specific = sorted(concepts, key=lambda c: (len(c.segments), -c.score))
        picks = [c.label for c in specific if " " in c.label][:1]
        taken = {w.lower().removesuffix("'s") for p in picks for w in p.split()}
        picks += [c.label for c in specific if " " not in c.label and c.label.lower().removesuffix("'s") not in taken][:1]
        title = " & ".join(picks) if picks else seg.title

    # summary retriever: the 1–3 most representative sentences of the section (no LLM)
    tick = time.perf_counter()
    ranked = summary_points(grounding, SUMMARY_POINTS)
    clock["summary"] += (time.perf_counter() - tick) * 1000
    central = ranked[0].text if ranked else ""
    gist = central if len(ranked) < 2 or len(central) > 140 else f"{central} {ranked[1].text}"
    section = make_node(
        truncate(str(title), 60),
        "section",
        1,
        seg.start,
        seg.end,
        summary=truncate(gist, 240),
        source=truncate(central, 300),
        tone=merge_tone(detect_tone(seg.text), None),
        keywords=seg.keywords,
        segmentId=seg.id,
        theme=seg.theme,
        points=[{"start": round(p.start, 2), "end": round(p.end, 2), "text": truncate(p.text, 200)} for p in sorted(ranked, key=lambda p: p.start)],
    )
    section.update({
        "explain": truncate(gist, 240),
        "generatedBy": "extractive",
        "groundingNote": "Extractive section summary from the transcript.",
    })
    section["_ranked"] = ranked  # read (and removed) by the Quick recall branch
    if seg.from_chapter:
        section["chapter"] = True  # creator's title: never relabelled

    raw_concepts = _heuristic_concepts(concepts, sents, cfg, used_terms)
    sims = grounding.sims([rc["term"] + " " + rc["source"] for rc in raw_concepts])
    used_leaves: set[int] = set()
    anchors: set[int] = set()
    for i, rc in enumerate(raw_concepts):
        at = grounding.locate(sims[i], rc["start"], seg, anchors)
        anchors.add(at)
        node = make_node(
            truncate(rc["label"], 48), "concept", 2, rc["start"], rc["end"],
            summary="",
            source=truncate(rc["source"], 300),
            topic=rc["term"],
            evidence=[{"start": round(st, 2), "end": round(en, 2), "text": truncate(tx, 300)} for st, en, tx in rc.get("evidence", [])[:4]],
            **rc.get("learning", {}),
        )
        for tag, child in rc["children"][: cfg.details]:
            text = compress(child.text, cfg.detail_words)
            extra = {"tag": tag} if tag else {}  # Detailed maps also list untagged supporting lines
            # Short / Standard: each item can grow lazily ("+"); Detailed: already complete, nothing to add
            node["children"].append(make_node(f"{tag}: {text}" if tag else text, "detail", 3, child.start, child.end, source=truncate(child.text, 300), more=not cfg.expanded, **extra))
        for j in grounding.best(sims[i], k=cfg.leaves, exclude=used_leaves):
            used_leaves.add(j)
            leaf = sents[j]
            node["children"].append(make_node(f"“{truncate(leaf.text, 150)}”", "transcript", 4, leaf.start, leaf.end, source=truncate(leaf.text, 300)))
        # The concept span must cover every detail it points to. Otherwise a detail's
        # timestamp looks mismatched against its parent (the defect in the old export).
        # Widen only to detail sentences that are themselves transcript evidence for this concept.
        starts = [node["start"]] + [c["start"] for c in node["children"] if c.get("start") is not None]
        ends = [node["end"]] + [c["end"] for c in node["children"] if c.get("end") is not None]
        node["start"] = round(min(starts), 2)
        node["end"] = round(max(ends), 2)
        node["collapsed"] = not cfg.expanded and any(c["type"] == "detail" for c in node["children"])
        section["children"].append(node)

    section["children"].sort(key=lambda n: n["start"] if n["start"] is not None else 1e9)
    return section


def summary_points(grounding: _Grounder, k: int) -> list[TranscriptEntry]:
    """
    Extractive summary retriever: the k sentences closest to the section's centroid (the
    LSA/BERT vectors are already computed), diversified with MMR. Ranked best first.
    """
    if not grounding.sents:
        return []
    keep = [i for i, t in enumerate(grounding.texts) if 6 <= len(t.split()) <= 45] or list(range(len(grounding.texts)))
    vecs = grounding.vecs[keep]
    rel = cosine_matrix(vecs.mean(axis=0, keepdims=True), vecs)[0].astype(float)
    chosen: list[int] = []
    seen: set[str] = set()
    while len(chosen) < k:
        score = rel.copy() if not chosen else 0.7 * rel - 0.3 * cosine_matrix(vecs, vecs[chosen]).max(axis=1)
        score[chosen] = -np.inf
        best = int(np.argmax(score))
        if not np.isfinite(score[best]):
            break
        rel[best] = -np.inf  # never picked twice
        norm = _norm(grounding.texts[keep[best]])
        if norm in seen:  # a caption line repeated later in the video
            continue
        seen.add(norm)
        chosen.append(best)
    return [grounding.sents[keep[i]] for i in chosen]


# ---------------------------------------------------------------------------
# Heuristic (English) labels: "Term: meaning" concepts and tagged level-3 items.
# The LLM rewrites them later; when it cannot, these are what the student reads.
# ---------------------------------------------------------------------------
COPULA_RE = re.compile(r"^(?:is|are|was|were|means|mean|refers to|refer to|is called|are called|is defined as|are defined as|stands for|describes)\s+", re.I)
LEAD_RE = re.compile(r"^(?:so|and|but|now|okay|ok|well|basically|actually|right|also|then|because)\b[,\s]+", re.I)
ARTICLE_RE = re.compile(r"^(?:a|an|the)\s+", re.I)
CLAUSE_RE = re.compile(r",|;|\s(?:which|because|so that|whereas|while|although)\s|\s[-–—]\s")
TAG_RULES = [
    ("Formula", re.compile(r"[=≈∝√∑]|\b(?:equals?|formula|equation|squared|cubed|divided by|multiplied by|proportional to|per cent|percent)\b|\d\s*[x×*/^+−-]\s*\d", re.I)),
    ("Eg", re.compile(r"\b(?:for example|for instance|e\.g\.|such as|imagine|let's say|say you|consider|like when|an example)\b", re.I)),
    ("Watch-out", re.compile(r"\b(?:careful|caution|mistakes?|wrong|avoid|never|don't|do not|cannot|can't|problems?|pitfalls?|unless|fragile|danger\w*|risk\w*|warning|tricky|confus\w+|limitations?|however|destroys?)\b", re.I)),
    ("Tip", re.compile(r"\b(?:remember|tip|trick|key|important|note that|always|make sure|rule of thumb|best way|should)\b", re.I)),
    ("Def", re.compile(r"\b(?:is a|is an|is the|are the|means|refers to|defined as|is called|are called|known as|stands for)\b", re.I)),
]


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def compress(text: str, max_words: int) -> str:
    """Short revision line from a spoken sentence: no discourse markers, cut at a clause when possible."""
    text = re.sub(r"\s+", " ", text or "").strip().strip("“”\"")
    while (m := LEAD_RE.match(text)) and m.end() < len(text):
        text = text[m.end():]
    text = text.rstrip(" .;,")
    if len(text.split()) > max_words:
        heads = [text[: m.start()].strip().rstrip(",;:") for m in CLAUSE_RE.finditer(text)]
        fits = [h for h in heads if 4 <= len(h.split()) <= max_words]
        if fits:
            text = fits[-1]
        else:
            text = " ".join(text.split()[:max_words]).rstrip(",;:") + "…"
    return text[:1].upper() + text[1:]


def classify(text: str) -> str | None:
    for tag, rule in TAG_RULES:
        if rule.search(text):
            return tag
    return None


def _lower_first(text: str) -> str:
    return text if text[:2].isupper() else text[:1].lower() + text[1:]


def _meaning(aliases: list[str], mentions: list[TranscriptEntry]) -> tuple[str, TranscriptEntry | None]:
    """Extract a grounded short definition from the clause that follows the term."""
    bad_starts = {"often", "really", "typically", "messages", "message", "you", "we", "they", "that's", "so"}
    for s in mentions:
        if TRANSCRIPT_MARKERS.search(s.text):
            continue
        for alias in aliases:
            pos = s.text.lower().find(alias)
            if pos < 0:
                continue
            clause = s.text[pos + len(alias):].lstrip(" ,")
            m = CLAUSE_RE.search(clause)
            if m and m.start() > 2:
                clause = clause[: m.start()]
            if COPULA_RE.match(clause):
                clause = ARTICLE_RE.sub("", COPULA_RE.sub("", clause, count=1), count=1)
            else:
                clause = re.sub(r"^(?:which|that|where)\s+", "", clause, flags=re.I)
            first = clause.split()[0].lower().strip(",.") if clause.split() else ""
            if first in bad_starts or first in FILLER_WORDS:
                continue
            meaning = compress(clause, 12)
            if len(meaning.split()) >= 3 and not TRANSCRIPT_MARKERS.search(meaning):
                return _lower_first(meaning), s
    return "", None


def _learning_unavailable(term: str) -> dict:
    """Honest fallback: no instructional content without verified evidence or AI generation."""
    return {
        "explain": "Explanation unavailable: no verified transcript evidence or AI study aid.",
        "mechanism": "",
        "example": "",
        "exercise": "",
        "diagram": "",
        "available": False,
        "generatedBy": "unavailable",
        "groundingNote": "Teaching content is not generated without evidence. See the timestamped transcript source.",
    }



def _heuristic_concepts(concepts, sents: list[TranscriptEntry], cfg: ModeConfig, used_terms: set[str]) -> list[dict]:
    """Select concepts that have real transcript evidence; reject weak/transcript-like candidates."""
    fresh = [c for c in concepts if c.label.lower() not in used_terms]
    picks = (fresh + [c for c in concepts if c.label.lower() in used_terms])[: cfg.concepts * 3]
    out: list[dict] = []
    for c in picks:
        core = re.sub(r"[^a-z0-9]+", " ", c.label.lower()).strip()
        if core in VAGUE_LABELS or len(core.split()) > 4:
            continue
        aliases = [a for a in c.aliases[:5] if a and len(a) >= 3 and a.lower() not in VAGUE_LABELS]
        if not aliases:
            continue
        mentions, seen = [], set()
        for sent in sents:
            if any(a in sent.text.lower() for a in aliases) and _norm(sent.text) not in seen:
                seen.add(_norm(sent.text))
                mentions.append(sent)
        if not mentions:
            continue  # reject: no evidence in this section
        def uscore(m: TranscriptEntry) -> float:
            definitional = 1.0 if COPULA_RE.search(m.text) or any(
                COPULA_RE.search(m.text[m.text.lower().find(a) + len(a):]) for a in aliases if a in m.text.lower()) else 0.0
            length_ok = 1.0 if 35 <= len(m.text) <= 260 else 0.0
            clean = 0.0 if TRANSCRIPT_MARKERS.search(m.text) else 1.0
            near = 1.0 / (1.0 + abs(m.start - c.first_ts) / 30.0)
            return definitional * 2 + length_ok + clean + near
        anchor = max(mentions, key=uscore)
        if len(anchor.text.strip()) < 18 or TRANSCRIPT_MARKERS.search(anchor.text):
            continue  # reject weak evidence
        # The anchor is the single best supporting passage. Later nearby mentions are extra evidence;
        # earlier ones are not evidence for this concept and must not pull its timestamp backwards.
        others = sorted((m for m in mentions if m is not anchor and m.start > anchor.start and m.start <= anchor.start + 60),
                        key=lambda m: m.start)
        evidence = [(anchor.start, anchor.end, anchor.text)] + [(m.start, m.end, m.text) for m in others]
        near = [m for m in mentions if m is not anchor and anchor.start - 1 <= m.start <= anchor.start + 60
                and not TRANSCRIPT_MARKERS.search(m.text)]
        tagged = [(tag, m) for m in near if (tag := classify(m.text))]
        first_per_tag: dict[str, tuple[str, TranscriptEntry]] = {}
        for item in tagged:
            first_per_tag.setdefault(item[0], item)
        ordered = list(first_per_tag.values()) + [item for item in tagged if item not in first_per_tag.values()]
        if not ordered and near:
            ordered = [("Def", near[0])]
        used_terms.add(core)
        out.append({
            "label": smart_title(c.label)[:48],
            "term": c.label,
            "start": evidence[0][0] if evidence else anchor.start,
            "end": evidence[0][1] if evidence else (anchor.end if anchor.end > anchor.start else anchor.start + 1.0),
            "source": evidence[0][2] if evidence else anchor.text,
            "evidence": evidence,
            "children": sorted(ordered[: cfg.details], key=lambda item: TAGS.index(item[0]) if item[0] else len(TAGS)),
            "learning": _learning_unavailable(c.label),
        })
        if len(out) >= cfg.concepts:
            break
    return out


def _recall_branch(sections: list[dict], duration: float) -> dict | None:
    """'Quick recall': a one-line summary + 3–6 must-remember points (the sections' best summary sentences)."""
    ranked = [sec.pop("_ranked", []) for sec in sections]
    want = min(6, max(3, len(sections)))
    picks: list[TranscriptEntry] = []
    seen: set[str] = set()
    for depth in range(SUMMARY_POINTS):
        for points in ranked:
            if len(picks) < want and depth < len(points) and _norm(points[depth].text) not in seen:
                seen.add(_norm(points[depth].text))
                picks.append(points[depth])
    if not picks:
        return None
    picks.sort(key=lambda p: p.start)
    one_line = _one_line(sections)
    first = truncate(picks[0].text, 300)
    node = make_node("Quick recall", "section", 1, 0, duration, summary=one_line, source=first, recall=True, color=len(sections) % 5)
    node["children"].append(make_node(f"In one line: {one_line}", "concept", 2, 0, duration, source=first, recall=True, oneline=True))
    for p in picks:
        node["children"].append(make_node(compress(p.text, 14), "concept", 2, p.start, p.end, source=truncate(p.text, 300), recall=True))
    return node


def _one_line(sections: list[dict]) -> str:
    names = [" ".join(w if w.isupper() else w.lower() for w in s["text"].split()) for s in sections[:5]]
    return compress("Covers " + ", ".join(names), 20) if names else ""


def _heuristic_overview(sections: list[dict]) -> str:
    return " · ".join(s["text"] for s in sections[:6])


def _short_title(title: str) -> str:
    # "How Quantum Computers Work | Full Lecture (2024)" -> "How Quantum Computers Work"
    core = re.split(r"\s[|\-–—:]\s|\(|\[", title)[0].strip()
    return smart_title(truncate(core or title, 48))


def _walk(node: dict):
    yield node
    for child in node.get("children", []):
        yield from _walk(child)


# ---------------------------------------------------------------------------
# Faithfulness: how well a node's text matches its own transcript span
# ---------------------------------------------------------------------------
FAITH_TYPES = {"section", "concept", "detail"}


def _latin(text: str) -> bool:
    letters = [ch for ch in text if ch.isalpha()]
    return bool(letters) and sum(ch.isascii() for ch in letters) / len(letters) > 0.6


def score_faithfulness(nodes: list[dict], encoder: Embedder, spans: dict[str, str] | None = None) -> None:
    """
    Set `faith` (0..1 cosine) = similarity between a node's text and its own transcript span
    (`spans[id]` when given, else the node's `source` sentence). Skipped for non-Latin
    transcripts, where an English label cannot be compared with the spoken words.
    """
    spans = spans or {}
    items = [(n, spans.get(n["id"]) or n.get("source") or "") for n in nodes if n.get("type") in FAITH_TYPES]
    items = [(n, span) for n, span in items if span and _latin(span)]
    if not items:
        return
    texts = encoder.encode([n.get("_faith_text") or n["text"] for n, _ in items])
    vecs = encoder.encode([span for _, span in items])
    for (node, _), a, b in zip(items, texts, vecs):
        node["faith"] = round(max(0.0, float(a @ b)), 2)


# ---------------------------------------------------------------------------
# Cross links: knowledge-graph edges + embedding similarity across sections
# ---------------------------------------------------------------------------
MAX_EDGES = 10
MAX_SIMILAR_EDGES = 4
SIMILAR_THRESHOLD = 0.55


def _cross_edges(root: dict, graph: ConceptGraph, index: SentenceIndex) -> list[dict]:
    concept_nodes = [(sec, c) for sec in root["children"] if not sec.get("recall") for c in sec["children"] if c["type"] == "concept"]
    if not concept_nodes:
        return []
    edges: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def add(a: dict, b: dict, label: str, source: str):
        key = tuple(sorted((a["id"], b["id"])))
        if a["id"] == b["id"] or key in seen:
            return
        seen.add(key)
        edges.append({"id": new_id("e"), "source": a["id"], "target": b["id"], "label": truncate(label, 32), "origin": source})

    # knowledge-graph edges mapped onto nodes whose text mentions the concept
    by_concept = {c.id: c for c in graph.concepts}

    def node_for(concept_id: str):
        concept = by_concept.get(concept_id)
        if not concept:
            return None
        for sec, node in concept_nodes:
            low = node["text"].lower()
            if any(alias in low for alias in concept.aliases[:4]):
                return sec, node
        return None

    for e in graph.edges:
        if len(edges) >= MAX_EDGES - MAX_SIMILAR_EDGES:
            break
        a, b = node_for(e["source"]), node_for(e["target"])
        if a and b and a[0]["id"] != b[0]["id"]:
            add(a[1], b[1], e["label"], "graph")

    # semantically close concepts that live in different sections
    vecs = index.encoder.encode([f"{c['text']}. {c['summary']}" for _, c in concept_nodes])
    sims = cosine_matrix(vecs)
    pairs = [
        (float(sims[i, j]), i, j)
        for i in range(len(concept_nodes))
        for j in range(i + 1, len(concept_nodes))
        if concept_nodes[i][0]["id"] != concept_nodes[j][0]["id"] and sims[i, j] >= SIMILAR_THRESHOLD
    ]
    added = 0
    for _, i, j in sorted(pairs, reverse=True):
        if len(edges) >= MAX_EDGES or added >= MAX_SIMILAR_EDGES:
            break
        before = len(edges)
        add(concept_nodes[i][1], concept_nodes[j][1], "related to", "similar")
        added += len(edges) - before
    return edges
