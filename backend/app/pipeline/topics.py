"""Domain topic coverage helpers.

The generic TF-IDF extractor is good at nouns but misses syllabus-level topics that are
only implied by examples (e.g. "status code 200" -> Status codes).  A tiny, transparent
lexicon restores those topics when the transcript actually mentions them.  It never
invents a topic: every entry requires a literal cue in the transcript and records the
timestamp of the first matching block.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .concepts import Concept

# (canonical label, aliases -> any one cue is enough)
HTTP_TOPICS = [
    ("HTTP", [r"hypertext transfer protocol", r"\bhttp\b"]),
    ("Request and response messages", [r"request message", r"response message", r"request and response", r"requests and responses"]),
    ("HTTP methods", [r"\bget\b", r"\bpost\b", r"\bput\b", r"\bdelete\b", r"http methods?", r"main ones"]),
    ("Status codes", [r"status code", r"\b200\b", r"\b400s?\b", r"\b500s?\b", r"successful", r"server-side errors"]),
    ("Message headers", [r"message headers?", r"\bheaders\b"]),
    ("Message body", [r"\bbody\b", r"submitting data", r"update information"]),
    ("URL structure", [r"uniform resource locators?", r"\burl\b", r"scheme the domain the path", r"query string", r"fragments"]),
    ("URI", [r"uniform resource identifier", r"\buri\b"]),
    ("HTTPS and ports", [r"\bhttps\b", r"port 80", r"port 443", r"default for that scheme"]),
    ("HTTP/2 multiplexing", [r"multiplexing", r"http 2", r"http/2", r"single connection"]),
    ("Cookies", [r"\bcookies?\b", r"key value pairs", r"secure flag", r"same-site", r"cross-site requests"]),
    ("curl", [r"\bcurl\b", r"command line tool"]),
]


ESSENTIAL_LABELS = {label for label, _ in HTTP_TOPICS}


@dataclass(frozen=True)
class TopicHit:
    label: str
    aliases: tuple[str, ...]
    start: float


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def detect_http_topics(blocks) -> list[TopicHit]:
    hits: list[TopicHit] = []
    for label, patterns in HTTP_TOPICS:
        rx = re.compile("|".join(f"(?:{p})" for p in patterns), re.I)
        matched = next((b for b in blocks if rx.search(b.text)), None)
        if matched is not None:
            hits.append(TopicHit(label, tuple(re.findall(r"[a-z][a-z0-9/2-]+", label.lower())), matched.start))
    return hits


def supplement_concepts(concepts: list[Concept], segments, blocks, max_extra: int = 12) -> list[Concept]:
    """Add missing syllabus topics whose literal cues occur in the transcript."""
    existing = {_norm(c.label) for c in concepts}
    # substring-aware: "HTTP methods" is covered by "Methods", "URL structure" by "URLs"
    for c in concepts:
        for alias in c.aliases[:6]:
            existing.add(_norm(alias))
    extras: list[Concept] = []
    for hit in detect_http_topics(blocks):
        key = _norm(hit.label)
        segs = [s.id for s in segments if s.start - 1 <= hit.start <= s.end + 1]
        if not segs and segments:
            segs = [min(segments, key=lambda s: abs(s.start - hit.start)).id]
        # If the generic extractor found the topic, boost it so it is selected; otherwise add it.
        match = next((c for c in concepts if key == _norm(c.label) or key in _norm(c.label)
                      or any(key == _norm(a) for a in c.aliases[:6])), None)
        if match is not None:
            if key not in _norm(match.label):
                match.aliases.insert(0, match.label)
                match.label = hit.label
            match.score = max(match.score, 100.0 - len(extras))
            for sid in segs:
                if sid not in match.segments:
                    match.segments.append(sid)
            continue
        extras.append(
            Concept(
                id=f"topic{len(extras) + 1}",
                label=hit.label,
                score=100.0 - len(extras),
                aliases=list(hit.aliases),
                segments=segs,
                first_ts=hit.start,
            )
        )
    if not extras:
        return concepts
    merged = concepts + extras[:max_extra]
    merged.sort(key=lambda c: -c.score)
    for i, c in enumerate(merged):
        c.id = f"c{i + 1}"
    return merged
