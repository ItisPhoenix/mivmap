"""Domain topic coverage helpers.

The generic TF-IDF extractor is good at nouns but misses syllabus-level topics that are
only implied by examples (e.g. "status code 200" -> Status codes, "class C" -> ISP tiers).
A small, transparent lexicon restores those topics when the transcript actually mentions
them. It never invents a topic: every entry requires a literal cue in the transcript and
records the timestamp of the first matching block.

Domains are detected from the transcript itself; HTTP is kept as the first lexicon for
backwards compatibility with earlier maps and tests.
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

# Networking / "history of the Internet" lectures (e.g. Networking for Hackers #2).
NETWORKING_TOPICS = [
    ("LAN", [r"\blan\b", r"local area network"]),
    ("PAN", [r"\bpan\b", r"personal area network"]),
    ("MAN", [r"\bman\b", r"metropolitan area network"]),
    ("WAN", [r"\bwan\b", r"wide area network"]),
    ("ARPANET", [r"arpanet", r"advanced research projects agency network"]),
    ("TCP/IP", [r"tcp/ip", r"tcp and ip", r"transmission control protocol", r"internet protocol suite"]),
    ("OSI model", [r"\bosi\b", r"open systems interconnection", r"seven.layer model"]),
    ("ISP tiers", [r"tier[\s-]?[123]", r"internet service provider", r"\bisps?\b"]),
    ("Fibre infrastructure", [r"\bfibre\b", r"\bfiber\b", r"optical (?:fibre|fiber)", r"undersea cable", r"submarine cable"]),
    ("CDNs", [r"\bcdns?\b", r"content delivery network", r"content distribution network"]),
]

# Domain anchors prevent generic cue words in one lexicon (e.g. "get", "body",
# "network") from pulling HTTP/Networking topics into unrelated lectures.
DOMAIN_ANCHORS = {
    "networking": re.compile(
        r"\b(?:lan|wan|man|pan|isp|isps|arpanet|osi|cdn|cdns|fibre|fiber|"
        r"internet service provider|content delivery network|wide area network|"
        r"local area network|metropolitan area network|personal area network)\b", re.I),
    "http": re.compile(
        r"\b(?:http|https|hypertext transfer protocol)\b", re.I),
}
TOPIC_LEXICONS: dict[str, list[tuple[str, list[str]]]] = {
    "networking": NETWORKING_TOPICS,
    "http": HTTP_TOPICS,
}

# Union kept for callers that historically checked every essential label.
ESSENTIAL_LABELS = {label for lexicon in TOPIC_LEXICONS.values() for label, _ in lexicon}


@dataclass(frozen=True)
class TopicHit:
    label: str
    aliases: tuple[str, ...]
    start: float


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _hits(blocks, topics: list[tuple[str, list[str]]]) -> list[TopicHit]:
    hits: list[TopicHit] = []
    for label, patterns in topics:
        rx = re.compile("|".join(f"(?:{p})" for p in patterns), re.I)
        matched = next((b for b in blocks
                    if rx.search(b.get("text", "") if isinstance(b, dict) else b.text)), None)
        if matched is not None:
            start = matched.get("start", 0) if isinstance(matched, dict) else matched.start
            hits.append(TopicHit(label, tuple(re.findall(r"[a-z][a-z0-9/2-]+", label.lower())), start))
    return hits


def detect_http_topics(blocks) -> list[TopicHit]:
    """Backwards-compatible helper used by older callers and tests."""
    return _hits(blocks, HTTP_TOPICS)


def detect_domain(blocks) -> tuple[str, list[TopicHit]]:
    """Return (domain, hits) only when a transcript has a real domain anchor."""
    text = " ".join(b.get("text", "") if isinstance(b, dict) else b.text for b in (blocks or []))
    ranked = sorted(((_hits(blocks, topics), domain) for domain, topics in TOPIC_LEXICONS.items()),
                    key=lambda pair: len(pair[0]), reverse=True)
    for hits, domain in ranked:
        if len(hits) >= 4 and DOMAIN_ANCHORS[domain].search(text):
            return domain, hits
    return "generic", []


def active_topics(mindmap: dict | None = None, blocks=None) -> list[tuple[str, list[str]]]:
    """Lexicon to use for coverage checks / supplementation."""
    if blocks is None and mindmap is not None:
        blocks = [{"text": str(e.get("text") or ""), "start": e.get("start", 0)}
                  for e in (mindmap.get("transcript") or [])[:400]]
    if not blocks:
        return []
    domain, _ = detect_domain(blocks)
    return TOPIC_LEXICONS.get(domain, [])


def supplement_concepts(concepts: list[Concept], segments, blocks, max_extra: int = 12) -> list[Concept]:
    """Add missing syllabus topics whose literal cues occur in the transcript."""
    existing = {_norm(c.label) for c in concepts}
    # substring-aware: "HTTP methods" is covered by "Methods", "URL structure" by "URLs"
    for c in concepts:
        for alias in c.aliases[:6]:
            existing.add(_norm(alias))
    extras: list[Concept] = []
    for hit in _hits(blocks, active_topics(blocks=blocks)):
        key = _norm(hit.label)
        segs = [s.id for s in segments if s.start - 1 <= hit.start <= s.end + 1]
        if not segs and segments:
            segs = [min(segments, key=lambda s: abs(s.start - hit.start)).id]
        # If the generic extractor found the topic, boost it so it is selected; otherwise add it.
        # Short acronyms must match as whole words: "man" must not match "command",
        # nor "pan" match "multinational".
        match = next((c for c in concepts if key == _norm(c.label)
                      or (len(key) >= 5 and key in _norm(c.label))
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
