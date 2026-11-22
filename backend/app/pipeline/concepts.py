"""
Layer 3 — Key Concept Extraction.

Goal: real concepts, not raw captions.

1. Candidate phrases
   * spaCy noun chunks + named entities when `en_core_web_sm` is installed,
   * otherwise 1–3-gram TF-IDF features filtered by stop-word rules.
2. TF-IDF weighting across segments (a concept that is specific to one
   segment scores higher than one said everywhere).
3. Semantic grouping: near-duplicate phrases ("neural net", "neural networks")
   are merged by agglomerative clustering on (BERT or LSA) embeddings.
4. Knowledge graph (networkx): nodes = concepts, edges = co-occurrence inside
   the same transcript block + semantic similarity. PageRank gives global
   importance, greedy modularity gives concept communities, and simple lexical
   patterns ("X is a Y", "X uses Y") give edges a relation label.

Output: `ConceptGraph` with concept nodes and labelled, weighted edges.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations

import networkx as nx
import numpy as np

from ..config import has_module
from .embeddings import cosine_matrix, embed
from .segmentation import Segment, stop_words
from .text_utils import FILLER_WORDS, candidate_phrases, smart_title, split_sentences
from .terminology import is_meaningful_label, normalize_label

log = logging.getLogger("mivmap.concepts")

# Labels that look like concepts but are really transcript scaffolding or filler.
VAGUE_TERMS = {
    "revisit", "foundation", "foundations", "additional", "information", "message", "messages",
    "overview", "topic", "topics", "part", "parts", "thing", "things", "stuff", "question",
    "answer", "example", "examples", "number", "numbers", "first", "second", "third", "last",
    "web", "server", "request", "response", "http", "url", "uri", "cookie", "cookies",
}
VAGUE_LABELS = VAGUE_TERMS

# Filler-style "concepts" observed in generated maps: generic states or process
# nouns that name nothing a learner can study.
FILLER_CONCEPT_LABELS = {
    "hard work", "hard working", "hardworking", "working hard",
    "motivated", "motivating", "motivation", "motivates", "motivate",
    "length", "lengths",
    "researching", "research", "researches",
    "effort", "efforts",
    "trying", "trying to",
    "crazy", "powerful", "reassuring", "complicated", "necessary", "distinct",
    "recognizing", "dig", "various", "general", "specific",
}
# Words/phrases observed as junk concepts in real exports (e.g. Networking for Hackers #2).
# "Hope"/"Ended"/"Extra" are outro or discourse words; "Quite Technical" is an evaluative
# aside. None of them names something a learner could study.
NON_CONCEPT_LABELS = VAGUE_TERMS | FILLER_CONCEPT_LABELS | {
    "hope", "hoped", "hoping", "ended", "ending", "ends", "extra", "extras", "quite technical",
    "technical", "quick", "quickly", "really", "pretty", "fairly", "somewhat", "totally",
    "advanced", "basic", "basics", "simple", "complex", "important", "main", "major",
    "other", "others", "something", "anything", "everything", "everyone", "guys", "friends",
    "subscribe", "subscribing", "like", "share", "comment", "comments", "description", "link",
    "links", "welcome", "thanks", "thank you", "everybody", "moment", "moments", "bit",
    "way", "ways", "lot", "lots", "kind", "sort", "point", "points", "idea", "ideas",
    "difference", "differences", "similarity", "similarities", "start", "beginning", "end",
    "research", "box", "infrastructure inside", "fiber equipment", "fibre equipment", "funding",
    "wi fi signal", "signal", "signals", "networking playlist", "phone connects", "square feet",
    "yesterday", "notes", "print", "money", "equipment", "layer", "model", "provider", "providers",
    "previous videos compared", "upcoming videos", "entire content", "extra infrastructure inside",
    "outside continents", "hardware level", "nearby", "fails", "gestures", "check", "storage",
    "offices", "subscriber", "data", "support", "area network", "fiber connection",
    "fiber optic connections", "wi-fi connection", "tier", "department of defense",
    "advanced research projects", "website request", "upcoming", "distant",
    "mb storage cost", "multiple cities", "research projects", "wired connections international",
    "experience", "curious", "videos compared", "entire networking module",
}
GENERIC_WORDS = {
    "previous", "upcoming", "videos", "video", "module", "project", "projects", "labs", "lab",
    "storage", "cost", "thousands", "energy", "distant", "fails", "curious", "experience",
    "entire", "international", "wired", "compared", "multiple", "cities", "research",
    "connections", "connection", "goodbye", "support", "data", "offices", "subscriber",
    "nearby", "gestures", "check", "piece", "equipment", "extra", "networking",
}
CANONICAL_LABELS = {
    "lan": "LAN", "pan": "PAN", "man": "MAN", "wan": "WAN", "isp": "ISP", "isps": "ISPs",
    "osi": "OSI", "cdn": "CDN", "cdns": "CDNs", "tcp": "TCP", "ip": "IP", "tcp ip": "TCP/IP",
    "arpanet": "ARPANET", "ont": "ONT", "http": "HTTP", "https": "HTTPS", "url": "URL", "uri": "URI",
}


def canonical_label(label: str) -> str:
    """Keep common acronyms in their conventional upper case on the canvas."""
    return CANONICAL_LABELS.get(re.sub(r"[^a-z0-9 ]", " ", label.lower()).strip(), label)
EVALUATIVE_PREFIX = re.compile(
    r"^(?:quite|very|really|pretty|fairly|somewhat|too|so|kind of|sort of|kind|sort|"
    r"basically|actually|literally|simply|just|especially|totally)\b", re.I)
PROMO_MARKERS = re.compile(
    r"\b(?:subscribe|like and subscribe|thanks for watching|thank you for watching|"
    r"see you in the next|hope you (?:enjoyed|liked|are)|check (?:out|the) description|"
    r"link in the description|comment below|hit the bell|next video|your momentum)\b", re.I)

TRANSCRIPT_MARKERS = re.compile(
    r"\b(?:revisit|number of reasons|first being|let's|we're going to|we have|here we|"
    r"take a look|as you can see|going to talk|today we|welcome back|so here|now let's)\b", re.I)


@dataclass
class Concept:
    id: str
    label: str
    score: float
    aliases: list[str] = field(default_factory=list)
    segments: list[str] = field(default_factory=list)
    first_ts: float = 0.0
    mentions: list[float] = field(default_factory=list)
    community: int = 0
    centrality: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "score": round(self.score, 4),
            "aliases": self.aliases,
            "segments": self.segments,
            "firstTs": self.first_ts,
            "community": self.community,
            "centrality": round(self.centrality, 4),
        }


@dataclass
class ConceptGraph:
    concepts: list[Concept]
    edges: list[dict]

    def for_segment(self, seg_id: str, limit: int = 8) -> list[Concept]:
        items = [c for c in self.concepts if seg_id in c.segments]
        return sorted(items, key=lambda c: -(c.score + c.centrality))[:limit]

    def to_dict(self) -> dict:
        return {"concepts": [c.to_dict() for c in self.concepts], "edges": self.edges}


# "<a> ... <b>": the text BETWEEN two concept mentions decides the relation.
# Compiled once — building per-pair regexes used to dominate pipeline time.
RELATION_PATTERNS = [
    (re.compile(r"\s+(?:is|are)\s+(?:a|an|the)?\s*(?:type|kind|form|example)\s+of\s+"), "is a type of"),
    (re.compile(r"\s+(?:is|are)\s+(?:a|an)\s+"), "is a"),
    (re.compile(r"\s+(?:uses?|relies on|depends on|requires?)\s+(?:the\s+)?"), "uses"),
    (re.compile(r"\s+(?:causes?|leads? to|results? in|produces?)\s+(?:the\s+)?"), "leads to"),
    (re.compile(r"\s+(?:contains?|includes?|consists? of|has|have)\s+(?:the\s+)?"), "contains"),
    (re.compile(r"\s+(?:vs\.?|versus|compared to|unlike)\s+(?:the\s+)?"), "contrasts with"),
    (re.compile(r"\s+(?:improves?|increases?|boosts?|enables?)\s+(?:the\s+)?"), "enables"),
    (re.compile(r"\s+(?:reduces?|decreases?|prevents?|limits?)\s+(?:the\s+)?"), "reduces"),
]


@lru_cache(maxsize=1)
def _spacy():
    if not has_module("spacy"):
        return None
    try:
        import spacy

        return spacy.load("en_core_web_sm", disable=["lemmatizer"])
    except Exception:  # noqa: BLE001 - model not downloaded
        return None


def _norm_label(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _valid_phrase(phrase: str, stops: set[str]) -> bool:
    words = phrase.split()
    if not words or len(words) > 4:
        return False
    if words[0] in stops or words[-1] in stops:
        return False
    if all(w in FILLER_WORDS or w in stops for w in words):
        return False
    if len(phrase) < 3 or phrase.isdigit():
        return False
    if TRANSCRIPT_MARKERS.search(phrase):
        return False
    core = re.sub(r"[^a-z0-9 ]", "", phrase.lower()).strip()
    if core in VAGUE_TERMS or core in NON_CONCEPT_LABELS:
        return False
    if EVALUATIVE_PREFIX.match(phrase.strip()) or PROMO_MARKERS.search(phrase):
        return False
    if any(w in GENERIC_WORDS for w in core.split()):
        from .topics import ESSENTIAL_LABELS  # local import avoids a module cycle
        if core not in {_norm_label(e) for e in ESSENTIAL_LABELS}:
            return False  # generic / meta wording, whatever combination it appears in
    if len(words) == 1 and core in VAGUE_TERMS:
        return False
    return len(set(words)) == len(words)  # "data data" style caption stutter


def extract_concepts(segments: list[Segment], max_concepts: int = 60) -> ConceptGraph:
    if not segments:
        return ConceptGraph([], [])
    from sklearn.feature_extraction.text import TfidfVectorizer

    stops = set(stop_words())
    docs = [s.text for s in segments]

    # ---- 1+2. candidates weighted by TF-IDF --------------------------------
    vectorizer = TfidfVectorizer(
        # clause-bounded noun-phrase-like n-grams (inner "of" allowed: "theory of mind")
        analyzer=lambda doc: candidate_phrases(doc, stops),
        sublinear_tf=True,
        min_df=1,
        max_df=0.95 if len(docs) > 3 else 1.0,
    )
    try:
        tfidf = vectorizer.fit_transform(docs)
    except ValueError:
        return ConceptGraph([], [])
    vocab = vectorizer.get_feature_names_out()

    allowed: set[str] | None = None
    nlp = _spacy()
    if nlp is not None:
        allowed = set()
        for doc in nlp.pipe((d[:100_000] for d in docs), batch_size=4):
            for chunk in doc.noun_chunks:
                toks = [t.text.lower() for t in chunk if not t.is_stop and t.is_alpha]
                if toks:
                    allowed.add(" ".join(toks))
            for ent in doc.ents:
                if ent.label_ not in {"CARDINAL", "ORDINAL", "DATE", "TIME", "PERCENT", "QUANTITY", "MONEY"}:
                    allowed.add(ent.text.lower())

    col_scores = np.asarray(tfidf.max(axis=0).todense()).ravel()
    per_segment_hits = (tfidf > 0).toarray()
    candidates: list[tuple[str, float, list[int]]] = []
    for j in np.argsort(-col_scores):
        phrase = vocab[j]
        if not _valid_phrase(phrase, stops):
            continue
        if allowed is not None and phrase not in allowed and not any(phrase in a for a in allowed):
            continue
        # multi-word phrases are more "concept-like" than single words
        boost = 1.0 + 0.2 * (len(phrase.split()) - 1)
        segs = list(np.nonzero(per_segment_hits[:, j])[0])
        candidates.append((phrase, float(col_scores[j]) * boost, segs))
        if len(candidates) >= max_concepts * 3:
            break
    if not candidates:
        return ConceptGraph([], [])

    # ---- 3. semantic grouping of near-duplicates ---------------------------
    phrases = [c[0] for c in candidates]
    vecs = embed(phrases, corpus=docs)
    groups = _group_similar(phrases, vecs)

    concepts: list[Concept] = []
    for members in groups:
        members.sort(key=lambda i: -candidates[i][1])
        head = candidates[members[0]]
        seg_idx = sorted({s for i in members for s in candidates[i][2]})
        concepts.append(
            Concept(
                id=f"c{len(concepts) + 1}",
                label=canonical_label(smart_title(head[0])),
                score=sum(candidates[i][1] for i in members),
                aliases=[candidates[i][0] for i in members],
                segments=[segments[s].id for s in seg_idx],
            )
        )
    # Final normalization + gate. This is where caption fragments become real
    # terminology ("Local Area" -> LAN, "Sign of Internet" -> Internet), where
    # section headings and generic words are rejected, and where two fragments
    # that normalize to the same topic are merged instead of duplicated.
    from .topics import ESSENTIAL_LABELS
    essentials = {_norm_label(e) for e in ESSENTIAL_LABELS}
    seg_text = {s.id: s.text for s in segments}
    normalized: dict[str, Concept] = {}
    kept: list[Concept] = []
    for c in concepts:
        core = _norm_label(c.label)
        context = " ".join(seg_text.get(sid, "") for sid in c.segments[:6])
        label = normalize_label(c.label, context)
        essential = core in essentials
        if essential and not label:
            label = c.label  # a syllabus topic always survives normalization
        if not label or not is_meaningful_label(label):
            continue
        if not essential and (core in NON_CONCEPT_LABELS
                              or (set(_norm_label(label).split()) & GENERIC_WORDS)
                              or EVALUATIVE_PREFIX.match(label)
                              or PROMO_MARKERS.search(label)):
            continue
        c.label = label
        c.aliases = list(dict.fromkeys(c.aliases + [label]))
        merge_key = _norm_label(label)
        if merge_key in normalized:
            other = normalized[merge_key]
            other.aliases = list(dict.fromkeys(other.aliases + c.aliases))
            other.segments = list(dict.fromkeys(other.segments + c.segments))
            other.score = max(other.score, c.score)
            continue
        normalized[merge_key] = c
        kept.append(c)
    concepts = kept
    concepts.sort(key=lambda c: -c.score)
    concepts = concepts[:max_concepts]
    for i, c in enumerate(concepts):
        c.id = f"c{i + 1}"

    blocks = [block for seg in segments for block in seg.blocks]
    presence = _mention_matrix(concepts, blocks)
    _locate_mentions(concepts, blocks, presence)
    edges = _build_graph(concepts, segments, blocks, presence)
    return ConceptGraph(concepts, edges)


def _group_similar(phrases: list[str], vecs: np.ndarray, threshold: float = 0.82) -> list[list[int]]:
    """Union-find over pairs that are semantically or lexically near-identical."""
    parent = list(range(len(phrases)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    sims = cosine_matrix(vecs) if len(phrases) > 1 else np.ones((1, 1))
    stems = [re.sub(r"(ies|es|s|ing|ed)\b", "", p) for p in phrases]
    for i, j in combinations(range(len(phrases)), 2):
        same_stem = stems[i] == stems[j]
        contained = (phrases[i] in phrases[j] or phrases[j] in phrases[i]) and sims[i, j] > threshold - 0.1
        near_vague = (
            phrases[i].split()[-1] == phrases[j].split()[-1]
            and (phrases[i].split()[0] in VAGUE_TERMS or phrases[j].split()[0] in VAGUE_TERMS)
        )
        if same_stem or contained or near_vague or sims[i, j] >= threshold:
            parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(phrases)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _mention_matrix(concepts: list[Concept], blocks: list) -> np.ndarray:
    """presence[i, j] = concept i is mentioned in block j (one combined word-boundary regex per concept)."""
    presence = np.zeros((len(concepts), len(blocks)), dtype=bool)
    for i, c in enumerate(concepts):
        pattern = re.compile(r"\b(?:" + "|".join(re.escape(a) for a in c.aliases[:5]) + r")\b", re.I)
        presence[i] = [bool(pattern.search(b.text)) for b in blocks]
    return presence


def _locate_mentions(concepts: list[Concept], blocks: list, presence: np.ndarray) -> None:
    for i, c in enumerate(concepts):
        c.mentions = sorted(blocks[j].start for j in np.nonzero(presence[i])[0])
        c.first_ts = c.mentions[0] if c.mentions else 0.0


def _build_graph(concepts: list[Concept], segments: list[Segment], blocks: list, presence: np.ndarray) -> list[dict]:
    graph = nx.Graph()
    for c in concepts:
        graph.add_node(c.id)

    sentences: list[str] = []
    for j, block in enumerate(blocks):
        present = [concepts[i].id for i in np.nonzero(presence[:, j])[0]]
        for a, b in combinations(present, 2):
            w = graph.get_edge_data(a, b, {}).get("weight", 0.0)
            graph.add_edge(a, b, weight=w + 1.0)
        sentences.extend(split_sentences(block.text))

    # semantic similarity edges connect concepts that are never said together
    if len(concepts) > 1:
        vecs = embed([c.label for c in concepts], corpus=[s.text for s in segments])
        sims = cosine_matrix(vecs)
        for i, j in combinations(range(len(concepts)), 2):
            if 0.55 <= sims[i, j] < 0.82:
                a, b = concepts[i].id, concepts[j].id
                w = graph.get_edge_data(a, b, {}).get("weight", 0.0)
                graph.add_edge(a, b, weight=w + float(sims[i, j]))

    if graph.number_of_edges():
        pr = nx.pagerank(graph, weight="weight")
        communities = nx.algorithms.community.greedy_modularity_communities(graph, weight="weight")
        for idx, comm in enumerate(communities):
            for cid in comm:
                next(c for c in concepts if c.id == cid).community = idx
        for c in concepts:
            c.centrality = pr.get(c.id, 0.0)

    by_id = {c.id: c for c in concepts}
    lowered = [s.lower() for s in sentences]
    alias_res = {c.id: re.compile("|".join(re.escape(x) for x in c.aliases[:3])) for c in concepts}
    mentions: dict[str, set[int]] = {}

    def mentioned_in(cid: str) -> set[int]:
        if cid not in mentions:
            mentions[cid] = {i for i, s in enumerate(lowered) if alias_res[cid].search(s)}
        return mentions[cid]

    edges = []
    for a, b, data in sorted(graph.edges(data=True), key=lambda e: -e[2]["weight"])[: len(concepts) * 2]:
        shared = sorted(mentioned_in(a) & mentioned_in(b))
        label, reverse = _relation_label(alias_res[a], alias_res[b], [lowered[i] for i in shared])
        if reverse:
            a, b = b, a
        edges.append({"source": a, "target": b, "weight": round(data["weight"], 3), "label": label})
    return edges


def _relation_label(re_a: re.Pattern, re_b: re.Pattern, sentences: list[str]) -> tuple[str, bool]:
    """Return (label, reversed) — reversed means the relation reads "b <label> a".
    `sentences` are lower-cased sentences that mention both concepts."""
    for low in sentences:
        spans_a = [m.span() for m in re_a.finditer(low)]
        spans_b = [m.span() for m in re_b.finditer(low)]
        for pattern, label in RELATION_PATTERNS:
            for (sa, ea), (sb, eb) in ((x, y) for x in spans_a for y in spans_b):
                if ea <= sb and pattern.fullmatch(low[ea:sb]):
                    return label, False
                if eb <= sa and pattern.fullmatch(low[eb:sa]):
                    return label, True
    return "related to", False
