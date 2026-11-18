"""Automated quality checks for generated Mivmap maps.

These checks are intentionally deterministic: they look for the concrete defects found in
the HTTP Fundamentals regression fixture (transcript-like labels, vague/duplicate concepts,
missing syllabus topics and detail nodes whose timestamps do not match their source).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

TRANSCRIPT_MARKERS = re.compile(
    r"\b(?:revisit|number of reasons|first being|let's|we're going to|welcome back|"
    r"take a look|as you can see|going to talk|today we|so here|now let's)\b", re.I)
VAGUE_WORDS = {"revisit", "additional", "information", "foundation", "foundations", "overview", "message", "messages"}

HTTP_REQUIRED_TOPICS = {
    "http": ["http"],
    "request": ["request"],
    "response": ["response"],
    "method": ["method", "get", "post", "put", "delete", "verb"],
    "status code": ["status code", "status", "200", "404", "500"],
    "header": ["header"],
    "body": ["body"],
    "url": ["url"],
    "uri": ["uri", "uniform resource identifier"],
    "https": ["https", "port 443", "secure"],
    "multiplexing": ["multiplexing"],
    "cookie": ["cookie"],
    "curl": ["curl"],
}


@dataclass
class QualityReport:
    passed: bool = True
    checks: dict = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def check(self, name: str, ok: bool, detail) -> None:
        self.checks[name] = detail
        if not ok:
            self.passed = False
            self.failures.append(f"{name}: {detail}")


def _walk(node: dict):
    yield node
    for child in node.get("children") or []:
        yield from _walk(child)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def analyze_map(mindmap: dict) -> QualityReport:
    r = QualityReport()
    _domain_transcript = mindmap.get("transcript") or []
    http_hits = sum(1 for e in _domain_transcript[:120] if re.search(r"(?<![a-z])http(?![a-z])", e.get("text") or "", re.I))
    http_evidence = http_hits >= 4 or any("hypertext transfer" in (e.get("text") or "").lower() for e in _domain_transcript[:120])
    nodes = list(_walk(mindmap["root"]))
    concepts = [n for n in nodes if n.get("type") == "concept" and not n.get("oneline") and not n.get("recall")]
    details = [n for n in nodes if n.get("type") == "detail"]

    transcript_like = [n["text"] for n in concepts if TRANSCRIPT_MARKERS.search(n.get("text", ""))]
    r.check("concepts_not_transcript_like", not transcript_like, {"count": len(transcript_like), "examples": transcript_like[:5]})

    vague = [n["text"] for n in concepts
             if _norm((n.get("text", "").split(":", 1)[0]).strip()) in VAGUE_WORDS]
    r.check("concepts_not_vague", not vague, {"count": len(vague), "examples": vague[:5]})

    norms = [_norm(n.get("text", "")) for n in concepts]
    dupes = sorted({x for x in norms if norms.count(x) > 1})
    r.check("concepts_unique", not dupes, {"count": len(dupes), "examples": dupes[:5]})

    corpus = " ".join((n.get("topic", "") + " " + n.get("text", "")).lower() for n in concepts)
    if http_evidence:
        missing = [name for name, aliases in HTTP_REQUIRED_TOPICS.items() if not any(a in corpus for a in aliases)]
        r.check("http_topic_coverage", not missing, {"missing": missing, "covered": len(HTTP_REQUIRED_TOPICS) - len(missing), "total": len(HTTP_REQUIRED_TOPICS)})

    concept_titles = " ".join(
        (n.get("topic", "") + " " + n.get("text", "") + " " + n.get("explain", "") + " " + n.get("mechanism", "")
         + " " + n.get("example", "") + " " + n.get("exercise", "")).lower()
        for n in concepts
    )
    if http_evidence:
        absent = [name for name, aliases in HTTP_REQUIRED_TOPICS.items() if not any(a in concept_titles for a in aliases)]
        r.check("essential_concepts_present", not absent, {"absent": absent})

    misaligned = []
    for d in details:
        parent = next((p for p in nodes if any(c.get("id") == d.get("id") for c in p.get("children") or [])), None)
        if not parent:
            continue
        ds, de = d.get("start"), d.get("end")
        ps, pe = parent.get("start"), parent.get("end")
        if ds is None or de is None or ps is None or pe is None:
            continue
        if ds < ps - 1:  # supporting detail must not predate its evidence anchor
            misaligned.append({"detail": d.get("text"), "detail_at": round(ds, 1), "concept_range": [round(ps, 1), round(pe, 1)]})
    r.check("detail_timestamp_alignment", not misaligned, {"count": len(misaligned), "examples": misaligned[:5]})

    with_explain = [n for n in concepts if n.get("explain")]
    r.check("beginner_explanations", len(with_explain) >= max(1, len(concepts) // 2), {"count": len(with_explain), "concepts": len(concepts)})

    with_examples = [n for n in concepts if n.get("example") or n.get("worked_example")]
    with_exercises = [n for n in nodes if n.get("exercise")]
    with_diagrams = [n for n in nodes if n.get("diagram")]
    r.check("learning_aids", bool(with_examples and with_exercises),
            {"examples": len(with_examples), "exercises": len(with_exercises), "diagrams": len(with_diagrams)})

    placeholders = [n["text"] for n in concepts if re.search(
        r"key idea introduced|see the timestamped source|named idea from this topic|"
        r"explanation unavailable|a concept covered in this section|step 1: locate|step 2: connect|"
        r"verified transcript evidence|evidence 1:|observed in the video at|using the evidence at",
        n.get("explain", "") + " " + n.get("mechanism", "") + " " + n.get("text", ""), re.I)]
    r.check("no_placeholder_explanations", not placeholders, {"count": len(placeholders), "examples": placeholders[:4]})

    unavailable = [n["text"] for n in concepts if n.get("available") is not True]
    r.check("all_explanations_available", not unavailable, {"count": len(unavailable), "examples": unavailable[:4]})

    weak = [n["text"] for n in concepts if n.get("faith", 1.0) < 0.30]
    r.check("concept_grounding", not weak, {"count": len(weak), "examples": weak[:4]})

    transcript = mindmap.get("transcript") or []
    is_http = http_evidence
    starts = {round(e.get("start", 0), 1) for e in transcript}
    bad_starts = [{"text": n["text"], "start": n.get("start")} for n in concepts
                  if n.get("start") is None or min((abs(n["start"] - st) for st in starts), default=1e9) > 5]
    r.check("valid_evidence_timestamps", not bad_starts, {"count": len(bad_starts), "examples": bad_starts[:4]})

    evidence_mismatch = [{"text": n["text"], "start": n.get("start"), "evidence": (n.get("evidence") or [{}])[0].get("start")}
                         for n in concepts if n.get("evidence") and abs((n.get("start") or 0) - ((n.get("evidence") or [{}])[0].get("start") or 0)) > 2]
    r.check("evidence_timestamp_alignment", not evidence_mismatch, {"count": len(evidence_mismatch), "examples": evidence_mismatch[:4]})

    weak_source = [{"text": n["text"], "source": (n.get("source") or "")[:40]} for n in concepts
                   if len((n.get("source") or "").strip()) < 30]
    r.check("evidence_source_length", not weak_source, {"count": len(weak_source), "examples": weak_source[:4]})

    exercises = [n.get("exercise", "") for n in concepts]
    generic = [e for e in exercises if re.search(r"find one real .* example and explain|try it: find one real", e, re.I)]
    distinct = {_norm(e) for e in exercises}
    r.check("specific_exercises", not generic and (not exercises or len(distinct) >= max(1, len(exercises) // 2)),
            {"generic": len(generic), "distinct": len(distinct), "total": len(exercises)})

    # Topic timestamps: a topic concept must start near its first literal mention in the transcript.
    transcript = mindmap.get("transcript") or []
    bad_timestamps = []
    if http_evidence:
        for name, aliases in HTTP_REQUIRED_TOPICS.items():
            titled = [n for n in concepts if any(a in (n.get("topic", "") + " " + n.get("text", "")).lower() for a in aliases)]
            if not titled:
                continue
            topic_node = titled[0]
            first = next((e.get("start") for e in transcript if any(re.search(r"(?<![a-z])" + re.escape(a) + r"(?![a-z])", e.get("text", ""), re.I) for a in aliases)), None)
            if first is None or topic_node.get("start") is None:
                continue
            if abs(topic_node["start"] - first) > 600:
                bad_timestamps.append({"topic": name, "node_at": round(topic_node["start"], 1), "first_mention": round(first, 1)})
    r.check("topic_timestamps", not bad_timestamps, {"count": len(bad_timestamps), "examples": bad_timestamps[:5]})

    # Explanations must be teaching text, not a restatement of the transcript evidence.
    copies = []
    for n in concepts:
        explain = _norm(n.get("explain", ""))
        evidence = _norm((n.get("evidence") or [{}])[0].get("text", "") or n.get("source", ""))
        import difflib
        overlap = difflib.SequenceMatcher(None, explain, evidence).ratio()
        if len(explain) < 40 or overlap > 0.75:
            copies.append({"text": n.get("text"), "overlap": round(overlap, 2)})
    r.check("definitions_not_transcript_copies", not copies, {"count": len(copies), "examples": copies[:4]})

    # Topic/title/evidence consistency.
    inconsistent = []
    for n in concepts:
        topic = _norm(n.get("topic", ""))
        evidence = _norm((n.get("evidence") or [{}])[0].get("text", "") or n.get("source", ""))
        title = _norm(n.get("text", ""))
        teaching = title + " " + _norm(n.get("explain", "")) + " " + _norm(n.get("mechanism", ""))
        if topic and not any(w in evidence for w in topic.split()) and not any(w in teaching for w in topic.split()):
            inconsistent.append({"text": n.get("text"), "topic": n.get("topic")})
    r.check("topic_evidence_consistency", not inconsistent, {"count": len(inconsistent), "examples": inconsistent[:4]})

    def_mismatch = []
    for n in concepts:
        topic = _norm(n.get("topic", ""))
        explain = _norm(" ".join(str(n.get(k) or "") for k in ("text", "explain", "mechanism", "example", "exercise")))
        if topic and not any(w in explain for w in topic.split()):
            def_mismatch.append({"text": n.get("text"), "topic": n.get("topic")})
    r.check("definition_topic_alignment", not def_mismatch, {"count": len(def_mismatch), "examples": def_mismatch[:4]})

    # Semantic deduplication across the whole map.
    import difflib
    keys = [(_norm(n.get("topic") or n.get("text")), _norm(n.get("text"))) for n in concepts]
    dup = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            if (a[0] and a[0] == b[0]) or (a[0] and a[0] == b[0] and a[1] and b[1]
                                          and difflib.SequenceMatcher(None, a[1], b[1]).ratio() >= 0.90):
                dup.append([concepts[i].get("text"), concepts[j].get("text")])
    r.check("semantic_dedupe", not dup, {"count": len(dup), "examples": dup[:4]})

    diagrams = [n for n in concepts if n.get("diagram")]
    r.check("educational_diagrams", len(diagrams) >= max(1, len(concepts) // 4),
            {"count": len(diagrams), "concepts": len(concepts)})

    # Concept ownership: a concept must live in the section that covers its evidence.
    wrong_section = []
    for sec in mindmap["root"].get("children", []):
        if sec.get("recall"):
            continue
        st, en = sec.get("start"), sec.get("end")
        for n in sec.get("children", []):
            if n.get("type") != "concept" or n.get("start") is None or st is None or en is None:
                continue
            if not (st - 2 <= n["start"] <= en + 2):
                wrong_section.append({"text": n.get("text"), "at": round(n["start"], 1),
                                      "section": [round(st, 1), round(en, 1)]})
    r.check("concept_section_ownership", not wrong_section, {"count": len(wrong_section), "examples": wrong_section[:4]})

    # Teaching content must be real instruction, not vague filler or transcript paraphrase.
    weak_teaching = []
    is_deep_quality = (mindmap.get("meta") or {}).get("mode") == "deep"
    for n in concepts:
        if is_deep_quality:
            if not n.get("lesson_available"):
                weak_teaching.append(n.get("text"))
            continue
        explain = (n.get("explain") or "").strip()
        mechanism = (n.get("mechanism") or "") or " ".join(n.get("steps") or [])
        example = (n.get("example") or n.get("worked_example") or "").strip()
        if (len(explain) < 40 or len(example) < 40
                or not re.search(r"\b\d[.)]\s", mechanism)
                or re.search(r"is a topic that will be covered|gaining more popularity|"
                             r"available information about|key idea introduced", explain, re.I)):
            weak_teaching.append(n.get("text"))
    r.check("technical_teaching_quality", not weak_teaching, {"count": len(weak_teaching), "examples": weak_teaching[:4]})

    unsupported = [n.get("text") for n in nodes if re.search(r"(?:the speaker|the video|the lecturer)\s+(?:says|states|said|claims)", n.get("explain", "") + " " + n.get("example", ""), re.I)]
    r.check("no_false_attribution", not unsupported, {"count": len(unsupported), "examples": unsupported[:3]})

    # Every node must have a valid time range. Broken ranges were observed in exported maps.
    bad_ranges = [{"text": n.get("text"), "start": n.get("start"), "end": n.get("end")}
                  for n in nodes if n.get("start") is not None and n.get("end") is not None
                  and float(n["end"]) < float(n["start"])]
    r.check("valid_time_ranges", not bad_ranges, {"count": len(bad_ranges), "examples": bad_ranges[:4]})

    # Confidence must be measured, never a row of perfect scores.
    faiths = [n.get("faith") for n in concepts if isinstance(n.get("faith"), (int, float))]
    if faiths:
        inflated = sum(1 for f in faiths if f >= 1.0)
        r.check("honest_confidence", inflated == 0,
                {"count": inflated, "max": max(faiths), "total": len(faiths)})

    # Deep Mode lessons must be complete, specific and grounded.
    is_deep = (mindmap.get("meta") or {}).get("mode") == "deep"
    if is_deep:
        incomplete = []
        for n in concepts:
            missing = []
            if not (n.get("explain") or "").strip(): missing.append("definition")
            if not (n.get("why_it_matters") or "").strip(): missing.append("why_it_matters")
            steps = n.get("steps")
            if not isinstance(steps, list) or len(steps) < 3: missing.append("steps")
            if not (n.get("worked_example") or "").strip(): missing.append("worked_example")
            diagram = (n.get("diagram") or "").strip()
            if len(diagram) < 20 or not re.search(r"->|<->|--|\[|\]|\|", diagram): missing.append("diagram")
            if not (n.get("misconception") or "").strip(): missing.append("misconception")
            if not (n.get("exercise") or "").strip(): missing.append("exercise")
            check = n.get("knowledge_check")
            if not isinstance(check, dict) or not check.get("question") or not check.get("answer"):
                missing.append("knowledge_check")
            if n.get("provenance", {}).get("example_source") != "supplemental":
                missing.append("supplemental_provenance")
            if missing:
                incomplete.append({"text": n.get("text"), "missing": missing})
        r.check("deep_lesson_complete", not incomplete, {"count": len(incomplete), "examples": incomplete[:4]})

        # Canvas must stay concise: no raw transcript leaves under concepts.
        leaves = [c.get("text") for sec in mindmap["root"].get("children", [])
                  for c in sec.get("children", []) if c.get("type") == "concept"
                  for ch in c.get("children", []) if ch.get("type") == "transcript"]
        r.check("deep_canvas_concise", not leaves, {"count": len(leaves), "examples": leaves[:3]})

        # Supplemental teaching must not be presented as video statements.
        copied = []
        for n in concepts:
            ev = _norm((n.get("evidence") or [{}])[0].get("text", "") or n.get("source", ""))
            for field in ("worked_example", "diagram", "misconception", "exercise"):
                val = _norm(str(n.get(field) or ""))
                if val and len(val) > 25 and val in ev:
                    copied.append(n.get("text"))
                    break
        r.check("supplemental_not_copied", not copied, {"count": len(copied), "examples": copied[:4]})
    return r
