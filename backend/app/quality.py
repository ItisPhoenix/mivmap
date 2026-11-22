"""Automated quality checks for generated Mivmap maps.

These checks are intentionally deterministic: they look for the concrete defects found in
the HTTP Fundamentals regression fixture (transcript-like labels, vague/duplicate concepts,
missing syllabus topics and detail nodes whose timestamps do not match their source).
"""
from __future__ import annotations

import difflib
import re
from collections import Counter
from dataclasses import dataclass, field

TRANSCRIPT_MARKERS = re.compile(
    r"\b(?:revisit|number of reasons|first being|let's|we're going to|welcome back|"
    r"take a look|as you can see|going to talk|today we|so here|now let's)\b", re.I)
VAGUE_WORDS = {"revisit", "additional", "information", "foundation", "foundations", "overview", "message", "messages"}

from .pipeline.topics import ESSENTIAL_LABELS, detect_domain
from .pipeline.text_utils import check_teaching_languages, unicode_norm
from .pipeline.terminology import (acronym_expansion, is_meaningful_label,
                                   correction_required, misleading_detail,
                                   normalize_label)

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

NETWORKING_REQUIRED_TOPICS = {
    "LAN": ["lan", "local area network", "लैन", "लोकल नेटवर्क"],
    "PAN": ["pan", "personal area network"],
    "MAN": ["man", "metropolitan area network"],
    "WAN": ["wan", "wide area network", "वान", "वैन", "वाइड एरिया नेटवर्क"],
    "ARPANET": ["arpanet"],
    "TCP/IP": ["tcp/ip", "tcp and ip", "transmission control protocol", "टीसीपी"],
    "OSI": ["osi", "open systems interconnection", "ओएसआई"],
    "ISP tiers": ["tier", "isp", "internet service provider"],
    "fibre infrastructure": ["fibre", "fiber", "optical fibre", "undersea cable", "फाइबर"],
    "CDNs": ["cdn", "content delivery network"],
}


CATEGORY = {
    # checks that fail a map's structural integrity (skeleton, timing, uniqueness)
    # default: structural
    # checks about measured grounding of concepts (faith, transcript copies)
    "all_explanations_available": "grounding",
    "concept_grounding": "grounding",
    "definitions_not_transcript_copies": "grounding",
    "honest_confidence": "grounding",
    # checks about whether the teaching itself is real, distinct and usable
    "beginner_explanations": "teaching_ready",
    "no_placeholder_explanations": "teaching_ready",
    "specific_exercises": "teaching_ready",
    "educational_diagrams": "teaching_ready",
    "definition_topic_alignment": "teaching_ready",
    "technical_teaching_quality": "teaching_ready",
    "grounded_study_aids": "teaching_ready",
    "substantive_teaching": "teaching_ready",
    "no_false_attribution": "teaching_ready",
    "deep_lesson_complete": "teaching_ready",
    "deep_canvas_concise": "teaching_ready",
    "supplemental_not_copied": "teaching_ready",
    "detail_parent_semantics": "teaching_ready",
    "detail_no_contradiction": "teaching_ready",
    "provenance_honest": "teaching_ready",
    "external_citations": "teaching_ready",
    "concept_label_explanation_consistency": "teaching_ready",
    "language_consistency": "teaching_ready",
    "recall_meaningful": "teaching_ready",
    "root_summary_clean": "teaching_ready",
}


@dataclass
class QualityReport:
    # Three readiness categories; `passed` requires all three.
    structural_passed: bool = True
    grounding_passed: bool = True
    teaching_ready: bool = True
    passed: bool = True
    checks: dict = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def check(self, name: str, ok: bool, detail, category: str | None = None) -> None:
        self.checks[name] = detail
        if not ok:
            self.passed = False
            setattr(self, category or CATEGORY.get(name, "structural"), False)
            self.failures.append(f"{name}: {detail}")


def _walk(node: dict):
    yield node
    for child in node.get("children") or []:
        yield from _walk(child)


def _norm(text: str) -> str:
    return unicode_norm(_s(text))


def _s(value) -> str:
    """Flatten any teaching-field value (LLM aids can be lists/dicts) to safe text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        return " ".join(_s(v) for v in value)
    if isinstance(value, dict):
        return " ".join(_s(v) for v in value.values())
    return str(value)


def analyze_map(mindmap: dict) -> QualityReport:
    r = QualityReport()
    _domain_transcript = mindmap.get("transcript") or []
    domain, _topic_hits = detect_domain([{"text": str(e.get("text") or ""), "start": e.get("start", 0)}
                                         for e in _domain_transcript[:400]])
    required_topics = {"http": HTTP_REQUIRED_TOPICS, "networking": NETWORKING_REQUIRED_TOPICS}.get(domain)
    http_evidence = required_topics is not None
    nodes = list(_walk(mindmap["root"]))
    concepts = [n for n in nodes if n.get("type") == "concept" and not n.get("oneline") and not n.get("recall")]
    taught = [n for n in concepts
              if (n.get("lesson_available") or n.get("available")) and not n.get("conceptUnavailable")]
    details = [n for n in nodes if n.get("type") == "detail"]
    # Legacy deep exports predate knowledge_check.explanation; keep them valid.
    is_deep_q = (mindmap.get("meta") or {}).get("mode") == "deep"
    legacy_deep = is_deep_q and not any((c.get("knowledge_check") or {}).get("explanation") for c in concepts)

    transcript_like = [n["text"] for n in concepts if TRANSCRIPT_MARKERS.search(n.get("text", ""))]
    r.check("concepts_not_transcript_like", not transcript_like, {"count": len(transcript_like), "examples": transcript_like[:5]})

    vague = [n["text"] for n in concepts
             if _norm((n.get("text", "").split(":", 1)[0]).strip()) in VAGUE_WORDS]
    r.check("concepts_not_vague", not vague, {"count": len(vague), "examples": vague[:5]})

    # Meaningful terminology: caption fragments and generic single words are not topics.
    fragmentary = [n.get("text") for n in concepts if not is_meaningful_label(n.get("text", ""))]
    r.check("concepts_not_fragmentary", not fragmentary,
            {"count": len(fragmentary), "examples": fragmentary[:5]}, "teaching_ready")

    # Non-Latin labels normalise to ""; fall back to the raw text so distinct
    # Hindi topics are not reported as duplicates of each other.
    norms = [_norm(n.get("text", "")) or _s(n.get("text", "")).strip().casefold() for n in concepts]
    dupes = sorted({x for x in norms if norms.count(x) > 1})
    r.check("concepts_unique", not dupes, {"count": len(dupes), "examples": dupes[:5]})

    transcript_text = " ".join(str(e.get("text") or "") for e in _domain_transcript[:400]).lower()
    corpus = " ".join((_s(n.get("topic")) + " " + _s(n.get("text"))).lower() for n in concepts)
    # A syllabus topic is only required when the video actually mentions it.
    # Demanding ARPANET/CDN concepts from a video that never says those words
    # makes the quality gate reject otherwise complete maps.
    expected_topics = ({name: aliases for name, aliases in (required_topics or {}).items()
                        if any(a.lower() in transcript_text for a in aliases)}
                       if http_evidence else {})
    if http_evidence:
        missing = [name for name, aliases in expected_topics.items()
                   if not any(a in corpus for a in aliases)]
        r.check("http_topic_coverage", not missing,
                {"missing": missing, "covered": len(expected_topics) - len(missing),
                 "total": len(expected_topics)})

    concept_titles = " ".join(
        (_s(n.get("topic")) + " " + _s(n.get("text")) + " " + _s(n.get("explain")) + " " + _s(n.get("mechanism"))
         + " " + _s(n.get("example")) + " " + _s(n.get("exercise"))).lower()
        for n in concepts
    )
    if http_evidence:
        absent = [name for name, aliases in expected_topics.items() if not any(a in concept_titles for a in aliases)]
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
        _s(n.get("explain")) + " " + _s(n.get("mechanism")) + " " + _s(n.get("text")), re.I)]
    r.check("no_placeholder_explanations", not placeholders, {"count": len(placeholders), "examples": placeholders[:4]})

    unavailable = [n["text"] for n in concepts if n.get("available") is not True]
    is_deep_q = (mindmap.get("meta") or {}).get("mode") == "deep"
    enough = (not unavailable) if is_deep_q else (
        not concepts or (len(concepts) - len(unavailable)) / len(concepts) >= 0.6)
    r.check("all_explanations_available", enough,
            {"count": len(unavailable), "examples": unavailable[:4], "mode": "deep" if is_deep_q else "revision"})

    essential_q = {e.lower() for e in ESSENTIAL_LABELS}
    weak = [n["text"] for n in concepts if n.get("faith", 1.0) < 0.30
            and (n.get("topic") or "").lower() not in essential_q]
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
        for name, aliases in required_topics.items():
            titled = [n for n in concepts if any(a in (_s(n.get("topic")) + " " + _s(n.get("text"))).lower() for a in aliases)]
            if not titled:
                continue
            topic_node = titled[0]
            mentions = [e.get("start") for e in transcript
                        if any(re.search(r"(?<![a-z])" + re.escape(a) + r"(?![a-z])", e.get("text", ""), re.I)
                               for a in aliases)]
            nearest = min(mentions, key=lambda x: abs(x - (topic_node.get("start") or 0))) if mentions else None
            if nearest is None or topic_node.get("start") is None:
                continue
            if abs(topic_node["start"] - nearest) > 600:
                bad_timestamps.append({"topic": name, "node_at": round(topic_node["start"], 1), "nearest_mention": round(nearest, 1)})
    r.check("topic_timestamps", not bad_timestamps, {"count": len(bad_timestamps), "examples": bad_timestamps[:5]})

    # Explanations must be teaching text, not a restatement of the transcript evidence.
    copies = []
    for n in concepts:
        if not n.get("available"):
            continue  # unavailable concepts are not successful lessons
        explain = _norm(n.get("explain", ""))
        evidence = _norm((n.get("evidence") or [{}])[0].get("text", "") or n.get("source", ""))
        import difflib
        overlap = difflib.SequenceMatcher(None, explain, evidence).ratio()
        extractive = (n.get("provenance") or {}).get("kind") == "transcript-extractive"
        limit = 0.92 if extractive else 0.75  # extractive notes quote the video on purpose
        if len(_s(n.get("explain", "")).strip()) < 40 or overlap > limit:
            copies.append({"text": n.get("text"), "overlap": round(overlap, 2)})
    r.check("definitions_not_transcript_copies", not copies, {"count": len(copies), "examples": copies[:4]})

    # Topic/title/evidence consistency.
    inconsistent = []
    for n in concepts:
        topic = _norm(n.get("topic", ""))
        evidence = _norm((n.get("evidence") or [{}])[0].get("text", "") or n.get("source", ""))
        title = _norm(n.get("text", ""))
        teaching = title + " " + _norm(_s(n.get("explain"))) + " " + _norm(_s(n.get("mechanism")))
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
    keys = [(_norm(n.get("topic") or n.get("text")) or _s(n.get("topic") or n.get("text")).strip().casefold(),
             _norm(n.get("text")) or _s(n.get("text")).strip().casefold()) for n in concepts]
    dup = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            if (a[0] and a[0] == b[0]) or (a[0] and a[0] == b[0] and a[1] and b[1]
                                          and difflib.SequenceMatcher(None, a[1], b[1]).ratio() >= 0.90):
                dup.append([concepts[i].get("text"), concepts[j].get("text")])
    r.check("semantic_dedupe", not dup, {"count": len(dup), "examples": dup[:4]})

    diagrams = [n for n in concepts if n.get("diagram")]
    sequential = [n for n in concepts if isinstance(n.get("steps"), list) and len(n["steps"]) >= 2]
    if sequential:
        r.check("educational_diagrams", len(diagrams) >= max(1, len(concepts) // 4),
                {"count": len(diagrams), "concepts": len(concepts)})
    else:
        r.check("educational_diagrams", True, {"count": 0, "skipped": "no sequential evidence in transcript"})

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
            if n.get("conceptUnavailable"):
                continue  # explicitly unavailable; reported by all_explanations_available
            if not n.get("lesson_available"):
                weak_teaching.append(n.get("text"))
            continue
        if not n.get("available"):
            continue  # fail-closed: unavailable concepts are not successful lessons
        check = n.get("knowledge_check") if isinstance(n.get("knowledge_check"), dict) else {}
        aids = [n.get("steps"), n.get("worked_example") or n.get("example"),
                n.get("diagram"), n.get("why_it_matters"), n.get("misconception")]
        if (len(_s(n.get("explain"))) < 40 or len(_s(n.get("exercise")).strip()) < 30
                or not check.get("question") or not check.get("answer")
                or re.search(r"is a topic that will be covered|gaining more popularity|"
                             r"available information about|key idea introduced", explain, re.I)):
            weak_teaching.append(n.get("text"))
    r.check("technical_teaching_quality", not weak_teaching, {"count": len(weak_teaching), "examples": weak_teaching[:4]})

    if not is_deep_quality and concepts:
        with_aid = sum(1 for n in concepts if any(
            (n.get("steps"), n.get("worked_example") or n.get("example"),
             n.get("diagram"), n.get("why_it_matters"), n.get("misconception"))))
        r.check("grounded_study_aids", with_aid >= max(1, len(concepts) // 4),
                {"with_aids": with_aid, "concepts": len(concepts)})

    # Polished-but-empty lessons: reject generic filler and repeated teaching text.
    generic_lesson = re.compile(
        r"this concept is important to understand|key part of the topic covered|"
        r"key idea introduced|available information|component that can be part", re.I)
    shallow = []
    for n in concepts:
        if not n.get("available"):
            continue
        explain = str(n.get("explain") or "").strip()
        check = n.get("knowledge_check") if isinstance(n.get("knowledge_check"), dict) else {}
        texts = [str(n.get(k) or "").strip() for k in
                 ("explain", "why_it_matters", "worked_example", "misconception", "exercise")]
        texts = [t for t in texts if t]
        repeated = any(difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() > 0.88
                       for i, a in enumerate(texts) for b in texts[i + 1:])
        # Fragments and single-clause scraps are not teachable notes.
        frag_fields = []
        for key in ("why_it_matters", "worked_example", "misconception", "exercise"):
            value = str(n.get(key) or "").strip()
            if value and (len(value.split()) < 5 or
                          re.match(r"^(?:and|but|or|so|meaning)\b", value, re.I)):
                frag_fields.append(key)
        for step in (n.get("steps") if isinstance(n.get("steps"), list) else []):
            if len(str(step).split()) < 5:
                frag_fields.append("steps")
        if re.search(r"\b(?:It|This|That|They|There)\.\s", explain):
            frag_fields.append("explain_fragment")
        if re.search(r"\b(?:It|This|That|They|There)\.?$", str(check.get("answer") or "").strip(), re.I):
            frag_fields.append("answer_fragment")
        if not legacy_deep and frag_fields:
            shallow.append(f"{n.get('text')} ({','.join(sorted(set(frag_fields)))})")
        if (len(explain) < 40 or not check.get("answer")
                or (not legacy_deep and not str(check.get("explanation") or "").strip())
                or repeated or generic_lesson.search(explain)):
            shallow.append(n.get("text"))
    r.check("substantive_teaching", not shallow, {"count": len(shallow), "examples": shallow[:4]})

    unsupported = [n.get("text") for n in nodes if re.search(r"(?:the speaker|the video|the lecturer)\s+(?:says|states|said|claims)", _s(n.get("explain")) + " " + _s(n.get("example")), re.I)]
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
        for n in taught:
            missing = []
            if not _s(n.get("explain")).strip(): missing.append("definition")
            if not _s(n.get("why_it_matters")).strip(): missing.append("why_it_matters")
            steps = n.get("steps")
            if not isinstance(steps, list) or len(steps) < 3: missing.append("steps")
            if not _s(n.get("worked_example")).strip(): missing.append("worked_example")
            diagram = _s(n.get("diagram")).strip()
            if len(diagram) < 20 or not re.search(r"->|<->|--|\[|\]|\|", diagram): missing.append("diagram")
            if not _s(n.get("misconception")).strip(): missing.append("misconception")
            if not _s(n.get("exercise")).strip(): missing.append("exercise")
            check = n.get("knowledge_check")
            if not isinstance(check, dict) or not check.get("question") or not check.get("answer"):
                missing.append("knowledge_check")
            elif not str(check.get("explanation") or "").strip() and not legacy_deep:
                missing.append("knowledge_check_explanation")
            if n.get("provenance", {}).get("example_source") not in ("supplemental", "ai-generated"):
                missing.append("example_source_provenance")
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
        for n in taught:
            ev = _norm((n.get("evidence") or [{}])[0].get("text", "") or n.get("source", ""))
            for field in ("worked_example", "diagram", "misconception", "exercise"):
                val = _norm(str(n.get(field) or ""))
                if val and len(val) > 25 and val in ev:
                    copied.append(n.get("text"))
                    break
        r.check("supplemental_not_copied", not copied, {"count": len(copied), "examples": copied[:4]})

        # Study-readiness provenance applies to maps generated with the new
        # study-ready pipeline. Legacy exports are deliberately left compatible.
        study_ready = (mindmap.get("meta") or {}).get("generationMs") is not None
        if study_ready:
            opaque = []
            stale_note = []
            wrong_fact = []
            for n in taught:
                prov = n.get("provenance") or {}
                if not all(isinstance(prov.get(k), list) for k in
                           ("video_supported", "ai_explanation", "external_supplement")):
                    opaque.append(n.get("text"))
                if "Teaching content is not generated" in str(n.get("groundingNote") or ""):
                    stale_note.append(n.get("text"))
                expected_expansion = acronym_expansion(n.get("topic") or "")
                if expected_expansion and str(n.get("acronym_expansion") or "").lower() != expected_expansion.lower():
                    opaque.append(f"{n.get('text')}: acronym")
                if correction_required(n.get("topic") or "") and not n.get("accuracy_notes"):
                    wrong_fact.append(f"{n.get('text')}: missing correction")
                body = " ".join(str(n.get(k) or "") for k in
                                ("explain", "intuition", "why_it_matters", "misconception")) + " " + " ".join(
                                    str(s) for s in (n.get("steps") or []) if isinstance(s, str))
                if (re.search(r"full form is (?:obviously )?International Network|a WAN is a type of internet|"
                              r"tower located at the US-based provider", body, re.I)
                        and not n.get("accuracy_notes")):
                    wrong_fact.append(n.get("text"))
            r.check("deep_source_transparency", not opaque,
                    {"count": len(opaque), "examples": opaque[:5]}, "teaching_ready")
            r.check("honest_grounding_note", not stale_note,
                    {"count": len(stale_note), "examples": stale_note[:4]}, "teaching_ready")
            r.check("technical_accuracy_corrections", not wrong_fact,
                    {"count": len(wrong_fact), "examples": wrong_fact[:5]}, "teaching_ready")

            no_stamps = [n.get("text") for n in taught
                         if not ((n.get("lesson_evidence") or {}).get("timestamps")
                                 or (n.get("lesson_evidence") or {}).get("starts"))]
            r.check("deep_lesson_evidence", not no_stamps,
                    {"count": len(no_stamps), "examples": no_stamps[:5]}, "grounding")

            # Quick recall must be revision Q&A, not a row of transcript fragments.
            recall_section = next((s for s in mindmap["root"].get("children", []) if s.get("recall")), None)
            qa_nodes = [c for c in (recall_section or {}).get("children", []) if c.get("qa")]
            if taught and not qa_nodes:
                r.check("quick_recall_is_qa", False,
                        {"qa": 0, "recall_children": len((recall_section or {}).get("children", []))},
                        "teaching_ready")
            else:
                r.check("quick_recall_is_qa", True, {"qa": len(qa_nodes)}, "teaching_ready")

    # New teaching gates (teaching_ready category):
    # a definition names the concept; no reused examples or exercises across concepts;
    # explanations are not duplicated; diagrams are meaningful.

    # 1) Definition contains the concept name (available concepts).
    teaching_terms = []
    for n in concepts:
        if not n.get("available"):
            continue
        topic = _norm(n.get("topic") or n.get("text") or "")
        words = [w for w in topic.split() if len(w) >= 3]
        teaching = _norm(" ".join(str(n.get(k) or "") for k in ("explain", "text"))
                         + " " + _norm(str((n.get("knowledge_check") or {}).get("answer") or "")))
        if words and not any(w in teaching for w in words):
            teaching_terms.append(n.get("text"))
    r.check("teaching_ready_terms", not teaching_terms,
            {"count": len(teaching_terms), "examples": teaching_terms[:4]}, "teaching_ready")

    # 2) No reused exercises (across concepts).
    exercises = [str(n.get("exercise") or "").strip() for n in concepts]
    ex_counts = Counter(_norm(e) for e in exercises if e)
    r.check("teaching_ready_exercise_distinct", not [e for e, c in ex_counts.items() if c > 1],
            {"duplicates": [e for e, c in ex_counts.items() if c > 1][:4]}, "teaching_ready")

    # 3) No reused examples/worked examples (across concepts).
    examples = [str(n.get("worked_example") or n.get("example") or "").strip() for n in concepts]
    ex_counts = Counter(_norm(e) for e in examples if e)
    r.check("teaching_ready_example_distinct", not [e for e, c in ex_counts.items() if c > 1],
            {"duplicates": [e for e, c in ex_counts.items() if c > 1][:4]}, "teaching_ready")

    # 4) No reused intuitions (across concepts).
    intuitions = [str(n.get("intuition") or "").strip() for n in concepts]
    ex_counts = Counter(_norm(e) for e in intuitions if e)
    r.check("teaching_ready_intuition_distinct", not [e for e, c in ex_counts.items() if c > 1],
            {"duplicates": [e for e, c in ex_counts.items() if c > 1][:4]}, "teaching_ready")

    # 5) No duplicated explanations (across available concepts).
    explanations = [str(n.get("explain") or "").strip() for n in concepts if n.get("available")]
    ex_counts = Counter(_norm(e) for e in explanations if e)
    r.check("teaching_ready_duplicate_explanations", not [e for e, c in ex_counts.items() if c > 1],
            {"duplicates": [e for e, c in ex_counts.items() if c > 1][:4]}, "teaching_ready")

    # 6) Meaningful diagrams: a concept with ordered steps needs a diagram with structure.
    shallow_diagrams = [n.get("text") for n in concepts
                        if isinstance(n.get("steps"), list) and len(n["steps"]) >= 2
                        and not re.search(r"->|<->|--|\[|\]|\|", str(n.get("diagram") or ""))]
    r.check("teaching_ready_diagrams", not shallow_diagrams,
            {"count": len(shallow_diagrams), "examples": shallow_diagrams[:4]}, "teaching_ready")

    # ------------------------------------------------------------------
    # Semantic integrity gates. These only apply to maps generated with the
    # study-ready pipeline; legacy exports stay importable and unchanged.
    # ------------------------------------------------------------------
    study_ready = (mindmap.get("meta") or {}).get("generationMs") is not None
    if study_ready:
        teaching_language = str((mindmap.get("meta") or {}).get("teachingLanguage") or "en").lower()
        taught = [n for n in concepts if (n.get("lesson_available") or n.get("available"))]
        stop = {"during", "with", "from", "that", "this", "into", "over", "under", "your", "the"}

        # 1) Every detail must name its own concept.
        wrong_parent = []
        for n in concepts:
            own_terms = [w for w in _norm(n.get("topic") or n.get("text") or "").split()
                         if len(w) >= 3 and w not in stop]
            foreign_terms = {
                w for other in concepts if other is not n
                for w in _norm(other.get("topic") or other.get("text") or "").split()
                if len(w) >= 4 and w not in own_terms
            }
            for ch in n.get("children", []):
                if ch.get("type") != "detail":
                    continue
                body = _norm(ch.get("text") or "")
                if own_terms and not any(t in body for t in own_terms):
                    wrong_parent.append({"parent": n.get("text"), "detail": ch.get("text")})
                elif any(t in body for t in foreign_terms):
                    wrong_parent.append({"parent": n.get("text"), "detail": ch.get("text")})
        r.check("detail_parent_semantics", not wrong_parent,
                {"count": len(wrong_parent), "examples": wrong_parent[:5]}, "teaching_ready")

        # 2) No raw, misleading video claim under a corrected concept.
        contradictions = []
        for n in concepts:
            for ch in n.get("children", []):
                if ch.get("type") == "detail" and misleading_detail(ch.get("text"), n.get("topic") or ""):
                    contradictions.append({"parent": n.get("text"), "detail": ch.get("text")})
        r.check("detail_no_contradiction", not contradictions,
                {"count": len(contradictions), "examples": contradictions[:5]}, "teaching_ready")

        # 3) Provenance must be truthful about external sources.
        opaque = []
        for n in taught:
            prov = n.get("provenance") or {}
            external = prov.get("external_supplement") or []
            if "external" in str(prov.get("source") or "").lower() and not external:
                opaque.append(f"{n.get('text')}: external claimed with empty list")
            if prov.get("example_source") == "external":
                opaque.append(f"{n.get('text')}: AI example marked external")
        r.check("provenance_honest", not opaque,
                {"count": len(opaque), "examples": opaque[:5]}, "teaching_ready")

        # 4) External corrections need citable titles and URLs.
        uncited = []
        for n in taught:
            for note in n.get("accuracy_notes") or []:
                if not note.get("title") or not str(note.get("url") or "").startswith("https://"):
                    uncited.append({"concept": n.get("text"), "note": str(note.get("video_claim") or "")[:60]})
        r.check("external_citations", not uncited,
                {"count": len(uncited), "examples": uncited[:5]}, "teaching_ready")

        # 5) A concept name must describe the teaching under it.
        mismatched = []
        for n in taught:
            label = str(n.get("text") or "").strip()
            terms = [w for w in _norm(label).split() if len(w) >= 3 and w not in stop]
            if terms and not any(t in _norm(n.get("explain") or "") for t in terms):
                mismatched.append(label)
                continue
            # A vague label whose own explanation names a narrower topic must
            # be renamed (e.g. "Fiber" teaching "local network during outage").
            canonical = normalize_label(label, str(n.get("explain") or ""))
            if canonical and _norm(canonical) != _norm(label):
                mismatched.append(f"{label} -> {canonical}")
        r.check("concept_label_explanation_consistency", not mismatched,
                {"count": len(mismatched), "examples": mismatched[:5]}, "teaching_ready")

        # 6) Every learner-facing field must use the selected teaching language.
        # Shared with lesson acceptance so the two checks cannot disagree.
        wrong_lang = []
        for n in taught:
            for detail in check_teaching_languages(n, teaching_language):
                wrong_lang.append(f"{n.get('text')}: {detail}")
        r.check("language_consistency", not wrong_lang,
                {"expected": teaching_language, "count": len(wrong_lang), "examples": wrong_lang[:5]},
                "teaching_ready")

        # 7) Quick recall: no fragments, real questions, grounded answers.
        recall = next((s for s in mindmap["root"].get("children", []) if s.get("recall")), {})
        recall_kids = recall.get("children", [])
        answers = {str((n.get("knowledge_check") or {}).get("answer") or "").strip() for n in taught}
        bad_recall = []
        if str(recall.get("summary") or "").lower().startswith("covers"):
            bad_recall.append("generic summary")
        for c in recall_kids:
            if c.get("oneline") and str(c.get("summary") or c.get("text") or "").lower().startswith("in one line: covers"):
                bad_recall.append("generic one-line")
            if c.get("qa"):
                q = str(c.get("question") or "").strip()
                a = str(c.get("answer") or "").strip()
                if "?" not in q and not re.search(r"\b(?:what|why|how|which|when|where|who)\b", q, re.I):
                    bad_recall.append(f"not a question: {q[:40]}")
                if not a or a not in answers:
                    bad_recall.append(f"ungrounded answer: {a[:40]}")
            elif not c.get("oneline"):
                bad_recall.append(f"raw fragment: {str(c.get('text') or '')[:40]}")
        if taught and not any(c.get("qa") for c in recall_kids):
            bad_recall.append("no Q&A")
        r.check("recall_meaningful", not bad_recall,
                {"count": len(bad_recall), "examples": bad_recall[:5]}, "teaching_ready")

        # 8) Root summary/keywords come from validated concepts, not captions.
        root = mindmap.get("root") or {}
        root_bad = []
        labels = {str(n.get("text") or "").strip() for n in taught}
        if taught and not str(root.get("summary") or "").strip():
            root_bad.append("empty summary")
        if str(root.get("summary") or "").lower().startswith("covers"):
            root_bad.append("generic summary")
        for kw in root.get("keywords") or []:
            if str(kw).strip() not in labels:
                root_bad.append(f"keyword not a validated concept: {kw}")
        r.check("root_summary_clean", not root_bad,
                {"count": len(root_bad), "examples": root_bad[:5]}, "teaching_ready")

    return r
