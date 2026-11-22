"""Reconcile reported stats with the actual generated export files."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from app.quality import analyze_map  # noqa: E402
from app.pipeline.terminology import is_meaningful_label  # noqa: E402

VER = ROOT / "examples" / "verification"


def walk(n):
    yield n
    for c in n.get("children", []):
        yield from walk(c)


def audit(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    meta = d.get("meta", {})
    sections = [s for s in d["root"].get("children", []) if not s.get("recall")]
    recall = next((s for s in d["root"].get("children", []) if s.get("recall")), {})
    concepts = [n for s in sections for n in s.get("children", []) if n.get("type") == "concept"]
    lesson_nodes = [c for c in concepts if c.get("lesson_available")]
    qa = [c for c in recall.get("children", []) if c.get("qa")]
    raw_recall = [c for c in recall.get("children", []) if not c.get("oneline") and not c.get("qa")]
    stats = meta.get("labelStats") or {}
    report = analyze_map(d)
    labels = [c.get("text") for c in concepts]
    original = meta.get("originalTranscript") or []
    transcript = d.get("transcript") or []
    return {
        "file": path.name,
        "videoId": meta.get("videoId"),
        "mode": meta.get("mode"),
        "sourceLanguage": meta.get("sourceLanguage"),
        "teachingLanguage": meta.get("teachingLanguage"),
        "provider": meta.get("llm"),
        "embeddingBackend": meta.get("embeddings"),
        "deepStatus": meta.get("deepStatus"),
        "generationMs": meta.get("generationMs"),
        "reported": {
            "lessonsAttempted": stats.get("lessonsAttempted"),
            "lessonsAccepted": stats.get("lessonsAccepted"),
            "lessonsRejected": stats.get("lessonsRejected"),
            "failedLabels": stats.get("lesson_failures"),
        },
        "actual": {
            "sections": len(sections),
            "concepts": len(concepts),
            "lessonAvailableNodes": len(lesson_nodes),
            "recallChildren": len(recall.get("children", [])),
            "recallQaNodes": len(qa),
            "rawTranscriptRecallFragments": len(raw_recall),
            "transcriptUnits": len(transcript),
            "originalTranscriptUnits": len(original),
            "originalTimestampsPreserved": all(
                abs(o["start"] - t["start"]) < 0.01 for o, t in zip(original, transcript)
            ) if original else None,
            "fragmentaryLabels": [l for l in labels if not is_meaningful_label(l or "")],
            "labels": labels,
        },
        "quality": {
            "passed": report.passed,
            "structural": report.structural_passed,
            "grounding": report.grounding_passed,
            "teaching_ready": report.teaching_ready,
            "failures": report.failures,
        },
        "reconciliation": {
            "acceptedEqualsNodes": (stats.get("lessonsAccepted") == len(lesson_nodes)),
            "acceptedAtGeneration": stats.get("lessonsAcceptedAtGeneration"),
            "droppedByGroundingGates": stats.get("lessonsDroppedByGroundingGates"),
            "attemptedEqualsAcceptedPlusRejected": (
                stats.get("lessonsAttempted") ==
                (stats.get("lessonsAcceptedAtGeneration", stats.get("lessonsAccepted")) or 0)
                + (stats.get("lessonsRejected") or 0)
            ),
            "deepStatusRejectedCountMatches": (
                str(stats.get("lessonsRejected")) in (meta.get("deepStatus") or "")
                if stats.get("lessonsRejected") is not None else None
            ),
        },
    }


def main():
    out = [audit(p) for p in sorted(VER.glob("*-verified.json"))]
    (VER / "audit-summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
