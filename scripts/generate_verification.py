"""Reproducible Mivmap verification CLI.

Three distinct modes, reported honestly:

1. ``--offline``            synthetic/offline verification (no provider needed)
2. ``--provider-synthetic`` provider-backed generation from original synthetic captions
3. ``--transcript-file``    generation from a user-supplied transcript

None of these claim a real YouTube video was fetched or accepted. A real-video
acceptance test requires fetching a transcript from YouTube and is reported as
``NOT RUN`` unless the caller explicitly performs it outside this script.

Usage (from repo root):
    backend\\.venv\\Scripts\\python.exe scripts\\generate_verification.py --offline
    backend\\.venv\\Scripts\\python.exe scripts\\generate_verification.py --provider-synthetic
    backend\\.venv\\Scripts\\python.exe scripts\\generate_verification.py \
        --transcript-file my_transcript.json --teaching-language hi --label "My Hindi test"

Transcript input format (UTF-8 JSON):
    {
      "title": "Optional human-readable title",
      "videoId": "optional-source-id",
      "teachingLanguage": "hi",
      "transcript": [
        {"start": 0.0, "end": 7.5, "text": "..."}
      ]
    }

No generated exports are written unless ``--out-dir`` is supplied. Output files
are verification artifacts, not repository fixtures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from app.pipeline import llm  # noqa: E402
from app.pipeline.builder import generate_mindmap  # noqa: E402
from app.pipeline.embeddings import backend_name  # noqa: E402
from app.pipeline.text_utils import normalize_language  # noqa: E402
from app.quality import analyze_map  # noqa: E402

SUPPORTED_TEACHING_LANGUAGES = {"en", "hi"}

# Original synthetic captions authored for this repository. They are NOT
# transcripts of any YouTube video and must never be reported as such.
SYNTHETIC_EN = [
    "A local area network connects devices inside one building.",
    "A LAN lets those devices share files and a printer.",
    "A router forwards traffic from the local network to another network.",
    "A wide area network connects networks across cities and countries.",
    "The internet is a global network of networks that uses TCP/IP.",
    "Fibre cables carry most long-distance data between continents.",
]

SYNTHETIC_HI = [
    "आज हम कंप्यूटर नेटवर्क समझेंगे। नेटवर्क का मतलब आपस में जुड़े उपकरण हैं।",
    "स्थानीय नेटवर्क एक इमारत के अंदर कंप्यूटरों और प्रिंटरों को जोड़ता है।",
    "राउटर पैकेट को एक नेटवर्क से दूसरे नेटवर्क तक ले जाता है।",
    "इंटरनेट दुनिया भर के नेटवर्कों को जोड़ने वाला एक बड़ा नेटवर्क है।",
    "ऑप्टिकल फाइबर केबल समुद्र के नीचे से डेटा ले जाती है।",
    "सेलुलर टावर फोन से रेडियो सिग्नल लेता है और फाइबर से ऑपरेटर को भेजता है।",
]

SYNTHETIC_HI_TECH = [
    "आज हम networking पढ़ेंगे। Internet और WAN में क्या अंतर है?",
    "LAN मतलब Local Area Network। यह एक office के अंदर devices को जोड़ता है।",
    "WAN कई cities और countries को जोड़ता है।",
    "Router पैकेट को एक network से दूसरे network तक ले जाता है।",
    "Fiber cable समुद्र के नीचे से international traffic ले जाती है।",
    "ONT यानी Optical Network Terminal, घर पर fiber को Ethernet में बदलता है।",
]


def _entries(lines: list[str], step: float = 9.0) -> list[dict]:
    return [{"start": round(i * step, 2), "end": round(i * step + step - 1, 2), "text": text}
            for i, text in enumerate(lines)]


def validate_input(data: Any) -> list[str]:
    """Validate a user-supplied transcript file. Never invent missing metadata."""
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["input must be a JSON object"]
    if not str(data.get("title") or "").strip():
        errors.append("title must be a non-empty string")
    lang_raw = data.get("teachingLanguage")
    if not lang_raw:
        errors.append("teachingLanguage is required")
    elif normalize_language(lang_raw) not in SUPPORTED_TEACHING_LANGUAGES:
        errors.append("teachingLanguage must be en or hi (locale variants allowed)")
    transcript = data.get("transcript")
    if not isinstance(transcript, list) or not transcript:
        return errors + ["transcript must be a non-empty list"]
    for i, entry in enumerate(transcript):
        if not isinstance(entry, dict):
            errors.append(f"transcript[{i}] must be an object")
            continue
        start, end, text = entry.get("start"), entry.get("end"), entry.get("text")
        if isinstance(start, bool) or not isinstance(start, (int, float)):
            errors.append(f"transcript[{i}].start must be numeric")
        if isinstance(end, bool) or not isinstance(end, (int, float)):
            errors.append(f"transcript[{i}].end must be numeric")
        if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end <= start:
            errors.append(f"transcript[{i}].end must be greater than start")
        if not isinstance(text, str) or not text.strip():
            errors.append(f"transcript[{i}].text must be non-empty")
    return errors


def real_video_status() -> dict:
    return {
        "status": "NOT RUN",
        "reason": "No transcript was fetched from YouTube; this script never claims real-video acceptance.",
    }


def provider_status(use_llm: bool, label: str) -> dict:
    if not use_llm:
        return {"status": "NOT RUN", "reason": "offline mode; no provider requested"}
    if not llm.fast_available():
        return {"status": "NOT RUN", "reason": f"no provider configured or available for {label}"}
    return {"status": "RUN", "provider": llm.provider_name()}


def build_report(label: str, scenario_type: str, request: dict, mm: dict,
                 seconds: float, use_llm: bool, video_id_label: str) -> dict:
    sections = [s for s in mm["root"].get("children", []) if not s.get("recall")]
    concepts = [c for s in sections for c in s.get("children", []) if c.get("type") == "concept"]
    available = [c for c in concepts if c.get("lesson_available") or c.get("available")]
    recall = next((s for s in mm["root"].get("children", []) if s.get("recall")), {})
    qa = [c for c in recall.get("children", []) if c.get("qa")]
    report = analyze_map(mm)
    meta = mm.get("meta") or {}
    return {
        "label": label,
        "verification_type": scenario_type,
        "source_id": video_id_label,
        "is_real_youtube_video": False,
        "provider_backed_generation": provider_status(use_llm, label),
        "real_youtube_acceptance": real_video_status(),
        "mode": meta.get("mode"),
        "sourceLanguage": meta.get("sourceLanguage"),
        "teachingLanguage": meta.get("teachingLanguage"),
        "provider": meta.get("llm"),
        "embeddingBackend": meta.get("embeddings"),
        "deepStatus": meta.get("deepStatus"),
        "deepRejectionReasons": meta.get("deepRejectionReasons") or [],
        "sections": len(sections),
        "concepts": len(concepts),
        "availableLessons": len(available),
        "rejected": (meta.get("labelStats") or {}).get("lessonsRejected"),
        "recallQaNodes": len(qa),
        "originalTranscriptUnits": len(meta.get("originalTranscript") or []),
        "generationSeconds": round(seconds, 2),
        "qualityPassed": report.passed,
        "qualityFailures": report.failures,
        "conceptLabels": [c.get("text") for c in concepts],
    }


def run_case(label: str, scenario_type: str, request: dict, use_llm: bool,
             out_dir: Path | None, video_id_label: str) -> dict:
    request = dict(request)
    request["useLLM"] = use_llm
    request["allowWhisper"] = False
    started = time.perf_counter()
    print(f"\n=== {label} | {scenario_type} | provider={'yes' if use_llm else 'no'} ===", flush=True)
    mm = generate_mindmap(request)
    result = build_report(label, scenario_type, request, mm,
                          time.perf_counter() - started, use_llm, video_id_label)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_-]+", "-", label).strip("-") or "verification"
        path = out_dir / f"{safe}.json"
        path.write_text(json.dumps(mm, ensure_ascii=False, indent=2), encoding="utf-8")
        result["exportWritten"] = str(path)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def _synthetic_cases() -> list[tuple[str, list[str], str]]:
    return [
        ("synthetic-en-offline", SYNTHETIC_EN, "en"),
        ("synthetic-hi-devanagari", SYNTHETIC_HI, "hi"),
        ("synthetic-hi-english-technical", SYNTHETIC_HI_TECH, "hi"),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true", help="synthetic verification, no provider calls")
    ap.add_argument("--provider-synthetic", action="store_true",
                    help="provider-backed generation from original synthetic captions")
    ap.add_argument("--transcript-file", type=Path, help="user-supplied UTF-8 transcript JSON")
    ap.add_argument("--teaching-language", help="override teaching language for --transcript-file")
    ap.add_argument("--label", default="user-supplied transcript", help="report label")
    ap.add_argument("--out-dir", type=Path, help="optional directory for generated verification artifacts")
    args = ap.parse_args()
    if sum(bool(x) for x in (args.offline, args.provider_synthetic, args.transcript_file)) != 1:
        ap.error("choose exactly one of --offline, --provider-synthetic or --transcript-file")

    results: list[dict] = []
    if args.transcript_file:
        try:
            data = json.loads(args.transcript_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            ap.error(f"cannot read transcript file: {exc}")
        errors = validate_input(data)
        if args.teaching_language:
            data["teachingLanguage"] = args.teaching_language
            errors = validate_input(data)
        if errors:
            ap.error("invalid transcript input: " + "; ".join(errors))
        video_id_label = str(data.get("videoId") or "user-supplied")
        request = {
            "videoId": f"external-{hashlib.sha1(video_id_label.encode('utf-8')).hexdigest()[:8]}",
            "title": str(data["title"]),
            "transcript": data["transcript"],
            "mode": "deep",
            "teachingLanguage": data["teachingLanguage"],
            "maxConcepts": 6,
        }
        if not llm.fast_available():
            print(json.dumps({
                "label": args.label,
                "verification_type": "user-supplied-transcript",
                "source_id": video_id_label,
                "provider_backed_generation": provider_status(True, args.label),
                "real_youtube_acceptance": real_video_status(),
                "status": "NOT RUN",
            }, ensure_ascii=False, indent=2), flush=True)
            return 0
        results.append(run_case(args.label, "user-supplied-transcript", request, True,
                                args.out_dir, video_id_label))
    else:
        use_llm = bool(args.provider_synthetic)
        if use_llm and not llm.fast_available():
            print(json.dumps({
                "verification_type": "provider-backed-synthetic",
                "provider_backed_generation": provider_status(True, "synthetic cases"),
                "real_youtube_acceptance": real_video_status(),
                "status": "NOT RUN",
            }, ensure_ascii=False, indent=2), flush=True)
            return 0
        for label, lines, lang in _synthetic_cases():
            request = {
                "videoId": f"synthetic-{label}",
                "title": label,
                "transcript": _entries(lines),
                "mode": "deep",
                "teachingLanguage": lang,
                "maxConcepts": 6,
            }
            results.append(run_case(label,
                                    "provider-backed-synthetic" if use_llm else "synthetic-offline",
                                    request, use_llm, args.out_dir, f"synthetic:{label}"))

    print("\n=== SUMMARY ===", flush=True)
    for r in results:
        print(f"{r['label']} | {r['verification_type']} | concepts {r['concepts']} | "
              f"lessons {r['availableLessons']} | QA {r['recallQaNodes']} | "
              f"quality {'PASS' if r['qualityPassed'] else 'FAIL'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
