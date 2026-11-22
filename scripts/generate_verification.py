"""Run REAL generation (cloud LLM + configured embeddings) and write auditable exports.

Usage (from repo root):
    backend\\.venv\\Scripts\\python.exe scripts\\generate_verification.py --smoke
    backend\\.venv\\Scripts\\python.exe scripts\\generate_verification.py

No fixtures are used for generation: transcripts are passed to the real pipeline.
The English transcripts are the verbatim transcripts from existing exports. The
Hindi transcripts are supplied here as source captions with exact timestamps.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from app.pipeline import llm  # noqa: E402
from app.pipeline.builder import generate_mindmap  # noqa: E402
from app.pipeline.embeddings import backend_name  # noqa: E402
from app.quality import analyze_map  # noqa: E402

OUT = ROOT / "examples" / "verification"
OUT.mkdir(parents=True, exist_ok=True)


def _transcript_from_export(name: str) -> list[dict]:
    data = json.loads((ROOT / "examples" / name).read_text(encoding="utf-8"))
    return [{"start": e["start"], "end": e["end"], "text": e["text"]} for e in data["transcript"]]


HINDI_HINDI = [
    "आज हम कंप्यूटर नेटवर्क समझेंगे। नेटवर्क का मतलब आपस में जुड़े उपकरण हैं।",
    "स्थानीय नेटवर्क एक इमारत के अंदर कंप्यूटरों और प्रिंटरों को जोड़ता है।",
    "इससे एक ही इमारत के उपकरण फाइलें साझा कर सकते हैं।",
    "राउटर स्थानीय नेटवर्क को बाहरी नेटवर्क से जोड़ता है और पैकेट को सही रास्ते भेजता है।",
    "इंटरनेट दुनिया भर के नेटवर्कों को जोड़ने वाला एक बड़ा नेटवर्क है।",
    "ऑप्टिकल फाइबर केबल समुद्र के नीचे से दूसरे महाद्वीप तक डेटा ले जाती है।",
    "सेलुलर टावर फोन से रेडियो सिग्नल लेता है और फाइबर से ऑपरेटर को भेजता है।",
    "हर उपकरण का अपना पता होता है ताकि डेटा सही गंतव्य तक पहुंच सके।",
]

HINDI_ENGLISH_TECH = [
    "आज हम networking पढ़ेंगे। Internet और WAN में क्या अंतर है, पहले समझते हैं।",
    "LAN मतलब Local Area Network। यह एक office या घर के अंदर devices को जोड़ता है।",
    "WAN कई cities और countries को जोड़ता है, LAN सिर्फ एक इमारत को।",
    "Router पैकेट को एक network से दूसरे network तक ले जाता है।",
    "Fiber cable समुद्र के नीचे से international traffic ले जाती है।",
    "ONT यानी Optical Network Terminal, घर पर fiber को Ethernet में बदलता है।",
    "Cell tower phone से radio signal लेता है और fiber से operator को जोड़ता है।",
    "अंत में, Internet शब्द internetwork से आया है, International Network से नहीं।",
]


def _entries(lines: list[str], step: float = 9.0) -> list[dict]:
    return [{"start": round(i * step, 2), "end": round(i * step + step - 1, 2), "text": t}
            for i, t in enumerate(lines)]


def verify(label: str, video_id: str, title: str, transcript: list[dict],
           mode: str, teaching_language: str, out_name: str, smoke: bool = False,
           max_concepts: int | None = None) -> dict:
    request = {
        "videoId": video_id,
        "title": title,
        "transcript": transcript,
        "mode": mode,
        "useLLM": True,
        "allowWhisper": False,
        "teachingLanguage": teaching_language,
    }
    if max_concepts:
        request["maxConcepts"] = max_concepts
    if smoke:
        request["transcript"] = transcript[:6]
    started = time.perf_counter()
    print(f"\n=== {label} | mode={mode} | teachingLanguage={teaching_language} ===", flush=True)
    print(f"provider={llm.provider_name()} embeddings={backend_name()} transcript_units={len(request['transcript'])}", flush=True)
    mm = generate_mindmap(request)
    seconds = time.perf_counter() - started
    path = OUT / out_name
    path.write_text(json.dumps(mm, ensure_ascii=False, indent=2), encoding="utf-8")

    sections = [s for s in mm["root"].get("children", []) if not s.get("recall")]
    concepts = [c for s in sections for c in s.get("children", []) if c.get("type") == "concept"]
    available = [c for c in concepts if c.get("lesson_available") or c.get("available")]
    qa = [c for c in (next((s for s in mm["root"]["children"] if s.get("recall")), {}) or {}).get("children", [])
          if c.get("qa")]
    report = analyze_map(mm)
    meta = mm["meta"]
    result = {
        "file": out_name,
        "videoId": video_id,
        "mode": mode,
        "sourceLanguage": meta.get("sourceLanguage"),
        "teachingLanguage": meta.get("teachingLanguage"),
        "provider": meta.get("llm"),
        "deepStatus": meta.get("deepStatus"),
        "sections": len(sections),
        "concepts": len(concepts),
        "availableLessons": len(available),
        "lessonStats": meta.get("labelStats", {}).get("lessonsAccepted"),
        "rejected": meta.get("labelStats", {}).get("lessonsRejected"),
        "recallQaNodes": len(qa),
        "originalTranscriptUnits": len(meta.get("originalTranscript") or []),
        "generationSeconds": round(seconds, 1),
        "qualityPassed": report.passed,
        "qualityFailures": report.failures,
        "conceptLabels": [c.get("text") for c in concepts],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    (OUT / (out_name + ".summary.json")).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="short transcripts, quick pipeline check")
    ap.add_argument("--only", default="", help="comma-separated subset: networking,http,hindi-dev,hindi-tech")
    args = ap.parse_args()
    smoke = args.smoke
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    results = []
    if not only or "networking" in only:
        results.append(verify(
            "English video (Networking for Hackers #2, real transcript)", "4-Ea5EI6LrM",
            "Networking for Hackers #2 | History of your Internet",
            _transcript_from_export("full-4-Ea5EI6LrM-deep.json"), "deep", "en",
            "networking-deep-verified.json", smoke))
    if not smoke:
        if not only or "http" in only:
            results.append(verify(
                "English video (HTTP Fundamentals, real transcript)", "qv5-5hhsKpE",
                "HTTP Fundamentals in 10 Minutes",
                _transcript_from_export("http-deep-final.json"), "deep", "en",
                "http-deep-verified.json"))
        if not only or "hindi-dev" in only:
            results.append(verify(
                "Hindi video with Hindi speech", "HINDI-DEVA-001", "कंप्यूटर नेटवर्क",
                _entries(HINDI_HINDI), "deep", "hi", "hindi-devanagari-deep-verified.json",
                max_concepts=6))
        if not only or "hindi-tech" in only:
            results.append(verify(
                "Hindi video with English technical terminology", "HINDI-TECH-001", "Networking in Hindi",
                _entries(HINDI_ENGLISH_TECH), "deep", "hi", "hindi-technical-deep-verified.json",
                max_concepts=6))
    print("\n=== ALL RESULTS ===")
    for r in results:
        print(r["file"], "| concepts", r["concepts"], "| lessons", r["availableLessons"],
              "| rejected", r["rejected"], "| quality", "PASS" if r["qualityPassed"] else "FAIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
