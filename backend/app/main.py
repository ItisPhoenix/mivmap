"""
Mivmap backend — FastAPI application.

Run:  cd backend && python -m app.main      (or: uvicorn app.main:app --port 8765)

Endpoints
---------
GET  /api/health                     capabilities (LLM provider, embeddings, whisper, keyframes)
POST /api/jobs                       Video → Mindmap: returns the complete skeleton map inline (≈0.1–0.5 s)
                                     plus a jobId whose single label call yields `update` ops
GET  /api/jobs/{id}?wait=&hasMap=    long-poll: returns when the skeleton / the label ops are ready
POST /api/prefetch                   warm the cache while the user is still on YouTube (no LLM)
POST /api/chat                       node chatbot (streamed NDJSON, one LLM call per message)
POST /api/transcript                 transcript only (layer 1)
POST /api/refine                     AI refinement -> ops (expand, rewrite, summarize, reorganize, merge)
POST /api/search                     semantic node search
POST /api/study                      flashcards + quiz
POST /api/merge                      merge several mindmaps
POST /api/link                       cross-video knowledge linking (+ Knowledge Hub map)
POST /api/export/notion              push a map to Notion
POST /api/rooms                      create collaboration room (cloud sync)
GET  /api/rooms/{id}                 fetch room snapshot
PUT  /api/rooms/{id}                 replace room map (optimistic version check)
POST /api/rooms/{id}/merge           merge a map into the room
WS   /ws/rooms/{id}                  real-time collaboration channel
"""
from __future__ import annotations

import copy
import html
from contextlib import asynccontextmanager
import json
import logging
import re
import threading
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Response, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from .collab.rooms import manager
from .config import has_module, settings
from .features.chat import chat_stream
from .features.explanations import generate_explanations, recover_explanations
from .features.recall import improve_recall
from .features.merge import link_videos, merge_maps
from .features.notion import push_to_notion
from .features.refine import refine
from .features.search import semantic_search
from .features.study import generate_study_set
from .jobs import Job, cache_get, cache_get_raw, cache_key, cache_put, jobs
from .origins import ALLOWED_ORIGINS
from .pipeline import labels, llm, multimodal, translate
from .pipeline.builder import DEFAULT_MODE, build_skeleton, finalize_concepts
from .pipeline.embeddings import backend_name, embedding_mode
from .pipeline.transcript import TranscriptUnavailable, get_transcript

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("mivmap")

WARMUP_TOPICS = [
    "Vector stores keep document embeddings so similar passages can be retrieved quickly. A vector database adds persistence and access control.",
    "Neural networks learn weights with gradient descent. The learning rate controls how far each weight update moves.",
    "Supply chains depend on reliable shipping routes. When a port closes, freight costs rise and delivery times grow.",
    "Photosynthesis turns sunlight, water and carbon dioxide into glucose. Chlorophyll absorbs the light energy inside the leaf.",
    "Interest rates shape borrowing costs. When the central bank raises rates, mortgage payments and business loans become expensive.",
]
WARMUP_TRANSCRIPT = [
    {"start": float(i * 6), "duration": 6.0, "text": sentence}
    for i, sentence in enumerate(sentence for topic in WARMUP_TOPICS for _ in range(4) for sentence in topic.split(". "))
]


def warm_up() -> None:
    """
    Import/JIT the heavy libraries and open the LLM connection before the first click.
    The NLLB translation model is loaded ONCE here, in its own thread (plus one dummy
    translation); requests that arrive meanwhile wait for that same load, never a second one.
    """
    threading.Thread(target=translate.warm_up, name="mivmap-nllb", daemon=True).start()

    def run() -> None:
        started = time.perf_counter()
        try:
            from .pipeline import embeddings

            embeddings.warmup()
            # a small but realistic lecture, so every stage (TF-IDF, LSA, concept graph,
            # PageRank, communities, LDA, grounding) runs once before the first real click
            for mode in ("academic", "revision"):
                build_skeleton({"videoId": "WARMUPVIDEO", "title": "Warm up", "transcript": WARMUP_TRANSCRIPT, "mode": mode, "useLLM": False, "allowWhisper": False})
        except Exception as exc:  # noqa: BLE001 - warm-up is best effort
            log.warning("warm-up (pipeline) failed: %s", exc)
        llm.warm_up()
        log.info("warm-up done in %.0f ms", (time.perf_counter() - started) * 1000)

    threading.Thread(target=run, name="mivmap-warmup", daemon=True).start()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    warm_up()
    yield


app = FastAPI(title="Mivmap", version="0.2.0-beta", description="YouTube → interactive mindmap pipeline", lifespan=lifespan)
# ALLOWED_ORIGINS is built-ins + user-registered IDs (app/origins.py), so a fresh
# Windows install never needs Python source edits. Wildcards are never accepted.

ALLOWED_HOSTS = {"127.0.0.1:8765", "localhost:8765", "testserver"}


@app.middleware("http")
async def local_security_guard(request, call_next):
    host = (request.headers.get("host") or "").strip().lower()
    if host and host not in ALLOWED_HOSTS:
        return Response("Forbidden host", status_code=403)
    origin = request.headers.get("origin")
    if origin and origin not in ALLOWED_ORIGINS:
        return Response("Forbidden origin", status_code=403)
    return await call_next(request)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=None,
    allow_methods=["GET", "POST", "PUT", "OPTIONS"],
    allow_headers=["Content-Type"],
    allow_credentials=False,
)

VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


@app.get("/player/{video_id}", response_class=HTMLResponse)
def video_player(video_id: str, t: int = 0) -> HTMLResponse:
    """Local HTTP origin for the YouTube embed so YouTube receives a valid Referer.

    Only an 11-character YouTube ID and a non-negative whole-second timestamp are
    accepted; nothing else from the request is reflected into the page.
    """
    if not VIDEO_ID_RE.match(video_id):
        raise HTTPException(400, "Invalid YouTube video id")
    seconds = max(0, int(t or 0))
    vid = html.escape(video_id, quote=True)
    allowed_js = json.dumps(ALLOWED_ORIGINS)
    embed = (
        f"https://www.youtube.com/embed/{vid}?start={seconds}&autoplay=1&controls=1"
        "&autohide=1&showinfo=0&iv_load_policy=3&rel=0&modestbranding=1&cc_load_policy=0"
        "&playsinline=1&fs=1&enablejsapi=1&origin=http://127.0.0.1:8765"
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="referrer" content="strict-origin-when-cross-origin" />
<title>Video preview</title>
<style>
  html,body {{ margin:0; height:100%; background:#000; overflow:hidden; }}
  iframe {{ display:block; width:100%; height:100%; border:0; }}
</style>
</head>
<body>
<iframe id="yt" title="YouTube video preview" src="{html.escape(embed, quote=True)}"
  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
  allowfullscreen referrerpolicy="strict-origin-when-cross-origin"></iframe>
<script>
  // Optional in-page seeking from the extension; the initial start is already in the URL.
  window.addEventListener('message', (event) => {{
    if (!{allowed_js}.includes(event.origin)) return;
    const data = event.data;
    if (!data || data.type !== 'mivmap:seek') return;
    const at = Number(data.seconds);
    if (!Number.isFinite(at) || at < 0) return;
    try {{
      if (window.YT && window.YT.Player && window.player && window.player.seekTo) {{
        window.player.seekTo(at, true);
        window.player.playVideo();
      }} else {{
        // Reload with the exact timestamp if the IFrame API is unavailable.
        document.getElementById('yt').src = {json.dumps(embed.split('?')[0])} + '?start=' + Math.floor(at) + '&autoplay=1&controls=1&autohide=1&showinfo=0&iv_load_policy=3&rel=0&modestbranding=1&cc_load_policy=0&playsinline=1&fs=1';
      }}
    }} catch (e) {{ /* the URL still carries the correct timestamp */ }}
  }});
  // Load the official IFrame API for in-page seek support when available.
  const tag = document.createElement('script');
  tag.src = 'https://www.youtube.com/iframe_api';
  document.head.appendChild(tag);
  window.onYouTubeIframeAPIReady = () => {{
    try {{
      window.player = new YT.Player('yt', {{ events: {{}} }});
    }} catch (e) {{ /* controls in the iframe still work */ }}
  }};
</script>
</body>
</html>"""
    return HTMLResponse(
        document,
        headers={
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": (
                "default-src 'none'; frame-src https://www.youtube.com https://www.youtube-nocookie.com; "
                "script-src https://www.youtube.com 'unsafe-inline'; "
                "style-src 'unsafe-inline'; img-src https: data:; connect-src https://www.youtube.com"
            ),
            "X-Content-Type-Options": "nosniff",
        },
    )



class MindmapRequest(BaseModel):
    videoId: str
    title: str | None = None
    channel: str | None = None
    description: str | None = ""
    duration: float | None = None
    chapters: list[dict] | None = None
    transcript: list[dict] | None = None
    languages: list[str] | None = None
    storyboardSpec: str | None = None
    mode: str = DEFAULT_MODE  # revision (Short) | academic (Standard) | deep (Detailed)
    useLLM: bool = True
    allowWhisper: bool = True
    frames: bool = False  # server-side slide/chart keyframes (needs opencv + yt-dlp)
    noCache: bool = False


class MapBody(BaseModel):
    map: dict[str, Any]


class ChatBody(MapBody):
    nodeId: str
    question: str
    history: list[dict] = Field(default_factory=list)


class RefineBody(MapBody):
    action: str
    nodeIds: list[str]
    instruction: str = ""


class SearchBody(MapBody):
    query: str
    limit: int = 12


class StudyBody(MapBody):
    count: int = Field(12, ge=1, le=40)
    focusIds: list[str] | None = None


class ExplainBody(MapBody):
    conceptIds: list[str] | None = None


class MapsBody(BaseModel):
    maps: list[dict[str, Any]]
    threshold: float | None = None


class RoomPut(MapBody):
    baseVersion: int | None = None


def extract_video_id(value: str) -> str:
    value = value.strip()
    if VIDEO_ID_RE.match(value):
        return value
    m = re.search(r"(?:v=|youtu\.be/|shorts/|embed/|live/)([A-Za-z0-9_-]{11})", value)
    if not m:
        raise HTTPException(400, "Not a YouTube video id or URL")
    return m.group(1)


# ---------------------------------------------------------------------------
@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "version": app.version,
        # Capability labels only: no endpoint, model identifiers or keys in public health.
        "llm": llm.provider_name().split(":", 1)[0],
        "openai_compatible": settings.llm_enabled,
        "llm_local": bool(settings.llm_enabled and settings.llm_is_local),
        "local": llm.local_available(),
        "can_generate": llm.generation_available(),
        "groq": settings.groq_enabled,
        "embeddings": embedding_mode(),
        "translation": "available" if translate.status() not in ("disabled", "unavailable", "loading") else translate.status(),
        "whisper": has_module("faster_whisper") or has_module("whisper"),
        "keyframes": multimodal.available(),
        "spacy": has_module("spacy"),
        "notion": bool(settings.notion_token and settings.notion_parent_page_id),
    }


SKELETON_WAIT = 25.0  # seconds POST /api/jobs waits for the skeleton before handing back a jobId


def _payload(req: MindmapRequest) -> dict:
    payload = req.model_dump()
    payload["videoId"] = extract_video_id(req.videoId)
    return payload


def _pipeline(job: Job, payload: dict, key: str) -> None:
    """Skeleton (cached immediately) → optional single label call → optional keyframes."""
    skeleton = build_skeleton(payload, job.report)
    _cache_map(key, skeleton)
    if job.publish_map(skeleton):
        _label(job, skeleton, key)
    else:
        _log_timing(job, skeleton)
    if payload.get("frames") and multimodal.available():
        try:
            frames = multimodal.extract_keyframes(payload["videoId"], progress=lambda m, p: job.report(m, 0.9 + p * 0.09))
            latest = cache_get(key) or skeleton
            latest["meta"]["keyframes"] = multimodal.attach_frames(latest, frames)
            cache_put(key, latest)
        except Exception as exc:  # noqa: BLE001 - keyframes are a bonus, never fatal
            log.warning("keyframe extraction failed: %s", exc)


def _label(job: Job, skeleton: dict, key: str) -> None:
    job.report("Polishing", 0.9)
    outcome = labels.label_map(skeleton)
    job.ops, job.meta = outcome["ops"], {**outcome["meta"], "labelStats": outcome["stats"]}
    started = time.perf_counter()
    if outcome["ops"] or outcome["meta"].get("labelled"):
        final = labels.apply_outcome(copy.deepcopy(skeleton), outcome)
        finalize_concepts(final)
        improve_recall(final)
        improve_recall(job.map)
        _cache_map(key, final)
    _log_timing(job, skeleton, outcome["stats"], (time.perf_counter() - started) * 1000)


def _coverage(mindmap: dict) -> float:
    """Share of the video the map was built from (< 1: a quick map from a partly translated transcript)."""
    return float(((mindmap.get("meta") or {}).get("translation") or {}).get("coverage", 1.0))


def _cache_map(key: str, mindmap: dict) -> None:
    """Cache a map unless a more complete one (full translation) is already there."""
    if _coverage(mindmap) < 1:
        current = cache_get(key)
        if current is not None and _coverage(current) > _coverage(mindmap):
            return
    cache_put(key, mindmap)


def _log_timing(job: Job, skeleton: dict, stats: dict | None = None, cache_ms: float = 0.0) -> None:
    """One line per map: where the time went between the request and the fully labelled map."""
    t = skeleton.get("meta", {}).get("timings") or {}
    stats = stats or {}
    log.info(
        "timing %s lang=%s detect=%dms translate=%dms skeleton=%dms summary=%dms llm=%dms apply=%dms total=%dms (llm_calls=%d%s)",
        skeleton["meta"].get("videoId"), skeleton["meta"].get("sourceLanguage", "en"), t.get("detect", 0), t.get("translate", 0), t.get("skeleton", 0),
        t.get("summary", 0), stats.get("llm_ms", 0), stats.get("apply_ms", 0) + cache_ms, (time.time() - job.created) * 1000,
        stats.get("calls", 0), f", errors={stats['errors']}" if stats.get("errors") else "",
    )


def _label_job(key: str, skeleton: dict) -> Job:
    running = jobs.running(key, labels_only=True)
    if running:
        return running
    job = jobs.submit(lambda j: _label(j, skeleton, key), key=f"labels:{key}", labels=True)
    job.publish_map(skeleton)
    return job


@app.post("/api/jobs")
def start_job(req: MindmapRequest) -> Response:
    started = time.perf_counter()
    payload = _payload(req)
    key = cache_key(payload)
    want_llm = req.useLLM and llm.fast_available()

    raw = None if req.noCache else cache_get_raw(key)
    cached = json.loads(raw) if raw is not None else None
    if cached is not None and _coverage(cached) < 1 and translate.is_complete(payload["videoId"], cached["meta"].get("sourceLanguage", "")):
        raw = cached = None  # a quick map whose full translation has finished since: rebuild it in full (~0.5 s)
    if cached is not None:
        job = _label_job(key, cached) if want_llm and not cached["meta"].get("labelled") else None
        # the cached JSON text is spliced in as-is: no re-encoding on the hot path
        body = f'{{"jobId": {json.dumps(job.id if job else None)}, "cached": true, "pending": {json.dumps(bool(job))}, "map": {raw}}}'
        log.info("POST /api/jobs %s cache hit in %.1f ms (labels %s)", payload["videoId"], (time.perf_counter() - started) * 1000, "pending" if job else "done/off")
        return Response(body, media_type="application/json")

    pipe = None if req.noCache else jobs.running(key)
    if pipe is not None and pipe.map is None and not pipe.labels and translate.in_progress(payload["videoId"]):
        pipe = None  # the prefetch is still translating the whole video: build a quick map from what is ready
    if pipe is None:
        # a click does not wait for the whole translation: 1/8 of the video, spread evenly, is enough to start
        click = {**payload, "minCoverage": translate.CLICK_COVERAGE}
        pipe = jobs.submit(lambda j: _pipeline(j, click, key), key=key, labels=want_llm)
    elif want_llm:
        pipe.request_labels()  # attach to a running prefetch
    pipe.map_ready.wait(SKELETON_WAIT)
    if pipe.status == "error":
        raise HTTPException(422, pipe.error or "pipeline failed")
    if pipe.map is None:  # still transcribing (e.g. Whisper): the client long-polls this job
        return _json({"jobId": pipe.id, "cached": False, "pending": True, "map": None})

    job = pipe if pipe.labels else (_label_job(key, pipe.map) if want_llm and not pipe.map["meta"].get("labelled") else None)
    log.info("POST /api/jobs %s skeleton in %.0f ms (labels %s)", payload["videoId"], (time.perf_counter() - started) * 1000, "pending" if job else "off")
    return _json({"jobId": job.id if job else None, "cached": False, "pending": bool(job), "map": pipe.map})


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str, wait: float = 0, hasMap: bool = False) -> Response:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    wait = max(0.0, min(wait, 25.0))
    if wait:
        (job.finished if hasMap else job.map_ready).wait(wait)
    return _json(job.to_dict(include_map=not hasMap))


@app.post("/api/prefetch")
def prefetch(req: MindmapRequest) -> dict:
    """Build + cache the skeleton while the user is still watching. Never calls the LLM."""
    payload = {**_payload(req), "useLLM": False, "allowWhisper": False, "frames": False}
    key = cache_key(payload)
    if cache_get_raw(key) is not None:
        return {"status": "cached"}
    if jobs.running(key):
        return {"status": "running"}
    jobs.submit(lambda j: _pipeline(j, payload, key), key=key, labels=False)
    return {"status": "started"}


@app.post("/api/chat")
def chat(body: ChatBody) -> StreamingResponse:
    try:
        stream = chat_stream(body.map, body.nodeId, body.question, body.history)
        first = next(stream)  # surfaces "node not found" as a 400 before streaming starts
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    def events():
        yield first
        yield from stream

    return StreamingResponse(events(), media_type="application/x-ndjson", headers={"Cache-Control": "no-store"})


def _json(obj: dict) -> Response:
    return Response(json.dumps(obj, ensure_ascii=False), media_type="application/json")


@app.post("/api/transcript")
def transcript(req: MindmapRequest) -> dict:
    try:
        t = get_transcript(extract_video_id(req.videoId), req.transcript, req.languages, req.allowWhisper)
    except TranscriptUnavailable as exc:
        raise HTTPException(404, str(exc)) from exc
    return t.to_dict()


@app.post("/api/refine")
def refine_nodes(body: RefineBody) -> dict:
    try:
        return refine(body.map, body.action, body.nodeIds, body.instruction)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/search")
def search(body: SearchBody) -> dict:
    return {"results": semantic_search(body.map, body.query, body.limit)}


@app.post("/api/study")
def study(body: StudyBody) -> dict:
    return generate_study_set(body.map, body.count, body.focusIds)


@app.post("/api/explanations/recover")
def recover_explanations_endpoint(body: ExplainBody) -> dict:
    """Attach previously validated cached lessons to concepts that lack one."""
    report = recover_explanations(body.map, body.conceptIds)
    return {"map": body.map, "report": report}


@app.post("/api/explanations/generate")
def generate_explanations_endpoint(body: ExplainBody) -> dict:
    """Generate a new validated lesson for a concept when an LLM is configured."""
    report = generate_explanations(body.map, body.conceptIds)
    return {"map": body.map, "report": report}


@app.post("/api/merge")
def merge(body: MapsBody) -> dict:
    try:
        return {"map": merge_maps(body.maps, body.threshold or 0.78)}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/link")
def link(body: MapsBody) -> dict:
    return link_videos(body.maps, body.threshold or 0.72)


@app.post("/api/export/notion")
def export_notion(body: MapBody) -> dict:
    try:
        return push_to_notion(body.map)
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc


# -- collaboration / cloud sync ------------------------------------------------------
@app.post("/api/rooms")
def create_room(body: MapBody) -> dict:
    room = manager.create(body.map)
    return {"roomId": room.id, "version": room.version}


@app.get("/api/rooms/{room_id}")
def get_room(room_id: str) -> dict:
    room = manager.get(room_id)
    if not room:
        raise HTTPException(404, "room not found")
    return {"roomId": room.id, "version": room.version, "map": room.map, "users": room.presence(), "leaderboard": room.leaderboard()}


@app.put("/api/rooms/{room_id}")
async def put_room(room_id: str, body: RoomPut) -> dict:
    room = manager.get(room_id)
    if not room:
        raise HTTPException(404, "room not found")
    return await manager.replace(room, body.map, body.baseVersion)


@app.post("/api/rooms/{room_id}/merge")
async def merge_room(room_id: str, body: MapBody) -> dict:
    room = manager.get(room_id)
    if not room:
        raise HTTPException(404, "room not found")
    return await manager.merge_into(room, body.map)


@app.websocket("/ws/rooms/{room_id}")
async def room_socket(ws: WebSocket, room_id: str) -> None:
    host = (ws.headers.get("host") or "").strip().lower()
    if host and host not in ALLOWED_HOSTS:
        await ws.close(code=1008)
        return
    origin = ws.headers.get("origin")
    if origin and origin not in ALLOWED_ORIGINS:
        await ws.close(code=1008)
        return
    await manager.session(ws, room_id)


def run() -> None:
    import uvicorn

    log.info("Mivmap backend on http://%s:%s  (LLM: %s, embeddings: %s)", settings.host, settings.port, llm.provider_name(), backend_name())
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    run()
