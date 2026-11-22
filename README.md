# Mivmap — Open-source YouTube Mind Map & AI Learning Extension

**Mivmap** is an open-source YouTube study tool and AI video learning extension: turn any YouTube video into an interactive, timestamp-linked mind map, then learn every concept with a structured Deep Mode lesson. It is local-first — a Brave/Chrome Manifest V3 extension plus a small Python backend — so maps, notes and progress stay in your browser and generation runs on your own machine.

**Release target:** `v0.1.0-beta` (source-only; no binary is distributed by this project).

> **Upstream credit:** Mivmap is a rebrand and continuation of the original MIT-licensed [TubeMind project](https://github.com/diyabhattacharjee-git/youtube-extension). Original work © TubeMind contributors; rebrand and subsequent changes © Mivmap contributors. See [LICENSE](LICENSE).

Repository: [`ItisPhoenix/mivmap`](https://github.com/ItisPhoenix/mivmap).

Suggested GitHub topics: `youtube` · `mindmap` · `mind-map` · `chrome-extension` · `brave-browser` · `ai-learning` · `study-tools` · `llm` · `openai` · `fastapi` · `transcript` · `education` · `local-first` · `self-hosted` · `python` · `javascript` · `svg`

---

## From YouTube video to mind map

- **Video → mind map.** The transcript is read from the video itself, split into sections, and turned into `Term: meaning` concepts with supporting detail lines. No LLM is required for the base map — it is built from the video's own words.
- **Interactive concepts.** Expand/collapse nodes, edit text inline, add notes and links, cross-link concepts, search, follow the video, study with flashcards and quizzes, and export to Markdown, Obsidian, Roam, OPML, PNG, SVG or JSON.
- **Timestamped video navigation.** Every node carries a real transcript timestamp. Clicking it opens one reusable floating player that seeks to that moment; a local player page gives YouTube a valid `Referer` so embeds work (YouTube Error 153 is avoided). You can also follow along as the video plays or jump in the original tab.
- **Deep Mode lessons.** The **Detailed** map mode generates a real lesson per concept: definition, why it matters, numbered steps, a worked example, an ASCII diagram, a common misconception, an exercise and a knowledge check. Examples are labelled as supplemental, never presented as claims from the video. Lessons are validated, cached per concept, and only missing or rejected lessons are retried. Concepts without a validated lesson are excluded from the export — no placeholders, no fabricated explanations.

## Contents

- [Install](#install)
- [Screenshots](#screenshots)
- [Using Mivmap](#using-mivmap)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Compatibility & limitations](#compatibility--limitations)
- [Development & tests](#development--tests)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Install

### 1. Backend (Python 3.10+)

```bash
cd Mivmap/backend
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m app.main                       # http://127.0.0.1:8765
```

Create a private `.env` at the **project root** (`Mivmap/.env`) by copying [`.env.example`](.env.example). The base map works with no keys at all. For AI-polished labels and Deep Mode lessons, set any OpenAI-compatible provider:

```env
LLM_API_KEY=your_key_here
LLM_MODEL=your-model-name
LLM_BASE_URL=https://your-provider.example/v1
```

This works with OpenAI, NVIDIA NIM, Kilo gateway, Together, a local vLLM/Ollama server, and similar providers. The key is read only by the backend and never reaches the browser, logs or git. `GET /api/health` reports the active provider.

Optional heavier extras: `pip install -r requirements-ml.txt`, then `python -m spacy download en_core_web_sm`. Translation of non-English videos needs a one-time NLLB-200 conversion (`python scripts/convert_nllb.py`); English videos skip it.

#### Embeddings

Embeddings are selected only from your private `.env`; no host, provider or model is compiled in.

1. **Remote** - set `EMBEDDING_BASE_URL` and `EMBEDDING_MODEL` (plus `EMBEDDING_API_KEY` if your endpoint requires it). `EMBEDDING_BASE_URL` is the service root; a trailing `/v1` is also accepted and the embedding path is appended exactly once. `EMBEDDING_TIMEOUT` is the HTTPX per-phase timeout in seconds: the same value applies to connection setup, reading the response, writing the request and waiting for a pooled connection. It is not a total deadline for the entire request. The configured endpoint is used exclusively: authentication failures and connection/DNS/timeout/5xx failures abort immediately; only an incompatible route or request format makes the client try the next supported wire format. Errors are explicit and never switch providers. See `.env.example` for the full variable list.
2. **Local** - with no endpoint configured, set `EMBEDDING_LOCAL_MODEL` to an already installed model id/path (absolute or relative, including Windows drive paths), or leave it empty to reuse a single cached compatible model. Automatic discovery inspects `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, `HF_HOME/hub` and the default `~/.cache/huggingface/hub`; it considers a cached model only when a snapshot contains a sentence-transformers marker file (`modules.json`, `config_sentence_transformers.json` or `sentence_bert_config.json`), so unrelated cached models are ignored. Any organisation namespace is supported. If more than one compatible model is found, none is selected automatically; set `EMBEDDING_LOCAL_MODEL` explicitly. Weights are never downloaded automatically.
3. **Fallback** - if no local model is usable, the backend uses its built-in TF-IDF/LSA vectors, fitted once from the original corpus and reused for every call. Once vectors have been produced, a late local failure raises an explicit error instead of switching vector spaces.

Vectors from stateless backends are cached on disk by endpoint/model identity using the exact submitted text; a per-identity dimension record drops stale entries. The corpus-dependent fallback is never cached. `GET /api/health` reports only the embedding capability (`remote`, `local` or `fallback`) and never an endpoint, model identifier or key.

#### Tests (offline by default)

The default backend suite needs no `.env`, credentials, cached model weights or network access. It installs a deterministic TF-IDF/LSA provider (one fixed space shared by every encoder) and blocks outbound HTTP. `useLLM=False` does not disable embeddings; it only skips the LLM label call.

Genuine provider tests are opt-in and skipped otherwise:

```bash
# real remote endpoint/model from your private .env (nothing is printed)
MIVMAP_TEST_REMOTE_EMBEDDINGS=1 python -m pytest tests/test_embedding_integration.py -q
# real local inference, weights must already be cached
MIVMAP_TEST_LOCAL_MODEL=<cached-model-id-or-path> python -m pytest tests/test_embedding_integration.py -q
```

### 2. Extension (Brave tested; Chrome/Edge should work with Manifest V3)

1. Open `chrome://extensions` (or `brave://extensions`) and enable **Developer mode**.
2. Click **Load unpacked** and select the [`extension/`](extension) folder.
3. Pin **Mivmap**. The popup dot turns green when the backend is reachable. The bundled demo map works even with the backend offline.

Acceptance testing for this release was performed on **Brave**. Chrome and Edge are expected to work (the extension is Manifest V3) but were **not verified** in this release.

> **Extension ID note.** An unpacked extension's ID is derived from its folder path, so a fresh Windows install usually has a new ID. You do **not** need to edit Python source: copy the ID from `brave://extensions` → Mivmap → Details, then run `python scripts/register_extension.py <that-id>` (or `python scripts/register_extension.py --scan` to find it in local browser profiles) and restart the backend. Alternatively set `CORS_ORIGINS` in `.env`. The backend still accepts only exact `chrome-extension://` origins — wildcards are rejected.

### 3. Generate

Open a YouTube video and click the **Mivmap** button under the player, press **Alt+M**, right-click the page, or use the toolbar popup (where you can paste any link). A viewer tab opens, shows pipeline progress, and draws the map. Choose **Detailed** for Deep Mode lessons.

Command line, without the extension:

```bash
cd backend
python -m app.cli "https://www.youtube.com/watch?v=<ID>" --mode deep --title "Lecture" --out ../examples/my-map.json
```

---

## Screenshots

The images in [`docs/screenshots/`](docs/screenshots) were captured from the rebranded Mivmap viewer: `blackboard-sketchbook.png`, `chalk-radial.png`, `doodle-theme.png`, `sketch-theme.png`, `inspector.png` (a real Deep Mode lesson) and `study-mode.png`. To regenerate them after a UI change:

1. Load a map in the viewer (☰ → Library → Import, or drag a JSON file onto the canvas).
2. Pick a theme/layout from the ☰ menu; open the Content map or the node inspector as needed.
3. Capture the `viewer.html` tab with Chrome/Brave DevTools (⚙ → Capture screenshot), or use **Export → PNG image** for a clean map image.
4. Save to `docs/screenshots/` and update the captions here.

> Screenshots are illustrative UI examples. The `inspector.png` image shows a real, validated Deep Mode lesson from the bundled HTTP export; the others show map navigation and themes.

---

## Using Mivmap

| Action | How |
|---|---|
| Jump to a moment | Click an underlined **1:23** timestamp (sections show ranges), press **p**, or use the floating player |
| Follow along | The matching node gets an outline as the video plays |
| Content map | **Content map** button: Quick recall, Jump to, then numbered sections with summaries and concepts |
| Concepts / Details | **Concepts · Details** buttons or keys **1 / 2**; Detailed maps start fully unfolded |
| Edit | Double-click, **F2** or **e**; **Tab** child, **Enter** sibling, **Delete** remove |
| Navigate | Arrows, wheel/pinch zoom, drag to pan, **f** to fit |
| Deep Mode lesson | Select a concept: Definition, Why it matters, Steps, Worked example, Diagram, Misconception, Exercise, Knowledge check |
| Search | `/` or Ctrl+F; **Enter** runs semantic search |
| Study | **🎓 Study**: Leitner flashcards plus a quiz |
| Voice | **🎙**, then e.g. "expand entanglement", "go to backpropagation", "next", "zoom in" |
| Collaborate | **👥 Share** creates a room code; presence dots show who is where |
| Export | **⤓ Export**: PNG · SVG · JSON · Markdown · Obsidian · Roam JSON · OPML · Notion |
| Import | Drag a `.mivmap.json` or legacy `.tubemind.json` file onto the canvas |

Maps are saved automatically to IndexedDB. Everything is local unless you point the extension at your own remote backend.

---

## Architecture

```text
Chrome / Brave extension (Manifest V3, vanilla JS, no build step)
  content/youtube.js   transcript + chapters + storyboard + seek
  background/sw.js     routing, context menu, Alt+M, prefetch
  viewer/              model · SVG renderer · features · floating player

Python backend (FastAPI, 127.0.0.1:8765)
  transcript -> optional NLLB translation -> segmentation -> concepts
             -> LLM labels / validated Deep lessons -> builder -> quality gates
  FastAPI routes: /api/jobs, /api/health, /api/chat, /api/refine, /api/study,
                  /api/rooms, /ws/rooms/{id}, /player/{video_id}

Local storage
  IndexedDB (maps, progress) · chrome.storage (settings) · backend/data (cache, rooms)
```

The whole map is built deterministically from the transcript first, then the LLM is used only for text polish and Deep Mode lessons. Per-section label caches and per-concept lesson caches make repeat runs cheap and resumable; generation retries only missing or rejected lessons with bounded exponential backoff and `Retry-After` support. The renderer is a dependency-free hand-drawn SVG engine (Balanced, Tree and Radial layouts). Collaboration uses WebSocket rooms with SQLite persistence.

See [`backend/app/pipeline/`](backend/app/pipeline) and [`extension/viewer/js/`](extension/viewer/js) for the implementation.

---

## Configuration

`.env` at the project root (copy [`.env.example`](.env.example)):

| Variable | Default | Purpose |
|---|---|---|
| `LLM_API_KEY`, `LLM_MODEL`, `LLM_BASE_URL` | empty | Primary OpenAI-compatible provider; empty = heuristic map, no Deep lessons |
| `GROQ_API_KEY`, `GROQ_MODEL`, `GROQ_FALLBACK_MODEL` | empty | Optional Groq fallback for refine/chat |
| `HF_TOKEN`, `HF_MODEL`, `HF_LOCAL_MODEL` | empty / `Qwen/Qwen2.5-7B-Instruct` | Hugging Face or local fallback |
| `LABEL_TIMEOUT`, `CHAT_TIMEOUT` | `1.2`, `12.0` | Hard deadlines in seconds |
| `NLLB_ENABLED`, `NLLB_MODEL_PATH` | `1`, converted path | Translate non-English transcripts to English once |
| `HOST`, `PORT` | `127.0.0.1`, `8765` | Loopback binding |
| `NOTION_TOKEN`, `NOTION_PARENT_PAGE_ID` | empty | Enables push to Notion |

The extension's ⚙ Settings page controls the backend URL, map size (**Short** / **Standard** / **Detailed = Deep**), theme, layout, AI polish, prefetch, speech-to-text, keyframes, follow-along and developer mode.

---

## Compatibility & limitations

- **Legacy identifiers are preserved on purpose.** Exported and saved maps keep `schema: "tubemind/1"`, the IndexedDB database name `tubemind`, and the collaboration file `backend/data/tubemind.db`. Legacy `.tubemind.json` files import normally. No migration is needed and no saved data is lost.
- **Unpacked extension IDs depend on the folder path.** Register a new ID with `scripts/register_extension.py` (see Install) and restart the backend. The backend binds to `127.0.0.1` and rejects unknown hosts/origins; it never uses wildcard CORS.
- **Transcript scraping depends on YouTube internals.** If YouTube changes them, the backend's transcript API → Whisper chain takes over. Cloud/datacenter IPs are often blocked by YouTube; run the backend locally.
- **Free LLM tiers are rate limited.** Deep lessons retry only missing concepts with backoff; a 429 or missing key never produces placeholder explanations. A cached/validated lesson is reused, not regenerated.
- **CPU translation is slow.** NLLB-200 takes roughly 1–2 minutes for a 50-minute lecture on a small laptop; English videos skip it.
- **Embedded playback can be blocked.** Browser autoplay policies, headless/CDP sessions, or network conditions may keep the YouTube iframe at `readyState 0` even when the embed loads correctly. The floating player also offers a direct "Watch on YouTube" link.
- **Collaboration rooms have no authentication.** Treat a room code like a share link; add auth before exposing the backend publicly.
- **Deep Mode needs a working, configured LLM provider** for real lessons; with no key the map still builds from the transcript but Deep lessons are omitted rather than faked. Free-tier quotas, timeouts and empty responses cause the affected concepts to be reported as failed and excluded — a complete fresh Deep export was **not verified** in this release with the available free provider, so use BYOK/paid access if you depend on full Deep Mode coverage.

**Privacy:** maps, notes and progress stay in your browser. Video data goes only to your own backend and, if configured, your chosen LLM provider.

---

## Development & tests

```bash
cd backend && .venv\Scripts\python.exe -m pytest -q     # 46 tests: pipeline, labels, origins/CORS, quality, exports
node --test extension/tests/*.test.js                    # 11 extension viewer tests
cd scripts && python make_icons.py                       # regenerate the Mivmap icons
```

The extension has no build step: reload it from `chrome://extensions` after editing. Keep `extension/viewer/js/mindmap/model.js` and `backend/app/collab/ops.py` in sync when adding operation types. Do not commit `.env`; only `.env.example` is publishable.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Popup dot stays grey / "Mivmap is not connected" | Start the backend from `backend/` (`python -m app.main`) and check `http://127.0.0.1:8765/api/health`. A non-loopback host or an unregistered extension origin is rejected by design. |
| Connection works from a terminal but not the extension | The unpacked extension ID changed with the folder path. Copy it from `brave://extensions` → Mivmap → Details and run `python scripts/register_extension.py <id>`, then **restart the backend**. `--scan` finds IDs for UI-loaded extensions; command-line-loaded ones are not persisted in browser profiles, so pass the ID manually. |
| Backend logs `403 Forbidden origin` | Only exact `chrome-extension://` IDs are accepted. Wildcards are never allowed; register the specific ID. |
| Deep Mode produces no lessons | A configured OpenAI-compatible provider is required. With no key the map still builds from the transcript but Deep lessons are omitted, never faked. |
| `HTTP 429` / "Free model usage limit reached" | Free-tier quota exhausted. Retry later, use BYOK/paid tier, or change `LLM_MODEL` in `.env`. Validated lessons already cached are reused; only missing lessons are retried. |
| "empty LLM response" / "unbalanced JSON" | Provider returned unusable output. The affected concepts are reported as failed and excluded from the export; no placeholder explanations are substituted. |
| Player iframe is black or paused | Confirm the backend serves `/player/{id}?t=...`, allow the YouTube iframe, and check whether the tab is hidden (background tabs pause media). The "Open on YouTube" link always works as a fallback. |
| CLI prints a Unicode error on Windows | Use `python -m app.cli` from `backend/`; CLI output is ASCII-safe. |

## Contributing

1. Fork the planned repository and create a focused branch.
2. Make the smallest change that solves the problem; preserve the legacy schema and storage identifiers.
3. Run both test commands above and add a test for behaviour changes.
4. Open a pull request describing what changed and which tests you ran.

Good first contributions: new themes in `themes.js`, more export targets, better segmentation, or sync adapters implementing the same `op`/`snapshot` protocol as `collab.js`.

## License

[MIT](LICENSE) — original work © TubeMind contributors; rebrand and modifications © Mivmap contributors.
