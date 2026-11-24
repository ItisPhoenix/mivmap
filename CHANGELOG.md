# Changelog

All notable changes to this project are documented here. GitHub releases are source-only; no binary assets are attached.

## [v0.2.0-beta] - 2026-11-24

Second beta focused on making the video-to-learning workflow clearer, more reliable, and more efficient.

### Added and improved

- Dedicated, readable concept explanations with validated generation, honest unavailable states, and retry/recovery behavior.
- Quick Recall prompts and distinct study sets, with quiz history and learning progress preserved when switching sets.
- A clearer study inspector with outline and recall views, section navigation, Focus Section and Full Map controls, and responsive drawer behavior.
- Shorter mind-map labels while preserving the full underlying concept content.
- More efficient concept selection and long-span embedding handling, with explicit embedding-provider configuration.
- Improved LLM provider compatibility, lesson quality repair, and cache/retry handling.
- Expanded backend and extension regression coverage for explanations, layout, study sets, provider behavior, and long-span processing.

### Release verification

- Backend: **317 passed, 2 skipped**.
- Extension: **53 passed**.
- Python compilation, JavaScript syntax checks (37 files), staged diff checks, and secret-pattern scan passed.
- Automated release validation was performed locally on Windows.

### Known limitations carried forward

- Acceptance testing covered Brave; Chrome and Edge remain unverified.
- A complete fresh Deep Mode export may still depend on provider availability, quotas, and response quality. Failed or incomplete lessons are not represented as successful lessons.
- Collaboration rooms have no authentication; treat room codes as share links.
- NLLB translation is optional, slow on CPU, and requires a one-time model conversion.
- Unpacked extension IDs change with the install path and must be registered once with the backend.

## [v0.1.0-beta] - 2026-11-18

First public beta of Mivmap, a local-first YouTube-to-mind-map extension with structured Deep Mode lessons. Mivmap is a rebrand and continuation of the MIT-licensed TubeMind project (see [LICENSE](LICENSE)).

### Included

- Manifest V3 extension for Brave/Chrome-style browsers and a localhost-only FastAPI backend.
- Video-to-mind-map generation from real YouTube transcripts, with interactive concepts, timestamp navigation and a floating player.
- Deep Mode lessons (definition, why it matters, steps, worked example, diagram, misconception, exercise, knowledge check) with strict validation and fail-closed exports.
- Per-concept lesson checkpoint cache: validated lessons are reused; only missing or rejected lessons retry with bounded backoff and `Retry-After` support.
- Strict extension-origin allowlist with local registration (`scripts/register_extension.py`) for path-derived unpacked IDs; no wildcard CORS.
- JSON/Markdown/Obsidian/Roam/OPML/PNG/SVG exports and legacy `tubemind/1` import compatibility.
- Six regenerated screenshots of the current UI and an installation/troubleshooting guide.

### Fixed in this release

- Windows cp1252 console crash in `scripts/register_extension.py --scan` (non-ASCII output).
- Windows cp1252 console crash in the documented CLI (`python -m app.cli`) output.

### Known limitations (confirmed, not speculative)

- Acceptance testing covered **Brave only**; Chrome and Edge remain unverified.
- A complete fresh Deep Mode export could not be verified with the available free provider tier (quota exhaustion, empty responses). The pipeline fails closed and reports the affected concepts; BYOK/paid providers are required for dependable full coverage.
- Collaboration rooms have no authentication; treat room codes as share links.
- NLLB translation is optional, slow on CPU, and requires a one-time model conversion.
- Embedded YouTube playback can be paused by browser visibility/autoplay policies; an "Open on YouTube" fallback is provided.
- Unpacked extension IDs change with the install path and must be registered once with the backend.

### Roadmap (limited to the confirmed items above)

1. Verify Chrome and Edge with the same acceptance checklist used for Brave.
2. Document and exercise BYOK/paid provider setups for complete fresh Deep Mode generation.
3. Add authentication to collaboration rooms before any non-local deployment.
4. Continue improving the embedded-player fallback guidance for restricted playback environments.
