"""Strict allowlist of browser extension origins allowed to call the local backend.

An unpacked Manifest V3 extension gets its ID from the folder it is loaded from,
so every Windows user has a different ID. We keep the two known IDs, and let a
user register their own ID without editing Python source:

  * ``CORS_ORIGINS`` in the project-root ``.env`` (comma separated), and
  * ``backend/data/allowed_origins.json`` (written by ``scripts/register_extension.py``).

Only exact ``chrome-extension://<32 letters a-p>`` origins are accepted. Wildcards,
arbitrary schemes and malformed IDs are ignored, so CORS stays strict.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from .config import settings

log = logging.getLogger("mivmap.origins")

# Chrome/Brave extension IDs are 32 characters drawn from the letters a..p.
EXTENSION_ORIGIN = re.compile(r"^chrome-extension://[a-p]{32}$")

# Kept for installs that already use these paths.
LEGACY_EXTENSION_ORIGIN = "chrome-extension://nkakfemcemcfdnafffnpebhljjkkjcgd"
MIVMAP_EXTENSION_ORIGIN = "chrome-extension://kkmhlabikmimffehdedebgllhfffkdkd"
BUILTIN_ORIGINS = (LEGACY_EXTENSION_ORIGIN, MIVMAP_EXTENSION_ORIGIN)

ALLOWLIST_FILE = settings.data_dir / "allowed_origins.json"


def is_valid_origin(origin: object) -> bool:
    """True only for an exact unpacked/packed extension origin, never a wildcard."""
    return isinstance(origin, str) and bool(EXTENSION_ORIGIN.fullmatch(origin.strip()))


def normalize_id(value: str) -> str | None:
    """Accept a bare 32-letter ID or a full chrome-extension:// origin."""
    value = (value or "").strip().rstrip("/")
    if value.startswith("chrome-extension://"):
        value = value[len("chrome-extension://") :]
    value = value.lower()
    origin = f"chrome-extension://{value}"
    return origin if is_valid_origin(origin) else None


def configured_origins() -> list[str]:
    """Origins from .env and the local registration file, strictly validated."""
    values: list[str] = []
    values.extend(settings.cors_origins)
    if ALLOWLIST_FILE.exists():
        try:
            data = json.loads(ALLOWLIST_FILE.read_text(encoding="utf-8"))
            raw = data.get("origins", []) if isinstance(data, dict) else data
            if isinstance(raw, list):
                values.extend(raw)
        except (OSError, ValueError) as exc:
            log.warning("Ignoring unreadable extension allowlist: %s", exc)
    clean: list[str] = []
    for value in values:
        origin = normalize_id(str(value))
        if origin:
            clean.append(origin)
        else:
            log.warning("Ignoring invalid extension origin (wildcards are never allowed): %r", value)
    return clean


def allowed_origins() -> tuple[str, ...]:
    """Complete strict allowlist: built-ins first, then configured origins."""
    out: list[str] = []
    for origin in (*BUILTIN_ORIGINS, *configured_origins()):
        if origin not in out:
            out.append(origin)
    return tuple(out)


# Snapshot used at server startup (CORS middleware needs a fixed iterable).
# The registration helper calls allowed_origins() to re-read the file live.
ALLOWED_ORIGINS = allowed_origins()
