"""Register an unpacked Mivmap extension ID with the local backend.

No Python source edits and no wildcard CORS: this writes one exact
``chrome-extension://<id>`` origin to ``backend/data/allowed_origins.json``
(gitignored). Restart the backend after registering.

Usage:
    python scripts/register_extension.py <extension-id-or-origin>
    python scripts/register_extension.py --scan          # find it in local browsers
    python scripts/register_extension.py --list          # show registered origins

Find the ID in Brave/Chrome at ``brave://extensions`` -> Mivmap -> Details.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from app.origins import ALLOWLIST_FILE, allowed_origins, normalize_id  # noqa: E402

BROWSER_PROFILE_ROOTS = [
    Path(os.environ.get("LOCALAPPDATA", "")) / "Google" / "Chrome" / "User Data",
    Path(os.environ.get("LOCALAPPDATA", "")) / "BraveSoftware" / "Brave-Browser" / "User Data",
    Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "Edge" / "User Data",
]


def scan_browsers() -> list[str]:
    """Best-effort: find extension IDs loaded from this project's extension folder."""
    target = str(ROOT / "extension").lower()
    found: set[str] = set()
    for base in BROWSER_PROFILE_ROOTS:
        if not base.exists():
            continue
        for prefs in list(base.glob("Default/Preferences")) + list(base.glob("Profile */Preferences")):
            try:
                data = json.loads(prefs.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            settings = (data.get("extensions") or {}).get("settings") or {}
            for ext_id, info in settings.items():
                path = str(info.get("path", "")).lower()
                if path and (path == target or path.rstrip("\\/") == target.rstrip("\\/")):
                    origin = normalize_id(ext_id)
                    if origin:
                        found.add(origin)
    return sorted(found)


def load_file() -> list[str]:
    if not ALLOWLIST_FILE.exists():
        return []
    try:
        data = json.loads(ALLOWLIST_FILE.read_text(encoding="utf-8"))
        raw = data.get("origins", []) if isinstance(data, dict) else data
        return [x for x in raw if isinstance(x, str)] if isinstance(raw, list) else []
    except (OSError, ValueError):
        return []


def save_file(origins: list[str]) -> None:
    ALLOWLIST_FILE.parent.mkdir(parents=True, exist_ok=True)
    ALLOWLIST_FILE.write_text(
        json.dumps({"origins": sorted(set(origins))}, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("extension_id", nargs="?", help="32-letter extension ID or chrome-extension:// origin")
    ap.add_argument("--scan", action="store_true", help="find the ID in local browser profiles")
    ap.add_argument("--list", action="store_true", help="list registered origins and exit")
    args = ap.parse_args()

    current = load_file()
    if args.list:
        print("Allowed extension origins:")
        for origin in allowed_origins():
            print("  " + origin)
        return

    origins: list[str] = []
    if args.scan:
        origins = scan_browsers()
        if not origins:
            print("No unpacked Mivmap extension found in local browser profiles.\n"
                  "Load it with chrome://extensions -> Load unpacked, or pass the ID manually.")
            return
    elif args.extension_id:
        origin = normalize_id(args.extension_id)
        if not origin:
            ap.error("invalid extension ID (expected 32 letters a-p, or a chrome-extension:// origin)")
        origins = [origin]
    else:
        ap.error("pass an extension ID, --scan, or --list")

    added = [o for o in origins if o not in current]
    save_file([*current, *origins])
    print("Registered:", ", ".join(origins))
    print("Allowed origins now:")
    for origin in allowed_origins():
        print("  " + origin)
    if added:
        print("Restart the backend so the new origin takes effect.")


if __name__ == "__main__":
    main()
