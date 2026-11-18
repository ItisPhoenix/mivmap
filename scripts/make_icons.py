"""Generate the Mivmap icons (PNG, zero dependencies).

The mark is a flat, geometric "M" -- a folded map path with a single warm
node at the centre junction. Three solid colours, no gradients, no shadows,
no borrowed logo. Run from the project root:

    python scripts/make_icons.py
"""
from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "extension" / "icons"

NAVY = (23, 38, 58)      # solid background
CREAM = (244, 239, 230)  # the folded-map path
AMBER = (240, 166, 59)   # one highlight node at the junction

# Normalised "M" polyline: left stem -> peak -> valley -> peak -> right stem.
PATH = [(0.20, 0.74), (0.29, 0.29), (0.50, 0.56), (0.71, 0.29), (0.80, 0.74)]


def dist_to_segment(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy or 1)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def render(size: int) -> bytes:
    ss = 4  # supersampling for clean edges at 16px
    n = size * ss
    radius = n * 0.22
    cx = cy = n / 2
    stroke = max(1.0, n * 0.072)
    rows = []
    for y in range(size):
        row = bytearray([0])
        for x in range(size):
            acc = [0, 0, 0, 0]
            for sy in range(ss):
                for sx in range(ss):
                    px, py = x * ss + sx + 0.5, y * ss + sy + 0.5
                    # rounded-square mask
                    qx = max(abs(px - cx) - (n / 2 - radius), 0)
                    qy = max(abs(py - cy) - (n / 2 - radius), 0)
                    if math.hypot(qx, qy) > radius:
                        continue
                    color = NAVY
                    for (ax, ay), (bx, by) in zip(PATH, PATH[1:]):
                        if dist_to_segment(px, py, ax * n, ay * n, bx * n, by * n) < stroke / 2:
                            color = CREAM
                    # single amber node at the central junction of the fold
                    jx, jy = PATH[2][0] * n, PATH[2][1] * n
                    if math.hypot(px - jx, py - jy) < stroke * 0.78:
                        color = AMBER
                    for i, c in enumerate(color):
                        acc[i] += c
                    acc[3] += 255
            samples = ss * ss
            covered = acc[3] / 255
            if covered:
                row.extend(int(acc[i] / covered) for i in range(3))
            else:
                row.extend((0, 0, 0))
            row.append(int(acc[3] / samples))
        rows.append(bytes(row))
    raw = b"".join(rows)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for size in (16, 32, 48, 128):
        (OUT / f"icon{size}.png").write_bytes(render(size))
        print("wrote", f"icon{size}.png")


if __name__ == "__main__":
    main()
