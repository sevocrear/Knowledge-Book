#!/usr/bin/env python3
"""Convert MP4 to GIF (stride / fps cap for smaller files). Used after Manim render."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio


def main() -> None:
    p = argparse.ArgumentParser(description="MP4 → GIF via imageio (for README / Telegram previews).")
    p.add_argument("input", type=Path, help="Source .mp4 path")
    p.add_argument("-o", "--output", type=Path, help="Output .gif (default: same stem as input)")
    p.add_argument("--max-fps", type=float, default=15.0, help="Cap GIF frame rate (default 15)")
    p.add_argument("--stride", type=int, default=0, help="Take every Nth frame (0 = derive from fps cap)")
    args = p.parse_args()

    mp4_path = args.input.resolve()
    if not mp4_path.is_file():
        raise SystemExit(f"Missing file: {mp4_path}")

    out = args.output
    if out is None:
        out = mp4_path.with_suffix(".gif")
    else:
        out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    reader = imageio.get_reader(str(mp4_path))
    meta = reader.get_meta_data()
    fps = float(meta.get("fps", 30) or 30)
    gif_fps = min(fps, args.max_fps)
    stride = max(1, int(round(fps / gif_fps))) if args.stride <= 0 else max(1, args.stride)

    frames = [frame for i, frame in enumerate(reader) if i % stride == 0]
    reader.close()

    imageio.mimsave(str(out), frames, fps=min(fps / stride, args.max_fps), loop=0)
    print(f"Wrote {out} ({len(frames)} frames, ~{min(fps / stride, args.max_fps):.1f} fps)")


if __name__ == "__main__":
    main()
