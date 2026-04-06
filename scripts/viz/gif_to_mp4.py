#!/usr/bin/env python3
"""Convert GIF to MP4 (lossy / convenience only; prefer Manim → MP4 as source of truth)."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio


def main() -> None:
    p = argparse.ArgumentParser(description="GIF → MP4 (helper when only GIF exists).")
    p.add_argument("input", type=Path, help="Source .gif path")
    p.add_argument("-o", "--output", type=Path, help="Output .mp4 (default: same stem as input)")
    p.add_argument("--fps", type=float, default=15.0, help="Output video fps (default 15)")
    args = p.parse_args()

    gif_path = args.input.resolve()
    if not gif_path.is_file():
        raise SystemExit(f"Missing file: {gif_path}")

    out = args.output
    if out is None:
        out = gif_path.with_suffix(".mp4")
    else:
        out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    reader = imageio.get_reader(str(gif_path))
    frames = [frame for frame in reader]
    reader.close()

    if not frames:
        raise SystemExit("GIF has no frames")

    with imageio.get_writer(str(out), fps=args.fps, codec="libx264", quality=8) as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"Wrote {out} ({len(frames)} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
