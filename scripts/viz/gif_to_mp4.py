#!/usr/bin/env python3
"""GIF → MP4 через ffmpeg (вспомогательно, когда есть только GIF; источник истины — HyperFrames-рендер)."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def convert(gif: Path, mp4: Path, *, fps: float = 15.0, crf: int = 23, ffmpeg: str = "ffmpeg") -> None:
    mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(gif),
        "-movflags", "+faststart",
        "-pix_fmt", "yuv420p",
        "-vf", f"fps={fps},scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-crf", str(crf),
        str(mp4),
    ]
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="GIF → MP4 (ffmpeg, libx264).")
    p.add_argument("input", type=Path, help="Исходный .gif")
    p.add_argument("-o", "--output", type=Path, help="Выходной .mp4 (по умолчанию — рядом с тем же именем)")
    p.add_argument("--fps", type=float, default=15.0, help="Частота кадров видео (по умолчанию 15)")
    p.add_argument("--crf", type=int, default=23, help="CRF libx264 (по умолчанию 23)")
    p.add_argument("--ffmpeg", default="ffmpeg", help="Путь к бинарю ffmpeg")
    args = p.parse_args(argv)

    gif = args.input.resolve()
    if not gif.is_file():
        print(f"Нет файла: {gif}", file=sys.stderr)
        return 2
    if shutil.which(args.ffmpeg) is None:
        print(f"ffmpeg не найден ({args.ffmpeg}); установите ffmpeg или укажите --ffmpeg", file=sys.stderr)
        return 2
    mp4 = (args.output or gif.with_suffix(".mp4")).resolve()
    convert(gif, mp4, fps=args.fps, crf=args.crf, ffmpeg=args.ffmpeg)
    print(f"Записан {mp4} ({mp4.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
