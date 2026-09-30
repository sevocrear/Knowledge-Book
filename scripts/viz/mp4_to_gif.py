#!/usr/bin/env python3
"""MP4 → GIF через ffmpeg (двухпроходная палитра). Используется после рендера HyperFrames.

GIF нужен как «везде рендерящийся» fallback для Markdown (GitHub, Obsidian, IDE-превью):
`![](./assets/visualizations/<name>.gif)`. MP4 остаётся полной версией.

Пример:
    uv run python scripts/viz/mp4_to_gif.py topics/<slug>/assets/visualizations/<name>.mp4 \
        -o topics/<slug>/assets/visualizations/<name>.gif --fps 12 --width 960
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def build_filter(fps: float, width: int, max_colors: int) -> str:
    scale = f"scale={width}:-2:flags=lanczos" if width > 0 else "scale=iw:ih"
    return (
        f"fps={fps},{scale},split[s0][s1];"
        f"[s0]palettegen=max_colors={max_colors}:stats_mode=diff[p];"
        f"[s1][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"
    )


def convert(mp4: Path, gif: Path, *, fps: float = 12.0, width: int = 960, max_colors: int = 128, ffmpeg: str = "ffmpeg") -> None:
    gif.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(mp4),
        "-vf", build_filter(fps, width, max_colors),
        "-loop", "0",
        str(gif),
    ]
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="MP4 → GIF (ffmpeg, palettegen/paletteuse).")
    p.add_argument("input", type=Path, help="Исходный .mp4")
    p.add_argument("-o", "--output", type=Path, help="Выходной .gif (по умолчанию — рядом с тем же именем)")
    p.add_argument("--fps", type=float, default=12.0, help="Частота кадров GIF (по умолчанию 12)")
    p.add_argument("--width", type=int, default=960, help="Ширина GIF в px, 0 = как у видео (по умолчанию 960)")
    p.add_argument("--max-colors", type=int, default=128, help="Размер палитры (по умолчанию 128)")
    p.add_argument("--ffmpeg", default="ffmpeg", help="Путь к бинарю ffmpeg")
    args = p.parse_args(argv)

    mp4 = args.input.resolve()
    if not mp4.is_file():
        print(f"Нет файла: {mp4}", file=sys.stderr)
        return 2
    if shutil.which(args.ffmpeg) is None:
        print(f"ffmpeg не найден ({args.ffmpeg}); установите ffmpeg или укажите --ffmpeg", file=sys.stderr)
        return 2
    gif = (args.output or mp4.with_suffix(".gif")).resolve()
    convert(mp4, gif, fps=args.fps, width=args.width, max_colors=args.max_colors, ffmpeg=args.ffmpeg)
    print(f"Записан {gif} ({gif.stat().st_size / 1e6:.1f} MB, {args.fps:g} fps, width={args.width})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
