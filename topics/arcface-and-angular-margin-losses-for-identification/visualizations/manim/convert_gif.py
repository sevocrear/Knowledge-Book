from __future__ import annotations

import glob
import os
import shutil

import imageio.v2 as imageio


def main() -> None:
    here = os.path.abspath(os.getcwd())
    topic_dir = os.path.abspath(os.path.join(here, "..", ".."))
    assets_dir = os.path.join(topic_dir, "assets", "visualizations")
    media_dir = os.path.join(here, ".media")

    mp4_files = sorted(glob.glob(os.path.join(media_dir, "**", "*.mp4"), recursive=True))
    if not mp4_files:
        raise FileNotFoundError("No MP4 files found under .media/")

    mp4_path = mp4_files[-1]
    os.makedirs(assets_dir, exist_ok=True)
    target_mp4 = os.path.join(assets_dir, "arcface_angular_margin.mp4")
    shutil.copy2(mp4_path, target_mp4)

    reader = imageio.get_reader(mp4_path)
    fps = float(reader.get_meta_data().get("fps", 30.0))
    gif_fps = min(15.0, fps)
    stride = max(1, int(round(fps / gif_fps)))
    frames = [frame for i, frame in enumerate(reader) if i % stride == 0]
    if not frames:
        raise RuntimeError("Rendered video is empty; no frames for GIF.")

    gif_path = os.path.join(assets_dir, "arcface_angular_margin.gif")
    imageio.mimsave(gif_path, frames, fps=gif_fps, loop=0)
    print(f"Saved MP4: {target_mp4}")
    print(f"Saved GIF: {gif_path} ({len(frames)} frames @ {gif_fps:.1f} fps)")


if __name__ == "__main__":
    main()
