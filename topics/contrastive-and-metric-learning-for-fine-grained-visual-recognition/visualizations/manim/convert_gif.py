import glob
import os

import imageio.v2 as imageio


def main() -> None:
    topic_dir = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
    assets_dir = os.path.join(topic_dir, "assets", "visualizations")
    media_dir = os.path.join(os.getcwd(), ".media")

    mp4_files = glob.glob(os.path.join(media_dir, "**", "*.mp4"), recursive=True)
    if not mp4_files:
        raise SystemExit("No mp4 files found in .media. Render with manim first.")

    mp4_path = sorted(mp4_files, key=os.path.getmtime)[-1]

    reader = imageio.get_reader(mp4_path)
    fps = reader.get_meta_data().get("fps", 30)
    gif_fps = min(fps, 15)
    stride = max(1, int(round(fps / gif_fps)))

    frames = [frame for i, frame in enumerate(reader) if i % stride == 0]
    os.makedirs(assets_dir, exist_ok=True)
    gif_path = os.path.join(assets_dir, "contrastive_embedding_space.gif")
    imageio.mimsave(gif_path, frames, fps=gif_fps, loop=0)
    print(f"Saved GIF: {gif_path} ({len(frames)} frames @ {gif_fps} fps)")


if __name__ == "__main__":
    main()

