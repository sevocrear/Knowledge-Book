"""Smoke tests: Manim renders last frame without error (optional viz group)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

TOPIC_SCENES: list[tuple[str, str]] = [
    ("contrastive-and-metric-learning-for-fine-grained-visual-recognition", "ContrastiveEmbeddingSpace"),
    ("arcface-and-angular-margin-losses-for-identification", "ArcFaceAngularMarginScene"),
    ("transformers-attention-and-vision-transformers-vit", "ViTPatchesAndAttentionScene"),
    ("retrieval-augmented-generation-rag", "RAGPipelineScene"),
]


@pytest.mark.manim
@pytest.mark.parametrize("topic_slug,scene_class", TOPIC_SCENES, ids=[t[0][:24] for t in TOPIC_SCENES])
def test_manim_scene_last_frame(topic_slug: str, scene_class: str, tmp_path: Path) -> None:
    pytest.importorskip("manim")
    manim_dir = REPO_ROOT / "topics" / topic_slug / "visualizations" / "manim"
    scene_py = manim_dir / "scene.py"
    assert scene_py.is_file(), f"Missing {scene_py}"
    media = tmp_path / "media"
    media.mkdir()
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "manim",
            "scene.py",
            scene_class,
            "-ql",
            "--media_dir",
            str(media),
            "--disable_caching",
            "-s",
        ],
        cwd=manim_dir,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stdout + "\n---\n" + proc.stderr
    pngs = list(media.rglob("*.png"))
    assert pngs, f"expected at least one PNG under {media}"
