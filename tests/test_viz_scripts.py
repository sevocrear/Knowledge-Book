"""Тесты хелперов scripts/viz (без запуска ffmpeg)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    path = REPO_ROOT / "scripts" / "viz" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_mp4_to_gif_filter_chain_has_palette_passes() -> None:
    mod = _load("mp4_to_gif")
    chain = mod.build_filter(12, 960, 128)
    assert chain.startswith("fps=12,scale=960:-2")
    assert "palettegen=max_colors=128" in chain and "paletteuse" in chain
    assert "scale=iw:ih" in mod.build_filter(10, 0, 64)


def test_mp4_to_gif_missing_input_returns_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    mod = _load("mp4_to_gif")
    assert mod.main([str(tmp_path / "nope.mp4")]) == 2
    assert "Нет файла" in capsys.readouterr().err


def test_gif_to_mp4_missing_input_returns_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    mod = _load("gif_to_mp4")
    assert mod.main([str(tmp_path / "nope.gif")]) == 2
    assert "Нет файла" in capsys.readouterr().err
