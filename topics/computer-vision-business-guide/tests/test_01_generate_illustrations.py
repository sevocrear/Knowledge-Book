#!/usr/bin/env python3
"""Tests for CV business guide illustration generator."""

from __future__ import annotations

from pathlib import Path

import pytest

TOPIC = Path(__file__).resolve().parents[1]
SCRIPT = TOPIC / "scripts" / "01_generate_illustrations.py"
OUT = TOPIC / "assets" / "illustrations"

EXPECTED_MIN = 14


@pytest.fixture(scope="module")
def generated_paths() -> list[Path]:
    import importlib.util

    spec = importlib.util.spec_from_file_location("gen_illus", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.generate_all()


def test_generates_minimum_illustration_count(generated_paths: list[Path]) -> None:
    assert len(generated_paths) >= EXPECTED_MIN


def test_illustration_files_exist_and_non_empty(generated_paths: list[Path]) -> None:
    for path in generated_paths:
        assert path.exists(), f"missing {path.name}"
        assert path.stat().st_size > 500, f"too small {path.name}"


def test_cover_hero_dimensions(generated_paths: list[Path]) -> None:
    from PIL import Image

    cover = OUT / "01_cover_hero.png"
    assert cover.exists()
    with Image.open(cover) as img:
        assert img.width >= 1000
        assert img.height >= 600
