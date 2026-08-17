#!/usr/bin/env python3
"""Tests for CV business guide HTML/EPUB builder."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

TOPIC = Path(__file__).resolve().parents[1]
BUILD = TOPIC / "scripts" / "02_build_book.py"
DIST = TOPIC / "dist"


@pytest.fixture(scope="module")
def build_outputs() -> dict[str, Path]:
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_book", BUILD)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    mod.main()
    return {
        "html": DIST / "index.html",
        "epub": DIST / "computer-vision-business-guide.epub",
        "zip": DIST / "computer-vision-business-guide-html.zip",
    }


def test_html_exists_and_has_chapters(build_outputs: dict[str, Path]) -> None:
    html = build_outputs["html"].read_text(encoding="utf-8")
    assert "Компьютерное зрение" in html
    assert html.count('class="chapter"') >= 20
    assert "assets/illustrations/" in html


def test_epub_valid(build_outputs: dict[str, Path]) -> None:
    epub = build_outputs["epub"]
    assert epub.exists()
    assert epub.stat().st_size > 50_000


def test_html_zip_contains_assets(build_outputs: dict[str, Path]) -> None:
    zpath = build_outputs["zip"]
    assert zpath.exists()
    with zipfile.ZipFile(zpath) as zf:
        names = zf.namelist()
        assert "index.html" in names
        assert any("assets/illustrations/" in n for n in names)
        assert any(n.endswith("book.css") for n in names)


def test_minimum_content_volume(build_outputs: dict[str, Path]) -> None:
    """Proxy for 30+ pages: substantial HTML body text."""
    html = build_outputs["html"].read_text(encoding="utf-8")
    text_len = len(html)
    assert text_len > 80_000, f"book too short: {text_len} chars"
