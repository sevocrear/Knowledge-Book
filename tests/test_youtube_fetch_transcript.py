"""Tests for scripts/youtube_fetch_transcript.py (parsing only; no network)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "youtube_fetch_transcript.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "youtube_fetch_transcript",
        _MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


_MOD = _load_module()
parse_video_id = _MOD.parse_video_id
format_timestamp = _MOD.format_timestamp
parse_vtt = _MOD.parse_vtt


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("C_GG5g38vLU", "C_GG5g38vLU"),
        ("https://www.youtube.com/watch?v=C_GG5g38vLU", "C_GG5g38vLU"),
        ("https://youtu.be/C_GG5g38vLU?t=42", "C_GG5g38vLU"),
        ("https://www.youtube.com/embed/C_GG5g38vLU", "C_GG5g38vLU"),
    ],
)
def test_parse_video_id(raw: str, expected: str) -> None:
    assert parse_video_id(raw) == expected


def test_parse_video_id_invalid() -> None:
    with pytest.raises(ValueError):
        parse_video_id("not-a-url")


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.0, "00:00"),
        (65.0, "01:05"),
        (3661.0, "01:01:01"),
    ],
)
def test_format_timestamp(seconds: float, expected: str) -> None:
    assert format_timestamp(seconds) == expected


def test_parse_vtt_minimal(tmp_path: Path) -> None:
    vtt = tmp_path / "sample.vtt"
    vtt.write_text(
        "WEBVTT\n\n"
        "1\n"
        "00:00:01.000 --> 00:00:03.000\n"
        "Hello world\n\n"
        "2\n"
        "00:00:04.000 --> 00:00:05.500\n"
        "Second line\n",
        encoding="utf-8",
    )
    snippets = parse_vtt(vtt)
    assert len(snippets) == 2
    assert snippets[0].text == "Hello world"
    assert snippets[0].start == pytest.approx(1.0)
    assert snippets[1].text == "Second line"
