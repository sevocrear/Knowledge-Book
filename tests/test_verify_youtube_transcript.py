"""Tests for scripts/verify_youtube_transcript.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "verify_youtube_transcript.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("verify_youtube_transcript", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


_MOD = _load()
verify_transcript = _MOD.verify_transcript


def test_verify_ok_sample(tmp_path: Path) -> None:
    path = tmp_path / "ok.txt"
    lines = ["# video: https://www.youtube.com/watch?v=abc\n", "# lang\n", "\n"]
    lines += [f"(00:{i:02d})  word {i} here.\n" for i in range(25)]
    path.write_text("".join(lines), encoding="utf-8")
    assert verify_transcript(path) == []


def test_verify_empty(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")
    assert verify_transcript(path)


def test_verify_too_short(tmp_path: Path) -> None:
    path = tmp_path / "short.txt"
    path.write_text("(00:01)  hi\n", encoding="utf-8")
    errs = verify_transcript(path)
    assert errs
