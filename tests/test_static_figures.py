"""Тест генератора статических иллюстраций (пропускается без matplotlib)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load():
    path = REPO_ROOT / "scripts" / "viz" / "static_figures.py"
    spec = importlib.util.spec_from_file_location("static_figures", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_every_registered_figure_is_referenced_from_its_topic_readme() -> None:
    mod = _load()
    for name, (slug, _fn) in mod.FIGURES.items():
        readme = (REPO_ROOT / "topics" / slug / "README.md").read_text(encoding="utf-8")
        assert f"](./assets/images/{name}.png)" in readme, f"{slug}/README.md не встраивает {name}.png"
        assert (REPO_ROOT / "topics" / slug / "assets" / "images" / f"{name}.png").is_file()


def test_generator_writes_png(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    mod = _load()
    assert mod.main(["--only", "venn_conditional", "--out-root", str(tmp_path)]) == 0
    out = tmp_path / "bayes-theorem-and-probability-foundations" / "assets" / "images" / "venn_conditional.png"
    assert out.is_file() and out.stat().st_size > 5000
