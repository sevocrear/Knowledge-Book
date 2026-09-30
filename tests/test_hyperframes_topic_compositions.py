"""Статические проверки HyperFrames-проектов визуализаций (`topics/*/visualizations/hyperframes/`).

Гоняются в CI без Node: структура проекта, согласованность оркестратора и сцен, детерминизм,
уникальность id, наличие артефактов и ссылок из README темы. Настоящий `npx hyperframes check`
включается маркером `hyperframes` (см. tests/conftest.py).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOPICS = REPO_ROOT / "topics"
PROJECTS = sorted(p.parent for p in TOPICS.glob("*/visualizations/hyperframes/index.html"))
PROJECT_IDS = [p.parents[1].name[:28] for p in PROJECTS]

FORBIDDEN = ("Math.random(", "Date.now(", "performance.now(", "repeat: -1", "repeat:-1")
_ATTR = lambda name: re.compile(rf'{name}="([^"]*)"')  # noqa: E731


def _attrs(tag: str) -> dict[str, str]:
    return dict(re.findall(r'([\w-]+)="([^"]*)"', tag))


def _hosts(index_html: str) -> list[dict[str, str]]:
    return [_attrs(m.group(0)) for m in re.finditer(r"<div[^>]*data-composition-src=[^>]*>", index_html)]


def test_at_least_one_project_exists() -> None:
    assert PROJECTS, "нет ни одного topics/*/visualizations/hyperframes/index.html"


@pytest.mark.parametrize("project", PROJECTS, ids=PROJECT_IDS)
def test_project_layout(project: Path) -> None:
    for name in ("index.html", "theme.css", "gsap.min.js", "hyperframes.json", "package.json", "storyboard.md"):
        assert (project / name).is_file(), f"нет {name} в {project}"
    assert list((project / "compositions").glob("*.html")), f"нет сцен в {project / 'compositions'}"
    assert not (project.parent / "manim").exists(), "старые Manim-исходники должны быть удалены"
    json.loads((project / "hyperframes.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("project", PROJECTS, ids=PROJECT_IDS)
def test_orchestrator_and_scenes_are_consistent(project: Path) -> None:
    index = (project / "index.html").read_text(encoding="utf-8")
    assert 'data-composition-id="root"' in index
    assert 'window.__timelines["root"]' in index
    assert 'src="./gsap.min.js"' in index, "GSAP должен грузиться локально, не с CDN"
    root = re.search(r"<div[^>]*data-composition-id=\"root\"[^>]*>", index)
    assert root is not None
    root_attrs = _attrs(root.group(0))
    assert root_attrs.get("data-width") == "1920" and root_attrs.get("data-height") == "1080"
    total = float(root_attrs["data-duration"])
    assert 30 <= total <= 60, f"длина клипа {total} с вне диапазона 30–60 с"

    hosts = _hosts(index)
    assert len(hosts) >= 2, "ожидается минимум две сцены-подкомпозиции"
    expected_start = 0.0
    for host in hosts:
        start, dur = float(host["data-start"]), float(host["data-duration"])
        assert abs(start - expected_start) < 1e-6, f"сцены должны идти встык: {host}"
        expected_start += dur
        scene_id = host["data-composition-id"]
        scene_path = project / host["data-composition-src"]
        assert scene_path.is_file(), f"нет файла сцены {scene_path}"
        scene = scene_path.read_text(encoding="utf-8")
        assert "<template>" in scene, f"{scene_path.name}: сцена должна быть обёрнута в <template>"
        assert f'data-composition-id="{scene_id}"' in scene, f"{scene_path.name}: id хоста и сцены различаются"
        assert f'window.__timelines["{scene_id}"]' in scene, f"{scene_path.name}: timeline не зарегистрирован под id сцены"
        assert 'gsap.timeline({ paused: true })' in scene.replace("{paused:true}", "{ paused: true }")
        assert 'href="theme.css"' in scene, f"{scene_path.name}: сцена должна подключать theme.css"
        assert "cdn.jsdelivr.net" not in scene and "unpkg.com" not in scene
    assert abs(expected_start - total) < 1e-6, f"сумма сцен {expected_start} ≠ data-duration root {total}"


@pytest.mark.parametrize("project", PROJECTS, ids=PROJECT_IDS)
def test_determinism_and_unique_ids(project: Path) -> None:
    files = [project / "index.html", *sorted((project / "compositions").glob("*.html"))]
    seen: dict[str, str] = {}
    for f in files:
        text = f.read_text(encoding="utf-8")
        for token in FORBIDDEN:
            assert token not in text, f"{f.name}: запрещённый вызов {token!r}"
        assert "@font-face" not in text, f"{f.name}: именованные шрифты запрещены (см. kb-video)"
        for html_id in re.findall(r'\sid="([^"]+)"', text):
            if html_id == "root":
                continue  # корень каждой сцены по контракту HyperFrames называется root
            assert html_id not in seen, f"id {html_id!r} повторяется в {f.name} и {seen[html_id]}"
            seen[html_id] = f.name


@pytest.mark.parametrize("project", PROJECTS, ids=PROJECT_IDS)
def test_assets_and_readme_embed(project: Path) -> None:
    topic = project.parents[1]
    assets = topic / "assets" / "visualizations"
    gifs = sorted(assets.glob("*.gif"))
    assert gifs, f"нет GIF в {assets}"
    readme = (topic / "README.md").read_text(encoding="utf-8")
    for gif in gifs:
        mp4 = gif.with_suffix(".mp4")
        assert mp4.is_file(), f"нет MP4 рядом с {gif.name}"
        assert gif.stat().st_size < 8 * 1024 * 1024, f"{gif.name} больше 8 MB"
        assert f"](./assets/visualizations/{gif.name})" in readme, f"README темы не встраивает {gif.name} картинкой"
        assert f"](./assets/visualizations/{mp4.name})" in readme, f"README темы не ссылается на {mp4.name}"
    assert "](./visualizations/hyperframes/storyboard.md)" in readme, "README темы должен ссылаться на сториборд"
    assert "<video" not in readme, "в README темы не должно быть <video> — GitHub/Obsidian его не играют"


@pytest.mark.hyperframes
@pytest.mark.parametrize("project", PROJECTS, ids=PROJECT_IDS)
def test_hyperframes_check_passes(project: Path) -> None:
    proc = subprocess.run(
        ["npx", "-y", "hyperframes@0.8.81", "check", "--json"],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ},
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + "\n---\n" + proc.stderr[-2000:]
