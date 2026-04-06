from __future__ import annotations

from pathlib import Path

import importlib.util
import sys


def _load_kb_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "kb_validate_links.py"
    spec = importlib.util.spec_from_file_location("kb_validate_links", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


_KB = _load_kb_module()
validate_markdown_files = _KB.validate_markdown_files
topic_has_table_of_contents = _KB.topic_has_table_of_contents
validate_root_readme_topic_index = _KB.validate_root_readme_topic_index


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_validate_markdown_files_ok(tmp_path: Path) -> None:
    repo = tmp_path
    topics = repo / "topics" / "t1"
    _write(topics / "README.md", "See [root](../../README.md) and [local](./asset.txt).\n")
    _write(repo / "README.md", "Hello\n")
    _write(topics / "asset.txt", "ok\n")

    broken = validate_markdown_files([topics / "README.md"], repo_root=repo)
    assert broken == []


def test_validate_markdown_files_broken_relative_link(tmp_path: Path) -> None:
    repo = tmp_path
    md = repo / "topics" / "t1" / "README.md"
    _write(md, "Broken [x](./missing.md)\n")

    broken = validate_markdown_files([md], repo_root=repo)
    assert len(broken) == 1
    assert broken[0].md_file == md
    assert broken[0].raw_target == "./missing.md"


def test_ignores_external_and_anchor_links(tmp_path: Path) -> None:
    repo = tmp_path
    md = repo / "README.md"
    _write(
        md,
        "\n".join(
            [
                "External [x](https://example.com/a/b)\n",
                "Anchor [y](#section)\n",
                "Mail [z](mailto:test@example.com)\n",
            ]
        ),
    )

    broken = validate_markdown_files([md], repo_root=repo)
    assert broken == []


def test_flags_directory_targets(tmp_path: Path) -> None:
    repo = tmp_path
    d = repo / "topics" / "t1" / "dir"
    d.mkdir(parents=True)
    md = repo / "topics" / "t1" / "README.md"
    _write(md, "Dir link [x](./dir)\n")

    broken = validate_markdown_files([md], repo_root=repo)
    assert len(broken) == 1
    assert "directory" in broken[0].reason


def test_topic_has_table_of_contents(tmp_path: Path) -> None:
    repo = tmp_path
    ok = repo / "topics" / "a" / "README.md"
    _write(ok, "# T\n\n## Table of Contents\n\n1. [x](#x)\n")
    bad = repo / "topics" / "b" / "README.md"
    _write(bad, "# T\n\n### Contents\n\n1. x\n")

    assert topic_has_table_of_contents(ok) is True
    assert topic_has_table_of_contents(bad) is False


def test_validate_root_readme_topic_index_complete(tmp_path: Path) -> None:
    repo = tmp_path
    _write(repo / "topics" / "t1" / "README.md", "## Table of Contents\n\n- [x](#x)\n")
    _write(repo / "topics" / "t2" / "README.md", "## Table of Contents\n\n- [x](#x)\n")
    _write(
        repo / "README.md",
        "## Contents\n\n- [T1](./topics/t1/README.md)\n- [T2](./topics/t2/README.md#anchor)\n",
    )
    missing, unknown = validate_root_readme_topic_index(repo)
    assert missing == set()
    assert unknown == set()


def test_validate_root_readme_topic_index_missing_and_unknown(tmp_path: Path) -> None:
    repo = tmp_path
    _write(repo / "topics" / "on-disk" / "README.md", "## Table of Contents\n\n- [x](#x)\n")
    _write(
        repo / "README.md",
        "## Contents\n\n- [Ghost](./topics/not-on-disk/README.md)\n",
    )
    missing, unknown = validate_root_readme_topic_index(repo)
    assert missing == {"on-disk"}
    assert unknown == {"not-on-disk"}
