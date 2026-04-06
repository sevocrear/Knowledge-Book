"""Skip Manim tests when the optional viz dependency group is not installed."""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    try:
        import manim  # noqa: F401
    except ImportError:
        skip = pytest.mark.skip(reason="manim not installed (run: uv sync --group viz)")
        for item in items:
            if "manim" in item.keywords:
                item.add_marker(skip)
