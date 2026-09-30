"""Пропуск тестов с маркером `hyperframes`, если не включён настоящий прогон `npx hyperframes check`."""

from __future__ import annotations

import os
import shutil

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    enabled = os.environ.get("KB_RUN_HYPERFRAMES") == "1" and shutil.which("npx") is not None
    if enabled:
        return
    skip = pytest.mark.skip(reason="hyperframes check отключён (KB_RUN_HYPERFRAMES=1 и Node 22+ с npx)")
    for item in items:
        if "hyperframes" in item.keywords:
            item.add_marker(skip)
