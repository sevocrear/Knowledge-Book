from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load(name: str, filename: str):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_concurrent_beats_exclusive_on_mixed_traffic() -> None:
    module = _load("multiplex", "01_concurrent_model_multiplexing.py")
    results = module.compare_multiplexing()
    exclusive = results["exclusive"]
    concurrent = results["concurrent"]

    assert concurrent.throughput > exclusive.throughput + 5.0
    assert concurrent.drop_rate < exclusive.drop_rate - 0.05
    assert concurrent.completed > exclusive.completed
    assert concurrent.mean_latency_s < exclusive.mean_latency_s
    assert concurrent.mean_latency_s < 0.15
