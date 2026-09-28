from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load(name: str, filename: str):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_continuous_batching_serves_more_sessions() -> None:
    module = _load("slots", "02_continuous_batching_slots.py")
    results = module.compare_slot_schedulers()
    static = results["static_batch"]
    continuous = results["continuous"]

    assert continuous.throughput_sessions > static.throughput_sessions + 0.5
    assert continuous.drop_rate <= static.drop_rate + 0.02
    assert continuous.completed > static.completed
    assert continuous.mean_occupancy > 0.2
