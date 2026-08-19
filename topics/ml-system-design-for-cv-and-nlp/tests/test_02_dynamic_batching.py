from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "02_dynamic_batching.py"
    spec = importlib.util.spec_from_file_location("dynamic_batching", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_gpu_time_grows_sublinearly_per_item() -> None:
    module = _load_module()
    t1 = module.gpu_batch_seconds(1, overhead_s=0.008, per_item_s=0.002)
    t8 = module.gpu_batch_seconds(8, overhead_s=0.008, per_item_s=0.002)
    assert t1 == 0.010
    assert t8 == 0.024
    # Пачка из 8 дешевле, чем 8 одиночных прогонов.
    assert t8 < 8 * t1


def test_dynamic_batching_raises_throughput_and_cuts_drops() -> None:
    module = _load_module()
    results = module.compare_batching()
    sequential = results["sequential"]
    batched = results["dynamic_batching"]

    assert batched.throughput > sequential.throughput + 80.0
    assert batched.drop_rate < sequential.drop_rate - 0.3
    assert batched.mean_batch_size > 2.0
    assert batched.mean_latency_s < sequential.mean_latency_s
