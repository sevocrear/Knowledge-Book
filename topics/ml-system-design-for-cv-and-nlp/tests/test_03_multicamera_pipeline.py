from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "03_multicamera_pipeline.py"
    spec = importlib.util.spec_from_file_location("multicamera_pipeline", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_latest_frame_is_fresher_than_fifo_on_50_cameras() -> None:
    module = _load_module()
    results = module.compare_cameras(n_cameras=50, duration_s=2.0)
    naive = results["naive_fifo"]
    smart = results["latest_frame_batch"]

    assert smart.mean_staleness_s < naive.mean_staleness_s * 0.15
    assert smart.processed_fps > naive.processed_fps + 100.0
    assert smart.unique_cameras_served == 50
    assert naive.unique_cameras_served == 50


def test_ten_cameras_keep_up_with_latest_frame_policy() -> None:
    module = _load_module()
    results = module.compare_cameras(n_cameras=10, duration_s=2.0)
    smart = results["latest_frame_batch"]
    naive = results["naive_fifo"]

    assert smart.drop_rate < 0.05
    assert smart.mean_staleness_s < 0.08
    assert smart.mean_staleness_s < naive.mean_staleness_s
    assert smart.processed_fps >= naive.processed_fps
