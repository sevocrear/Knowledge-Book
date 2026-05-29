from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


def _load_module():
    topic_dir = Path(__file__).resolve().parents[1]
    script_path = topic_dir / "scripts" / "01_patch_nn_anomaly_score.py"
    spec = importlib.util.spec_from_file_location("patch_nn_anomaly_score", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_defect_score_exceeds_normal_score() -> None:
    pnas = _load_module()
    score_ok, score_bad = pnas.demo_scores(seed=0)
    margin = 0.05
    assert score_bad > score_ok + margin, (
        f"expected defect score > normal + {margin}, got ok={score_ok}, bad={score_bad}"
    )


def test_top1_percent_aggregation_uses_worst_patches() -> None:
    pnas = _load_module()
    dist = np.zeros(100, dtype=np.float64)
    dist[0] = 1.0
    s = pnas.image_anomaly_score(dist, top_fraction=0.01)
    assert s >= 0.99
