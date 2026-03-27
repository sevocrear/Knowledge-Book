from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_train_fn():
    topic_dir = Path(__file__).resolve().parents[1]
    script_path = topic_dir / "scripts" / "01_metric_learning_triplet_synthetic.py"
    spec = importlib.util.spec_from_file_location("metric_learning_triplet_synth", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.train_metric_learning


def test_metric_learning_improves_recall() -> None:
    train_metric_learning = _load_train_fn()
    before, after = train_metric_learning(
        seed=123,
        steps=700,
        p=10,
        k=6,
        margin=0.25,
        lr=2e-3,
        weight_decay=1e-4,
        device="cpu",
    )

    assert after["recall@1"] > before["recall@1"] + 0.015
    # Recall@5 may fluctuate depending on how tightly clusters form; ensure no large regression.
    assert after["recall@5"] > before["recall@5"] - 0.05
