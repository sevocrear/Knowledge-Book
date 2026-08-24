from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "01_capacity_planning.py"
    spec = importlib.util.spec_from_file_location("capacity_planning", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_rejects_invalid_workers() -> None:
    module = _load_module()
    try:
        module.simulate_inference_queue(
            module.CapacityConfig(
                n_clients=10,
                req_per_client_per_s=0.2,
                service_rate_per_s=50.0,
                n_workers=0,
            )
        )
        assert False, "expected ValueError"
    except ValueError:
        assert True


def test_one_thousand_clients_need_more_gpus() -> None:
    """Сигнал темы: 100 клиентов стабильны, 1000 на 1 GPU ломаются, масштаб чинит."""

    module = _load_module()
    results = module.compare_100_vs_1000()
    small = results["clients_100"]
    overloaded = results["clients_1000_one_gpu"]
    scaled = results["clients_1000_scaled"]

    assert small.utilization < 1.0
    assert small.drop_rate == 0.0
    assert small.p99_latency_s < 0.4

    assert overloaded.utilization > 1.0
    assert overloaded.drop_rate > small.drop_rate + 0.4
    assert overloaded.p99_latency_s > small.p99_latency_s + 0.1

    assert scaled.utilization < 1.0
    assert scaled.drop_rate == 0.0
    assert scaled.p99_latency_s < overloaded.p99_latency_s - 0.2
    assert scaled.completed > overloaded.completed * 3


def test_little_law_holds_when_stable() -> None:
    module = _load_module()
    metrics = module.compare_100_vs_1000()["clients_100"]
    predicted = metrics.little_predicted_queue
    observed = metrics.mean_queue_length
    assert predicted > 0
    rel_err = abs(predicted - observed) / predicted
    assert rel_err < 0.15
