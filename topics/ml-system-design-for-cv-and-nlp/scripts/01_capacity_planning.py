"""Планирование ёмкости serving: 100 клиентов vs 1000 на одном GPU.

Скрипт симулирует очередь инференса с несколькими воркерами (репликами модели)
и показывает закон Литтла плюс утилизацию ρ = λ / (n_workers · μ).

Ожидаемое поведение:
- при 100 клиентах и 1 GPU система стабильна (ρ < 1), p99-латентность ограничена;
- при 1000 клиентах на том же GPU ρ > 1: растут дропы и/или латентность;
- после горизонтального масштаба GPU пропорционально нагрузке p99 снова
  становится сопоставимым со сценарием на 100 клиентов.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class CapacityConfig:
    """Параметры дискретно-событийной симуляции очереди инференса."""

    n_clients: int
    req_per_client_per_s: float
    service_rate_per_s: float
    n_workers: int
    duration_s: float = 8.0
    timeout_s: float = 0.5
    seed: int = 7


@dataclass(frozen=True)
class CapacityMetrics:
    """Метрики, которые сравнивают сценарии 100 vs 1000 клиентов."""

    arrival_rate: float
    service_capacity: float
    utilization: float
    completed: int
    dropped: int
    drop_rate: float
    mean_latency_s: float
    p99_latency_s: float
    mean_queue_length: float
    little_predicted_queue: float


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return math.inf
    idx = min(len(sorted_values) - 1, max(0, math.ceil(q * len(sorted_values)) - 1))
    return sorted_values[idx]


def _poisson_arrivals(arrival_rate: float, duration_s: float, rng: random.Random) -> list[float]:
    arrivals: list[float] = []
    t = 0.0
    while True:
        u = max(rng.random(), 1e-12)
        t += -math.log(u) / arrival_rate
        if t >= duration_s:
            break
        arrivals.append(t)
    return arrivals


def simulate_inference_queue(cfg: CapacityConfig) -> CapacityMetrics:
    """Симуляция M/D/n с таймаутом: постоянное время сервиса GPU, пуассоновские приходы."""

    if cfg.n_workers < 1:
        raise ValueError("n_workers must be >= 1")
    if cfg.service_rate_per_s <= 0:
        raise ValueError("service_rate_per_s must be > 0")
    if cfg.n_clients < 1:
        raise ValueError("n_clients must be >= 1")

    arrival_rate = cfg.n_clients * cfg.req_per_client_per_s
    service_capacity = cfg.n_workers * cfg.service_rate_per_s
    utilization = arrival_rate / service_capacity
    service_s = 1.0 / cfg.service_rate_per_s

    rng = random.Random(cfg.seed)
    arrivals = _poisson_arrivals(arrival_rate, cfg.duration_s, rng)
    worker_free_at = [0.0] * cfg.n_workers
    latencies: list[float] = []
    dropped = 0
    in_system_samples: list[int] = []
    finishes: list[float] = []

    for arrival_t in arrivals:
        in_system = sum(1 for finish in finishes if finish > arrival_t)
        in_system_samples.append(in_system)
        worker_i = min(range(cfg.n_workers), key=lambda w: worker_free_at[w])
        start = max(arrival_t, worker_free_at[worker_i])
        wait = start - arrival_t
        if wait > cfg.timeout_s:
            dropped += 1
            continue
        finish = start + service_s
        worker_free_at[worker_i] = finish
        finishes.append(finish)
        latencies.append(wait + service_s)

    latencies.sort()
    mean_latency = sum(latencies) / len(latencies) if latencies else math.inf
    p99 = _percentile(latencies, 0.99)
    mean_q = sum(in_system_samples) / len(in_system_samples) if in_system_samples else 0.0
    offered = len(arrivals)
    drop_rate = dropped / offered if offered else 0.0
    little_l = arrival_rate * mean_latency if math.isfinite(mean_latency) else math.inf

    return CapacityMetrics(
        arrival_rate=arrival_rate,
        service_capacity=service_capacity,
        utilization=utilization,
        completed=len(latencies),
        dropped=dropped,
        drop_rate=drop_rate,
        mean_latency_s=mean_latency,
        p99_latency_s=p99,
        mean_queue_length=mean_q,
        little_predicted_queue=little_l,
    )


def compare_100_vs_1000(
    req_per_client_per_s: float = 0.4,
    service_rate_per_s: float = 50.0,
    seed: int = 7,
) -> dict[str, CapacityMetrics]:
    """Три канонических сценария: 100×1 GPU, 1000×1 GPU, 1000×10 GPU."""

    small = simulate_inference_queue(
        CapacityConfig(
            n_clients=100,
            req_per_client_per_s=req_per_client_per_s,
            service_rate_per_s=service_rate_per_s,
            n_workers=1,
            seed=seed,
        )
    )
    overloaded = simulate_inference_queue(
        CapacityConfig(
            n_clients=1000,
            req_per_client_per_s=req_per_client_per_s,
            service_rate_per_s=service_rate_per_s,
            n_workers=1,
            seed=seed + 1,
        )
    )
    scaled = simulate_inference_queue(
        CapacityConfig(
            n_clients=1000,
            req_per_client_per_s=req_per_client_per_s,
            service_rate_per_s=service_rate_per_s,
            n_workers=10,
            seed=seed + 2,
        )
    )
    return {"clients_100": small, "clients_1000_one_gpu": overloaded, "clients_1000_scaled": scaled}


if __name__ == "__main__":
    results = compare_100_vs_1000()
    for name, metrics in results.items():
        print(f"=== {name} ===")
        print(f"λ={metrics.arrival_rate:.1f}/s  capacity={metrics.service_capacity:.1f}/s  ρ={metrics.utilization:.2f}")
        print(
            f"completed={metrics.completed} dropped={metrics.dropped} drop_rate={metrics.drop_rate:.3f}"
        )
        print(
            f"mean_latency={metrics.mean_latency_s*1000:.1f} ms  "
            f"p99={metrics.p99_latency_s*1000:.1f} ms  mean_queue={metrics.mean_queue_length:.2f}"
        )
        print()
