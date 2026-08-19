"""Dynamic batching на GPU: почему пачка ускоряет serving.

Модель времени GPU: T(B) = overhead + per_item · B.
Из-за фиксированного overhead пропускная способность растёт с размером пачки,
пока хватает памяти.

Ожидаемое поведение при высокой нагрузке:
- sequential (B=1) упирается в overhead и даёт меньший throughput;
- dynamic batching (ждём до max_wait или max_batch) обрабатывает больше запросов
  в секунду и снижает долю таймаутов.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class BatchingConfig:
    """Параметры симуляции sequential vs dynamic batching."""

    arrival_rate: float
    duration_s: float = 6.0
    gpu_overhead_s: float = 0.008
    gpu_per_item_s: float = 0.002
    max_batch: int = 8
    max_wait_s: float = 0.008
    timeout_s: float = 0.25
    seed: int = 11


@dataclass(frozen=True)
class BatchingMetrics:
    throughput: float
    completed: int
    dropped: int
    drop_rate: float
    mean_latency_s: float
    mean_batch_size: float


def gpu_batch_seconds(batch_size: int, overhead_s: float, per_item_s: float) -> float:
    """Время прогона пачки: фиксированный кернельный overhead плюс линейный член."""

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    return overhead_s + per_item_s * batch_size


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


def simulate_sequential(cfg: BatchingConfig) -> BatchingMetrics:
    """Каждый запрос идёт на GPU отдельно (batch=1)."""

    rng = random.Random(cfg.seed)
    arrivals = _poisson_arrivals(cfg.arrival_rate, cfg.duration_s, rng)
    gpu_free = 0.0
    latencies: list[float] = []
    dropped = 0
    service = gpu_batch_seconds(1, cfg.gpu_overhead_s, cfg.gpu_per_item_s)
    for arrival_t in arrivals:
        start = max(arrival_t, gpu_free)
        wait = start - arrival_t
        if wait > cfg.timeout_s:
            dropped += 1
            continue
        gpu_free = start + service
        latencies.append(wait + service)
    duration = max(cfg.duration_s, gpu_free)
    return BatchingMetrics(
        throughput=len(latencies) / duration,
        completed=len(latencies),
        dropped=dropped,
        drop_rate=dropped / len(arrivals) if arrivals else 0.0,
        mean_latency_s=sum(latencies) / len(latencies) if latencies else math.inf,
        mean_batch_size=1.0,
    )


def simulate_dynamic_batching(cfg: BatchingConfig) -> BatchingMetrics:
    """Сборщик пачек: ждём max_wait_s или пока наберётся max_batch."""

    rng = random.Random(cfg.seed)
    arrivals = _poisson_arrivals(cfg.arrival_rate, cfg.duration_s, rng)
    gpu_free = 0.0
    latencies: list[float] = []
    dropped = 0
    batch_sizes: list[int] = []
    i = 0
    n = len(arrivals)
    while i < n:
        # Пачка стартует, когда GPU свободен и есть хотя бы один запрос.
        first = arrivals[i]
        open_t = max(first, gpu_free)
        if open_t - first > cfg.timeout_s:
            dropped += 1
            i += 1
            continue
        deadline = open_t + cfg.max_wait_s
        batch = [first]
        i += 1
        while i < n and len(batch) < cfg.max_batch:
            nxt = arrivals[i]
            if nxt > deadline:
                break
            if deadline - nxt > cfg.timeout_s and nxt < open_t:
                dropped += 1
                i += 1
                continue
            batch.append(nxt)
            i += 1
        start = max(open_t, gpu_free)
        service = gpu_batch_seconds(len(batch), cfg.gpu_overhead_s, cfg.gpu_per_item_s)
        finish = start + service
        kept: list[float] = []
        for arrival_t in batch:
            latency = finish - arrival_t
            if latency > cfg.timeout_s:
                dropped += 1
            else:
                latencies.append(latency)
                kept.append(arrival_t)
        if kept:
            batch_sizes.append(len(kept))
        gpu_free = finish

    duration = max(cfg.duration_s, gpu_free)
    return BatchingMetrics(
        throughput=len(latencies) / duration,
        completed=len(latencies),
        dropped=dropped,
        drop_rate=dropped / n if n else 0.0,
        mean_latency_s=sum(latencies) / len(latencies) if latencies else math.inf,
        mean_batch_size=sum(batch_sizes) / len(batch_sizes) if batch_sizes else 0.0,
    )


def compare_batching(arrival_rate: float = 250.0, seed: int = 11) -> dict[str, BatchingMetrics]:
    """Высокая нагрузка, при которой sequential не успевает из-за GPU overhead."""

    cfg = BatchingConfig(arrival_rate=arrival_rate, seed=seed)
    return {
        "sequential": simulate_sequential(cfg),
        "dynamic_batching": simulate_dynamic_batching(cfg),
    }


if __name__ == "__main__":
    results = compare_batching()
    for name, metrics in results.items():
        print(f"=== {name} ===")
        print(f"throughput={metrics.throughput:.1f} items/s  mean_batch={metrics.mean_batch_size:.2f}")
        print(
            f"completed={metrics.completed} dropped={metrics.dropped} drop_rate={metrics.drop_rate:.3f} "
            f"mean_latency={metrics.mean_latency_s*1000:.1f} ms"
        )
        print()
