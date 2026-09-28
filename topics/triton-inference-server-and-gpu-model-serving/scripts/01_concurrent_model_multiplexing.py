"""Concurrent execution двух моделей на одной GPU (идея Triton).

Демонстрирует: если планировщик пускает модели только по очереди (строго
одна за раз), смешанный трафик даёт длинные хвосты и дропы. Если разрешить
одновременное исполнение двух instance (как concurrent model execution
в Triton: по одному инстансу на модель), суммарный throughput растёт,
а доля таймаутов падает.

Модель времени:
- exclusive: один FIFO на GPU, сервис = полное service_s;
- concurrent: до одного активного запроса на модель A и на модель B;
  при одновременной работе обе замедляются в ``interference`` раз.

Ожидаемое поведение:
- concurrent.throughput > exclusive.throughput;
- concurrent.drop_rate < exclusive.drop_rate;
- concurrent.mean_latency_s < exclusive.mean_latency_s при перегрузке.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class MultiplexConfig:
    """Параметры смешанного трафика двух моделей на одной GPU."""

    duration_s: float = 8.0
    arrival_rate_a: float = 40.0
    arrival_rate_b: float = 35.0
    service_a_s: float = 0.020
    service_b_s: float = 0.025
    interference: float = 1.35
    timeout_s: float = 0.20
    seed: int = 7


@dataclass(frozen=True)
class MultiplexMetrics:
    throughput: float
    completed: int
    dropped: int
    drop_rate: float
    mean_latency_s: float
    p99_latency_s: float


def _poisson_arrivals(rate: float, duration_s: float, rng: random.Random) -> list[float]:
    arrivals: list[float] = []
    t = 0.0
    while True:
        u = max(rng.random(), 1e-12)
        t += -math.log(u) / rate
        if t >= duration_s:
            break
        arrivals.append(t)
    return arrivals


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, max(0, int(math.ceil(q * len(sorted_values)) - 1)))
    return sorted_values[idx]


def _finalize(latencies: list[float], dropped: int, duration_s: float) -> MultiplexMetrics:
    completed = len(latencies)
    latencies = sorted(latencies)
    return MultiplexMetrics(
        throughput=completed / duration_s,
        completed=completed,
        dropped=dropped,
        drop_rate=dropped / max(1, completed + dropped),
        mean_latency_s=(sum(latencies) / completed) if completed else 0.0,
        p99_latency_s=_percentile(latencies, 0.99),
    )


def simulate_exclusive(cfg: MultiplexConfig) -> MultiplexMetrics:
    """Строго одна модель на GPU: общий FIFO."""

    rng = random.Random(cfg.seed)
    events: list[tuple[float, str, float]] = []
    for t in _poisson_arrivals(cfg.arrival_rate_a, cfg.duration_s, rng):
        events.append((t, "a", cfg.service_a_s))
    for t in _poisson_arrivals(cfg.arrival_rate_b, cfg.duration_s, rng):
        events.append((t, "b", cfg.service_b_s))
    events.sort(key=lambda x: x[0])

    gpu_free = 0.0
    latencies: list[float] = []
    dropped = 0
    for arrival, _kind, service in events:
        start = max(arrival, gpu_free)
        wait = start - arrival
        if wait > cfg.timeout_s:
            dropped += 1
            continue
        gpu_free = start + service
        latencies.append(wait + service)
    return _finalize(latencies, dropped, cfg.duration_s)


def simulate_concurrent(cfg: MultiplexConfig) -> MultiplexMetrics:
    """По одному instance на модель; overlap замедляет обе в interference раз."""

    rng = random.Random(cfg.seed)
    events: list[tuple[float, str, float]] = []
    for t in _poisson_arrivals(cfg.arrival_rate_a, cfg.duration_s, rng):
        events.append((t, "a", cfg.service_a_s))
    for t in _poisson_arrivals(cfg.arrival_rate_b, cfg.duration_s, rng):
        events.append((t, "b", cfg.service_b_s))
    events.sort(key=lambda x: x[0])

    queues: dict[str, list[tuple[float, float]]] = {"a": [], "b": []}
    # work_left — оставшаяся работа в «полных» GPU-секундах; arrival — время прихода.
    work_left: dict[str, float] = {"a": 0.0, "b": 0.0}
    arrival_of: dict[str, float | None] = {"a": None, "b": None}
    latencies: list[float] = []
    dropped = 0
    t = 0.0
    idx = 0

    def n_busy() -> int:
        return sum(1 for k in ("a", "b") if arrival_of[k] is not None)

    def speed() -> float:
        return 1.0 / cfg.interference if n_busy() == 2 else 1.0

    def drop_stale(now: float) -> None:
        nonlocal dropped
        for kind in ("a", "b"):
            kept: list[tuple[float, float]] = []
            for arrival, service in queues[kind]:
                if now - arrival > cfg.timeout_s:
                    dropped += 1
                else:
                    kept.append((arrival, service))
            queues[kind] = kept

    def try_start(now: float) -> None:
        nonlocal dropped
        for kind in ("a", "b"):
            if arrival_of[kind] is not None:
                continue
            while queues[kind]:
                arrival, service = queues[kind].pop(0)
                if now - arrival > cfg.timeout_s:
                    dropped += 1
                    continue
                work_left[kind] = service
                arrival_of[kind] = arrival
                break

    def wall_to_finish(kind: str) -> float:
        if arrival_of[kind] is None:
            return math.inf
        return work_left[kind] / speed()

    while idx < len(events) or any(queues.values()) or n_busy() > 0:
        drop_stale(t)
        try_start(t)

        next_arrival = events[idx][0] if idx < len(events) else None
        finishes = [wall_to_finish(k) for k in ("a", "b") if arrival_of[k] is not None]
        next_finish_dt = min(finishes) if finishes else None

        candidates_abs: list[float] = []
        if next_arrival is not None:
            candidates_abs.append(next_arrival)
        if next_finish_dt is not None:
            candidates_abs.append(t + next_finish_dt)
        if not candidates_abs:
            break

        t_next = min(candidates_abs)
        dt = t_next - t
        if dt > 0 and n_busy() > 0:
            sp = speed()
            for kind in ("a", "b"):
                if arrival_of[kind] is not None:
                    work_left[kind] = max(0.0, work_left[kind] - sp * dt)
        t = t_next

        for kind in ("a", "b"):
            if arrival_of[kind] is not None and work_left[kind] <= 1e-12:
                latencies.append(t - float(arrival_of[kind]))
                arrival_of[kind] = None
                work_left[kind] = 0.0

        if next_arrival is not None and abs(t - next_arrival) < 1e-12:
            arrival, kind, service = events[idx]
            idx += 1
            if t - arrival > cfg.timeout_s:
                dropped += 1
            else:
                queues[kind].append((arrival, service))

    return _finalize(latencies, dropped, cfg.duration_s)


def compare_multiplexing(cfg: MultiplexConfig | None = None) -> dict[str, MultiplexMetrics]:
    cfg = cfg or MultiplexConfig()
    return {
        "exclusive": simulate_exclusive(cfg),
        "concurrent": simulate_concurrent(cfg),
    }


def main() -> None:
    results = compare_multiplexing()
    for name, m in results.items():
        print(
            f"{name:12s}  throughput={m.throughput:6.1f}/s  "
            f"drop={m.drop_rate:5.1%}  mean_lat={m.mean_latency_s*1000:6.1f}ms  "
            f"p99={m.p99_latency_s*1000:6.1f}ms"
        )


if __name__ == "__main__":
    main()
