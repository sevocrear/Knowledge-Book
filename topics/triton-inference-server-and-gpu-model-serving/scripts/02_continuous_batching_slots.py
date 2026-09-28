"""Continuous batching vs static batch для LLM-подобных слотов.

Идея vLLM / TensorRT-LLM: слот GPU занят, пока идёт decode пользователя.
Как только генерация закончилась — слот сразу отдаётся следующему в очереди
(continuous / in-flight batching). Static batch ждёт, пока *все* в пачке
закончат, и только потом берёт новую пачку.

Ожидаемое поведение при том же max_slots и том же потоке сессий:
- continuous завершает больше сессий (выше throughput);
- continuous даёт меньшую долю таймаутов в очереди.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class SlotConfig:
    """Параметры симуляции decode-слотов."""

    duration_s: float = 10.0
    arrival_rate: float = 12.0
    max_slots: int = 4
    mean_tokens: float = 40.0
    token_time_s: float = 0.012
    timeout_s: float = 2.0
    seed: int = 21


@dataclass(frozen=True)
class SlotMetrics:
    throughput_sessions: float
    completed: int
    dropped: int
    drop_rate: float
    mean_latency_s: float
    mean_occupancy: float


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


def _sample_tokens(mean_tokens: float, rng: random.Random) -> int:
    # Геометрическое-подобное: минимум 5 токенов.
    n = int(rng.expovariate(1.0 / mean_tokens))
    return max(5, n)


def simulate_static_batch(cfg: SlotConfig) -> SlotMetrics:
    """Пачка размера max_slots стартует только когда набрана или очередь ждёт."""

    rng = random.Random(cfg.seed)
    arrivals = _poisson_arrivals(cfg.arrival_rate, cfg.duration_s, rng)
    queue: list[tuple[float, int]] = []  # (arrival, tokens)
    t = 0.0
    ai = 0
    completed = 0
    dropped = 0
    latencies: list[float] = []
    busy_integral = 0.0
    last_t = 0.0

    def flush_drops(now: float) -> None:
        nonlocal dropped, queue
        kept: list[tuple[float, int]] = []
        for arrival, tokens in queue:
            if now - arrival > cfg.timeout_s:
                dropped += 1
            else:
                kept.append((arrival, tokens))
        queue = kept

    while ai < len(arrivals) or queue:
        # Добираем прибытия до текущего t, если очередь пуста — прыгаем вперёд.
        if not queue and ai < len(arrivals):
            t = max(t, arrivals[ai])

        while ai < len(arrivals) and arrivals[ai] <= t:
            tokens = _sample_tokens(cfg.mean_tokens, rng)
            queue.append((arrivals[ai], tokens))
            ai += 1
        flush_drops(t)
        if not queue:
            if ai >= len(arrivals):
                break
            continue

        # Ждём наполнения пачки небольшим lookahead, но не дольше timeout головы.
        batch_deadline = queue[0][0] + min(0.15, cfg.timeout_s)
        while (
            len(queue) < cfg.max_slots
            and ai < len(arrivals)
            and arrivals[ai] <= batch_deadline
        ):
            tokens = _sample_tokens(cfg.mean_tokens, rng)
            queue.append((arrivals[ai], tokens))
            ai += 1
            t = max(t, arrivals[ai - 1])
            flush_drops(t)

        if not queue:
            continue

        batch = queue[: cfg.max_slots]
        queue = queue[cfg.max_slots :]
        # Static: длительность = max по членам пачки (все слоты заняты до конца самого длинного).
        max_tokens = max(tok for _, tok in batch)
        service = max_tokens * cfg.token_time_s
        start = t
        end = start + service
        busy_integral += cfg.max_slots * service
        for arrival, _tokens in batch:
            latencies.append(end - arrival)
            completed += 1
        last_t = end
        t = end

    wall = max(last_t, cfg.duration_s)
    return SlotMetrics(
        throughput_sessions=completed / wall,
        completed=completed,
        dropped=dropped,
        drop_rate=dropped / max(1, completed + dropped),
        mean_latency_s=(sum(latencies) / completed) if completed else 0.0,
        mean_occupancy=busy_integral / (wall * cfg.max_slots),
    )


def simulate_continuous(cfg: SlotConfig) -> SlotMetrics:
    """Слот освобождается сразу после конца своей генерации."""

    rng = random.Random(cfg.seed)
    arrivals = _poisson_arrivals(cfg.arrival_rate, cfg.duration_s, rng)
    queue: list[tuple[float, int]] = []
    # active: finish_time, arrival
    active: list[tuple[float, float]] = []
    t = 0.0
    ai = 0
    completed = 0
    dropped = 0
    latencies: list[float] = []
    busy_integral = 0.0
    last_t = 0.0

    def flush_drops(now: float) -> None:
        nonlocal dropped, queue
        kept: list[tuple[float, int]] = []
        for arrival, tokens in queue:
            if now - arrival > cfg.timeout_s:
                dropped += 1
            else:
                kept.append((arrival, tokens))
        queue = kept

    def admit() -> None:
        nonlocal queue, active
        while queue and len(active) < cfg.max_slots:
            arrival, tokens = queue.pop(0)
            finish = t + tokens * cfg.token_time_s
            active.append((finish, arrival))

    while ai < len(arrivals) or queue or active:
        next_arrival = arrivals[ai] if ai < len(arrivals) else None
        next_finish = min((f for f, _ in active), default=None)

        candidates = [x for x in (next_arrival, next_finish) if x is not None]
        if not candidates:
            break
        t_next = min(candidates)

        # Интеграл занятости на интервале.
        dt = t_next - t
        if dt > 0:
            busy_integral += len(active) * dt
            t = t_next

        if next_finish is not None and abs(t - next_finish) < 1e-12:
            still: list[tuple[float, float]] = []
            for finish, arrival in active:
                if abs(finish - t) < 1e-12:
                    latencies.append(t - arrival)
                    completed += 1
                    last_t = t
                else:
                    still.append((finish, arrival))
            active = still

        if next_arrival is not None and abs(t - next_arrival) < 1e-12:
            tokens = _sample_tokens(cfg.mean_tokens, rng)
            queue.append((arrivals[ai], tokens))
            ai += 1

        flush_drops(t)
        admit()

    wall = max(last_t, cfg.duration_s, t)
    return SlotMetrics(
        throughput_sessions=completed / wall,
        completed=completed,
        dropped=dropped,
        drop_rate=dropped / max(1, completed + dropped),
        mean_latency_s=(sum(latencies) / completed) if completed else 0.0,
        mean_occupancy=busy_integral / (wall * cfg.max_slots),
    )


def compare_slot_schedulers(cfg: SlotConfig | None = None) -> dict[str, SlotMetrics]:
    cfg = cfg or SlotConfig()
    return {
        "static_batch": simulate_static_batch(cfg),
        "continuous": simulate_continuous(cfg),
    }


def main() -> None:
    results = compare_slot_schedulers()
    for name, m in results.items():
        print(
            f"{name:14s}  sessions/s={m.throughput_sessions:5.2f}  "
            f"drop={m.drop_rate:5.1%}  mean_lat={m.mean_latency_s:5.2f}s  "
            f"occupancy={m.mean_occupancy:4.1%}"
        )


if __name__ == "__main__":
    main()
