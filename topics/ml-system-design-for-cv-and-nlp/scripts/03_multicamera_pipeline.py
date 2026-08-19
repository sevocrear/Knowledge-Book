"""Мультикамерный пайплайн: 10 vs 50 камер на одном GPU.

Наивная схема кладёт все кадры в общий FIFO: при перегрузке очередь растёт,
и детектор смотрит в прошлое (высокая stale-латентность).

Эффективная схема:
- на камеру хранится только последний кадр (drop-oldest);
- GPU забирает пачку кадров с разных камер.

Ожидаемое поведение при 50 камерах × 30 FPS и GPU ~80 кадр/с:
- FIFO даёт огромную среднюю устарелость кадра;
- latest-frame + batching держит устарелость ограниченной
  и обрабатывает больше уникальных камер в единицу времени.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CameraPipelineConfig:
    n_cameras: int
    camera_fps: float = 30.0
    duration_s: float = 2.0
    gpu_overhead_s: float = 0.004
    gpu_per_item_s: float = 0.0015
    max_batch: int = 8


@dataclass(frozen=True)
class CameraPipelineMetrics:
    incoming_fps: float
    processed_frames: int
    processed_fps: float
    unique_cameras_served: int
    mean_staleness_s: float
    p95_staleness_s: float
    drop_rate: float


def gpu_batch_seconds(batch_size: int, overhead_s: float, per_item_s: float) -> float:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    return overhead_s + per_item_s * batch_size


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return math.inf
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[idx]


def _frame_schedule(cfg: CameraPipelineConfig) -> list[tuple[float, int]]:
    """Все кадры (время захвата, id камеры), отсортированные по времени."""

    dt = 1.0 / cfg.camera_fps
    frames: list[tuple[float, int]] = []
    n_ticks = int(cfg.duration_s * cfg.camera_fps)
    for tick in range(n_ticks):
        t = tick * dt
        for cam in range(cfg.n_cameras):
            # Небольшой сдвиг, чтобы камеры не били в GPU абсолютно синхронно.
            frames.append((t + cam * dt / max(cfg.n_cameras, 1) * 0.05, cam))
    frames.sort(key=lambda item: item[0])
    return frames


def simulate_naive_fifo(cfg: CameraPipelineConfig) -> CameraPipelineMetrics:
    """Общий FIFO: обрабатываем кадры в порядке поступления, batch=1."""

    frames = _frame_schedule(cfg)
    gpu_free = 0.0
    staleness: list[float] = []
    served: set[int] = set()
    dropped = 0
    service = gpu_batch_seconds(1, cfg.gpu_overhead_s, cfg.gpu_per_item_s)
    # Живое видео: если кадр уже ждал больше 1/fps, он «протух» относительно следующей волны,
    # но FIFO всё равно его обработает. Считаем drop только если старт уехал за duration.
    horizon = cfg.duration_s + 0.25
    for capture_t, cam in frames:
        start = max(capture_t, gpu_free)
        if start > horizon:
            dropped += 1
            continue
        finish = start + service
        gpu_free = finish
        staleness.append(finish - capture_t)
        served.add(cam)
    incoming = len(frames)
    processed = len(staleness)
    return CameraPipelineMetrics(
        incoming_fps=cfg.n_cameras * cfg.camera_fps,
        processed_frames=processed,
        processed_fps=processed / cfg.duration_s,
        unique_cameras_served=len(served),
        mean_staleness_s=sum(staleness) / processed if processed else math.inf,
        p95_staleness_s=_percentile(staleness, 0.95),
        drop_rate=dropped / incoming if incoming else 0.0,
    )


def simulate_latest_frame_batch(cfg: CameraPipelineConfig) -> CameraPipelineMetrics:
    """На камеру — только последний кадр; GPU забирает пачку."""

    frames = _frame_schedule(cfg)
    pending: dict[int, float] = {}
    gpu_free = 0.0
    staleness: list[float] = []
    served: set[int] = set()
    produced = 0
    dropped_overwritten = 0
    i = 0
    t = 0.0
    end_t = cfg.duration_s

    while t <= end_t or pending:
        while i < len(frames) and frames[i][0] <= t:
            capture_t, cam = frames[i]
            if cam in pending:
                dropped_overwritten += 1
            pending[cam] = capture_t
            produced += 1
            i += 1

        if pending and t >= gpu_free:
            cams = list(pending.keys())[: cfg.max_batch]
            batch = [(cam, pending.pop(cam)) for cam in cams]
            service = gpu_batch_seconds(len(batch), cfg.gpu_overhead_s, cfg.gpu_per_item_s)
            finish = t + service
            gpu_free = finish
            for cam, capture_t in batch:
                staleness.append(finish - capture_t)
                served.add(cam)
            t = finish
            continue

        if i < len(frames):
            t = max(t, frames[i][0])
            if gpu_free > t and pending:
                t = gpu_free
            continue
        if pending:
            t = max(t, gpu_free)
            if t > end_t + 0.5:
                break
            continue
        break

    incoming = len(frames)
    processed = len(staleness)
    dropped = incoming - processed
    return CameraPipelineMetrics(
        incoming_fps=cfg.n_cameras * cfg.camera_fps,
        processed_frames=processed,
        processed_fps=processed / cfg.duration_s,
        unique_cameras_served=len(served),
        mean_staleness_s=sum(staleness) / processed if processed else math.inf,
        p95_staleness_s=_percentile(staleness, 0.95),
        drop_rate=dropped / incoming if incoming else 0.0,
    )


def compare_cameras(
    n_cameras: int,
    duration_s: float = 2.0,
) -> dict[str, CameraPipelineMetrics]:
    cfg = CameraPipelineConfig(n_cameras=n_cameras, duration_s=duration_s)
    return {
        "naive_fifo": simulate_naive_fifo(cfg),
        "latest_frame_batch": simulate_latest_frame_batch(cfg),
    }


if __name__ == "__main__":
    for n_cam in (10, 50):
        print(f"##### {n_cam} cameras #####")
        for name, metrics in compare_cameras(n_cam).items():
            print(f"=== {name} ===")
            print(
                f"incoming={metrics.incoming_fps:.0f} fps  processed={metrics.processed_fps:.1f} fps  "
                f"cameras_served={metrics.unique_cameras_served}"
            )
            print(
                f"mean_stale={metrics.mean_staleness_s*1000:.1f} ms  "
                f"p95_stale={metrics.p95_staleness_s*1000:.1f} ms  drop_rate={metrics.drop_rate:.3f}"
            )
            print()
