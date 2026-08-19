"""
Что показывает скрипт:
  межкадровое (inter) предсказание: кадр t почти равен сдвинутому кадру t-1.
  Кодек ищет вектор движения, вычитает предсказание и кодирует остаток (residual).

Ожидаемое поведение:
  энергия остатка после компенсации движения заметно меньше, чем у сырой
  разницы кадров; ненулевых DCT-коэффициентов остатка тоже меньше.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch


def _load_intra():
    path = Path(__file__).resolve().with_name("01_block_dct_quantization.py")
    spec = spec_from_file_location("intra_codec_demo", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("не удалось загрузить 01_block_dct_quantization.py")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_intra = _load_intra()


@dataclass(frozen=True)
class MotionDemoConfig:
    height: int = 64
    width: int = 64
    shift_y: int = 2
    shift_x: int = 3
    search_radius: int = 4
    qp: float = 8.0
    block: int = 8
    seed: int = 11
    noise_std: float = 1.2


def shift_frame(frame: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """Циклический сдвиг — удобная модель глобального движения сцены."""
    return torch.roll(frame, shifts=(dy, dx), dims=(0, 1))


def make_pair(cfg: MotionDemoConfig) -> tuple[torch.Tensor, torch.Tensor]:
    intra_cfg = _intra.IntraCodecConfig(height=cfg.height, width=cfg.width, seed=cfg.seed)
    ref = _intra.make_smooth_frame(intra_cfg)
    cur = shift_frame(ref, cfg.shift_y, cfg.shift_x)
    g = torch.Generator().manual_seed(cfg.seed + 99)
    noise = cfg.noise_std * torch.randn(cfg.height, cfg.width, generator=g, dtype=ref.dtype)
    cur = (cur + noise).clamp(0.0, 255.0)
    return ref, cur


def energy(x: torch.Tensor) -> float:
    return float((x.double() ** 2).mean().item())


def sad(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().sum().item())


def search_global_motion(ref: torch.Tensor, cur: torch.Tensor, radius: int) -> tuple[int, int, float]:
    """Полный перебор целочисленных (dy, dx) в окне — упрощённый analog ME в H.264."""
    best = (0, 0)
    best_sad = sad(cur, ref)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            pred = shift_frame(ref, dy, dx)
            s = sad(cur, pred)
            if s < best_sad:
                best_sad = s
                best = (dy, dx)
    return best[0], best[1], best_sad


def quantized_nonzero(residual: torch.Tensor, qp: float, block: int) -> int:
    c = _intra.orthonormal_dct_matrix(block, dtype=residual.dtype)
    nonzero = 0
    for _, _, blk in _intra.iter_blocks(residual, block):
        levels = torch.round(_intra.dct2(blk, c) / qp)
        nonzero += int((levels != 0).sum().item())
    return nonzero


def evaluate_motion_compensation(cfg: MotionDemoConfig | None = None) -> dict[str, float]:
    cfg = cfg or MotionDemoConfig()
    ref, cur = make_pair(cfg)
    raw = cur - ref
    dy, dx, _ = search_global_motion(ref, cur, cfg.search_radius)
    pred = shift_frame(ref, dy, dx)
    residual = cur - pred

    raw_nz = quantized_nonzero(raw, cfg.qp, cfg.block)
    res_nz = quantized_nonzero(residual, cfg.qp, cfg.block)
    raw_e = energy(raw)
    res_e = energy(residual)
    return {
        "found_dy": float(dy),
        "found_dx": float(dx),
        "true_dy": float(cfg.shift_y),
        "true_dx": float(cfg.shift_x),
        "raw_energy": raw_e,
        "residual_energy": res_e,
        "energy_ratio": raw_e / max(res_e, 1e-12),
        "raw_nonzero": float(raw_nz),
        "residual_nonzero": float(res_nz),
        "nonzero_ratio": raw_nz / max(res_nz, 1),
        "motion_found": float(dy == cfg.shift_y and dx == cfg.shift_x),
    }


if __name__ == "__main__":
    stats = evaluate_motion_compensation()
    print("компенсация движения: остаток гораздо меньше сырой разницы кадров")
    print(
        f"найденный MV=({int(stats['found_dy'])},{int(stats['found_dx'])})  "
        f"истинный=({int(stats['true_dy'])},{int(stats['true_dx'])})"
    )
    print(
        f"энергия raw={stats['raw_energy']:.3f}  residual={stats['residual_energy']:.3e}  "
        f"выигрыш ×{stats['energy_ratio']:.1f}"
    )
    print(
        f"ненулевые DCT: raw={int(stats['raw_nonzero'])}  residual={int(stats['residual_nonzero'])}  "
        f"выигрыш ×{stats['nonzero_ratio']:.1f}"
    )
