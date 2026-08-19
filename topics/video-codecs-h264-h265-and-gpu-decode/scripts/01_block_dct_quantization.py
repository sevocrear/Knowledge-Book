"""
Что показывает скрипт:
  внутрикадровое (intra) сжатие в духе JPEG/H.264: блок 8×8 → ортонормированное
  DCT-II → равномерное квантование. Это ядро I-кадра.

Ожидаемое поведение:
  - энергия DCT сосредоточена в низких частотах;
  - чем больше шаг квантования QP, тем больше нулевых коэффициентов
    (прокси битрейта падает) и тем ниже PSNR реконструкции.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class IntraCodecConfig:
    block: int = 8
    height: int = 64
    width: int = 64
    seed: int = 7


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)


def orthonormal_dct_matrix(n: int, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Матрица ортонормированного DCT-II размера n×n."""
    k = torch.arange(n, dtype=dtype)
    i = torch.arange(n, dtype=dtype)
    angles = math.pi * (2.0 * i + 1.0).unsqueeze(0) * k.unsqueeze(1) / (2.0 * n)
    c = torch.cos(angles).clone()
    c[0] *= math.sqrt(1.0 / n)
    c[1:] *= math.sqrt(2.0 / n)
    return c


def dct2(block: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return c @ block @ c.T


def idct2(coeff: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return c.T @ coeff @ c


def make_smooth_frame(cfg: IntraCodecConfig) -> torch.Tensor:
    """Кадр с градиентом и мягкими пятнами — типичный « intra-friendly» контент."""
    yy = torch.linspace(0.0, 1.0, cfg.height).unsqueeze(1)
    xx = torch.linspace(0.0, 1.0, cfg.width).unsqueeze(0)
    base = 40.0 + 80.0 * xx + 50.0 * yy
    blob = 70.0 * torch.exp(-((xx - 0.35) ** 2 + (yy - 0.4) ** 2) / 0.04)
    edge = 35.0 * torch.sigmoid(18.0 * (xx - 0.7))
    g = torch.Generator().manual_seed(cfg.seed)
    noise = 2.5 * torch.randn(cfg.height, cfg.width, generator=g)
    return (base + blob + edge + noise).clamp(0.0, 255.0).to(torch.float64)


def iter_blocks(frame: torch.Tensor, block: int) -> list[tuple[int, int, torch.Tensor]]:
    h, w = frame.shape
    out: list[tuple[int, int, torch.Tensor]] = []
    for y in range(0, h, block):
        for x in range(0, w, block):
            out.append((y, x, frame[y : y + block, x : x + block]))
    return out


def low_frequency_energy_fraction(coeff: torch.Tensor, low: int = 3) -> float:
    """Доля энергии в верхнем левом low×low (DC + низкие частоты)."""
    total = float((coeff**2).sum().item())
    if total <= 0.0:
        return 0.0
    low_e = float((coeff[:low, :low] ** 2).sum().item())
    return low_e / total


def encode_intra(frame: torch.Tensor, qp: float, block: int) -> dict[str, float | torch.Tensor]:
    """DCT + квантование всех блоков. bits_proxy = число ненулевых уровней."""
    c = orthonormal_dct_matrix(block, dtype=frame.dtype)
    recon = torch.zeros_like(frame)
    nonzero = 0
    total_coeff = 0
    lf_fracs: list[float] = []
    for y, x, blk in iter_blocks(frame, block):
        coeff = dct2(blk, c)
        lf_fracs.append(low_frequency_energy_fraction(coeff))
        levels = torch.round(coeff / qp)
        nonzero += int((levels != 0).sum().item())
        total_coeff += levels.numel()
        rec = idct2(levels * qp, c)
        recon[y : y + block, x : x + block] = rec
    recon = recon.clamp(0.0, 255.0)
    mse = float(((frame - recon) ** 2).mean().item())
    psnr = 10.0 * math.log10((255.0**2) / max(mse, 1e-12))
    return {
        "recon": recon,
        "nonzero": float(nonzero),
        "total_coeff": float(total_coeff),
        "sparsity": 1.0 - nonzero / total_coeff,
        "psnr": psnr,
        "mean_lf_energy": float(sum(lf_fracs) / len(lf_fracs)),
        "compression_proxy": total_coeff / max(nonzero, 1),
    }


def compare_qp_ladder(frame: torch.Tensor, qps: tuple[float, ...] = (4.0, 8.0, 16.0), block: int = 8) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for qp in qps:
        stats = encode_intra(frame, qp=qp, block=block)
        rows.append(
            {
                "qp": qp,
                "psnr": float(stats["psnr"]),
                "sparsity": float(stats["sparsity"]),
                "compression_proxy": float(stats["compression_proxy"]),
                "mean_lf_energy": float(stats["mean_lf_energy"]),
                "nonzero": float(stats["nonzero"]),
            }
        )
    return rows


if __name__ == "__main__":
    cfg = IntraCodecConfig()
    set_seed(cfg.seed)
    frame = make_smooth_frame(cfg)
    print("intra DCT+quant: PSNR падает, нулей становится больше")
    for row in compare_qp_ladder(frame):
        print(
            f"QP={row['qp']:4.1f}  PSNR={row['psnr']:6.2f} dB  "
            f"sparsity={row['sparsity']:.3f}  "
            f"compress≈{row['compression_proxy']:.2f}×  "
            f"LF-energy={row['mean_lf_energy']:.3f}"
        )
