"""
Что показывает скрипт:
  бюджет битрейта GOP. All-intra (каждый кадр как I, аналог MJPEG) против
  структуры IPPP с компенсацией движения и кодированием residual — как в H.264/H.265.
  «HEVC-like» здесь — более крупный блок (16×16) на том же QP: меньше служебных
  коэффициентов на гладких областях, как идея CTU.

Ожидаемое поведение:
  - I-кадр дороже P-кадра (больше ненулевых коэффициентов);
  - средний битрейт IPPP заметно ниже all-intra при близком PSNR;
  - HEVC-like GOP дешевле H.264-like GOP на гладком синтетическом ролике.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch


def _load_named(filename: str, module_name: str):
    path = Path(__file__).resolve().with_name(filename)
    spec = spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"не удалось загрузить {filename}")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_intra():
    return _load_named("01_block_dct_quantization.py", "intra_codec_demo")


def _load_motion():
    return _load_named("02_motion_compensation_residual.py", "motion_codec_demo")


_intra = _load_intra()
_motion = _load_motion()


@dataclass(frozen=True)
class GopConfig:
    height: int = 64
    width: int = 64
    n_frames: int = 8
    shift_y: int = 1
    shift_x: int = 2
    qp: float = 10.0
    seed: int = 21


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = float(((a - b) ** 2).mean().item())
    return 10.0 * math.log10((255.0**2) / max(mse, 1e-12))


def make_clip(cfg: GopConfig) -> list[torch.Tensor]:
    intra_cfg = _intra.IntraCodecConfig(height=cfg.height, width=cfg.width, seed=cfg.seed)
    base = _intra.make_smooth_frame(intra_cfg)
    frames = [base]
    for t in range(1, cfg.n_frames):
        frames.append(_motion.shift_frame(base, dy=cfg.shift_y * t, dx=cfg.shift_x * t))
    return frames


def encode_frame_intra(frame: torch.Tensor, qp: float, block: int) -> tuple[torch.Tensor, int]:
    stats = _intra.encode_intra(frame, qp=qp, block=block)
    recon = stats["recon"]
    assert isinstance(recon, torch.Tensor)
    return recon, int(stats["nonzero"])


def encode_p_frame(
    frame: torch.Tensor,
    ref_recon: torch.Tensor,
    qp: float,
    block: int,
    radius: int = 4,
) -> tuple[torch.Tensor, int]:
    dy, dx, _ = _motion.search_global_motion(ref_recon, frame, radius)
    pred = _motion.shift_frame(ref_recon, dy, dx)
    residual = frame - pred
    c = _intra.orthonormal_dct_matrix(block, dtype=frame.dtype)
    recon_res = torch.zeros_like(residual)
    nonzero = 0
    for y, x, blk in _intra.iter_blocks(residual, block):
        levels = torch.round(_intra.dct2(blk, c) / qp)
        nonzero += int((levels != 0).sum().item())
        recon_res[y : y + block, x : x + block] = _intra.idct2(levels * qp, c)
    recon = (pred + recon_res).clamp(0.0, 255.0)
    # вектор движения тоже «стоит» бит — фиксированная надбавка на кадр
    nonzero += 2
    return recon, nonzero


def encode_all_intra(frames: list[torch.Tensor], qp: float, block: int) -> dict[str, float]:
    bits = 0
    psnrs: list[float] = []
    for fr in frames:
        rec, nz = encode_frame_intra(fr, qp=qp, block=block)
        bits += nz
        psnrs.append(psnr(fr, rec))
    return {
        "total_bits_proxy": float(bits),
        "mean_bits_per_frame": bits / len(frames),
        "mean_psnr": float(sum(psnrs) / len(psnrs)),
    }


def encode_ippp(frames: list[torch.Tensor], qp: float, block: int) -> dict[str, float]:
    i_rec, i_bits = encode_frame_intra(frames[0], qp=qp, block=block)
    bits = [i_bits]
    psnrs = [psnr(frames[0], i_rec)]
    ref = i_rec
    p_bits: list[int] = []
    for fr in frames[1:]:
        rec, nz = encode_p_frame(fr, ref, qp=qp, block=block)
        bits.append(nz)
        p_bits.append(nz)
        psnrs.append(psnr(fr, rec))
        ref = rec
    return {
        "total_bits_proxy": float(sum(bits)),
        "mean_bits_per_frame": sum(bits) / len(frames),
        "i_bits": float(i_bits),
        "mean_p_bits": float(sum(p_bits) / max(len(p_bits), 1)),
        "mean_psnr": float(sum(psnrs) / len(psnrs)),
        "i_over_p": i_bits / max(sum(p_bits) / max(len(p_bits), 1), 1.0),
    }


def compare_codec_generations(cfg: GopConfig | None = None) -> dict[str, float]:
    cfg = cfg or GopConfig()
    frames = make_clip(cfg)
    all_intra = encode_all_intra(frames, qp=cfg.qp, block=8)
    avc = encode_ippp(frames, qp=cfg.qp, block=8)
    hevc = encode_ippp(frames, qp=cfg.qp, block=16)
    return {
        "all_intra_bits": all_intra["total_bits_proxy"],
        "all_intra_psnr": all_intra["mean_psnr"],
        "avc_bits": avc["total_bits_proxy"],
        "avc_psnr": avc["mean_psnr"],
        "avc_i_bits": avc["i_bits"],
        "avc_p_bits": avc["mean_p_bits"],
        "avc_i_over_p": avc["i_over_p"],
        "hevc_bits": hevc["total_bits_proxy"],
        "hevc_psnr": hevc["mean_psnr"],
        "ippp_vs_intra": all_intra["total_bits_proxy"] / max(avc["total_bits_proxy"], 1.0),
        "hevc_vs_avc": avc["total_bits_proxy"] / max(hevc["total_bits_proxy"], 1.0),
    }


if __name__ == "__main__":
    stats = compare_codec_generations()
    print("GOP IPPP дешевле all-intra; крупный блок (HEVC-like) ещё чуть дешевле")
    print(
        f"all-intra: bits={stats['all_intra_bits']:.0f}  PSNR={stats['all_intra_psnr']:.2f} dB"
    )
    print(
        f"H.264-like 8×8 IPPP: bits={stats['avc_bits']:.0f}  "
        f"PSNR={stats['avc_psnr']:.2f} dB  I/P={stats['avc_i_over_p']:.2f}×"
    )
    print(
        f"HEVC-like 16×16 IPPP: bits={stats['hevc_bits']:.0f}  "
        f"PSNR={stats['hevc_psnr']:.2f} dB"
    )
    print(
        f"экономия IPPP vs intra ×{stats['ippp_vs_intra']:.2f}; "
        f"HEVC-like vs AVC-like ×{stats['hevc_vs_avc']:.2f}"
    )
