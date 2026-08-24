import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch


def _load():
    repo = Path(__file__).resolve().parents[3]
    path = (
        repo
        / "topics"
        / "video-codecs-h264-h265-and-gpu-decode"
        / "scripts"
        / "01_block_dct_quantization.py"
    )
    spec = spec_from_file_location("intra_codec_demo", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load intra codec demo")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dct_is_orthonormal_and_invertible() -> None:
    """Ядро intra-кодека: ортонормированное DCT-II обратимо и сохраняет энергию."""

    m = _load()
    n = 8
    c = m.orthonormal_dct_matrix(n)
    eye = torch.eye(n, dtype=c.dtype)
    assert torch.allclose(c @ c.T, eye, atol=1e-12)
    assert torch.allclose(c.T @ c, eye, atol=1e-12)

    g = torch.Generator().manual_seed(0)
    block = torch.randn(n, n, dtype=torch.float64, generator=g)
    coeff = m.dct2(block, c)
    recon = m.idct2(coeff, c)
    assert torch.allclose(recon, block, atol=1e-12)
    energy_x = float((block**2).sum().item())
    energy_y = float((coeff**2).sum().item())
    assert abs(energy_x - energy_y) / max(energy_x, 1e-12) < 1e-12


def test_dct_energy_lives_in_low_frequencies() -> None:
    m = _load()
    frame = m.make_smooth_frame(m.IntraCodecConfig())
    rows = m.compare_qp_ladder(frame, qps=(8.0,))
    assert rows[0]["mean_lf_energy"] > 0.85


def test_higher_qp_increases_sparsity_and_drops_psnr() -> None:
    m = _load()
    frame = m.make_smooth_frame(m.IntraCodecConfig())
    low, high = m.compare_qp_ladder(frame, qps=(4.0, 16.0))
    assert high["sparsity"] > low["sparsity"] + 0.15
    assert high["compression_proxy"] > low["compression_proxy"] + 0.4
    assert low["psnr"] > high["psnr"] + 4.0


def test_reconstruction_stays_in_valid_range() -> None:
    m = _load()
    frame = m.make_smooth_frame(m.IntraCodecConfig())
    stats = m.encode_intra(frame, qp=8.0, block=8)
    recon = stats["recon"]
    assert float(recon.min()) >= 0.0
    assert float(recon.max()) <= 255.0
    assert stats["psnr"] > 30.0
