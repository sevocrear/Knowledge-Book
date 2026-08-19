import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load():
    repo = Path(__file__).resolve().parents[3]
    path = (
        repo
        / "topics"
        / "video-codecs-h264-h265-and-gpu-decode"
        / "scripts"
        / "03_gop_bitrate_budget.py"
    )
    spec = spec_from_file_location("gop_codec_demo", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load GOP bitrate demo")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_i_frame_is_more_expensive_than_p_frame() -> None:
    m = _load()
    stats = m.compare_codec_generations()
    assert stats["avc_i_over_p"] > 2.0
    assert stats["avc_i_bits"] > stats["avc_p_bits"]


def test_inter_gop_beats_all_intra_bitrate() -> None:
    m = _load()
    stats = m.compare_codec_generations()
    assert stats["ippp_vs_intra"] > 1.8
    # качество не должно развалиться: PSNR IPPP близок к all-intra
    assert stats["avc_psnr"] >= stats["all_intra_psnr"] - 3.0


def test_hevc_like_larger_blocks_save_bits_on_smooth_content() -> None:
    m = _load()
    stats = m.compare_codec_generations()
    assert stats["hevc_vs_avc"] > 1.05
    assert stats["hevc_psnr"] >= stats["avc_psnr"] - 4.0
