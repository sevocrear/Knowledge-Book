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
        / "02_motion_compensation_residual.py"
    )
    spec = spec_from_file_location("motion_codec_demo", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load motion compensation demo")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_search_recovers_true_translation() -> None:
    m = _load()
    stats = m.evaluate_motion_compensation()
    assert stats["motion_found"] == 1.0


def test_residual_energy_much_smaller_than_raw_difference() -> None:
    m = _load()
    stats = m.evaluate_motion_compensation()
    assert stats["energy_ratio"] > 50.0
    assert stats["residual_energy"] < 8.0


def test_residual_needs_fewer_dct_coefficients() -> None:
    m = _load()
    stats = m.evaluate_motion_compensation()
    assert stats["nonzero_ratio"] > 5.0
    assert stats["residual_nonzero"] < stats["raw_nonzero"]
