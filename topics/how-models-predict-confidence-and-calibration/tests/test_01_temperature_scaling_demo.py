from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import torch


def _load_demo_module():
    repo_root = Path(__file__).resolve().parents[3]
    script_path = (
        repo_root
        / "topics"
        / "how-models-predict-confidence-and-calibration"
        / "scripts"
        / "01_temperature_scaling_demo.py"
    )
    spec = spec_from_file_location("temperature_scaling_demo", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load temperature scaling demo module")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_temperature_scaler_rejects_non_positive_temperature() -> None:
    module = _load_demo_module()
    logits = torch.tensor([[1.0, 2.0]])
    scaler = module.TemperatureScaler(temperature=0.0)
    try:
        scaler.scale(logits)
        assert False, "Expected ValueError for non-positive temperature"
    except ValueError:
        assert True


def test_probabilities_sum_to_one() -> None:
    module = _load_demo_module()
    logits = torch.tensor([[2.5, 1.0, -1.0]])
    probs = module.calibrated_probs(logits, temperature=2.0)
    assert torch.allclose(probs.sum(dim=-1), torch.tensor([1.0]), atol=1e-6)
