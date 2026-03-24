"""Temperature scaling demo for confidence calibration.

This script shows how temperature changes model confidence distribution
without changing class ranking. Expected result: larger temperature values
produce softer probabilities and lower overconfidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TemperatureScaler:
    """Apply a scalar temperature to logits."""

    temperature: float = 1.0

    def scale(self, logits: torch.Tensor) -> torch.Tensor:
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        return logits / self.temperature


def calibrated_probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    scaler = TemperatureScaler(temperature=temperature)
    scaled = scaler.scale(logits)
    return torch.softmax(scaled, dim=-1)


if __name__ == "__main__":
    sample_logits = torch.tensor([[3.2, 1.1, -0.6], [1.8, 1.7, 1.6]])
    for temp in (1.0, 2.0, 4.0):
        probs = calibrated_probs(sample_logits, temp)
        print(f"temperature={temp}")
        print(probs)
