from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "01_naive_vs_harness.py"
    spec = importlib.util.spec_from_file_location("naive_vs_harness", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_naive_agent_lies_about_upvote() -> None:
    """Сигнал темы: без verify агент кликает на login wall и врёт про успех."""

    module = _load_module()
    result = module.run_agent(harnessed=False)
    assert result.claimed_success is True
    assert result.actually_done is False
    assert result.truthful is False
    assert result.events[0].note == "upvote_on_login_wall"


def test_harness_completes_and_reports_truth() -> None:
    """Harness логинит секретами вне промпта и проверяет side effect."""

    module = _load_module()
    result = module.run_agent(harnessed=True)
    assert result.actually_done is True
    assert result.claimed_success is True
    assert result.truthful is True
    assert result.killed_by_guardrail is False
    notes = [event.note for event in result.events]
    assert "logged_in" in notes
    assert "upvoted" in notes


def test_max_steps_guardrail_kills_unproductive_loop() -> None:
    module = _load_module()

    def stuck_policy(_world: module.World) -> module.Action:
        return module.Action.CLICK_UPVOTE

    original = module.harness_agent_policy
    module.harness_agent_policy = stuck_policy
    try:
        result = module.run_agent(harnessed=True, max_steps=3)
    finally:
        module.harness_agent_policy = original

    assert result.killed_by_guardrail is True
    assert result.actually_done is False
    assert result.claimed_success is False
    assert result.steps == 3


def test_harness_beats_naive_on_success_and_truthfulness() -> None:
    """Сводный сигнал: harness поднимает и success rate, и правдивость отчёта."""

    module = _load_module()
    metrics = module.compare_naive_vs_harness(repeats=20)
    assert metrics["naive_success_rate"] == 0.0
    assert metrics["naive_truthfulness"] == 0.0
    assert metrics["harness_success_rate"] == 1.0
    assert metrics["harness_truthfulness"] == 1.0
    assert metrics["harness_success_rate"] > metrics["naive_success_rate"] + 0.5
    assert metrics["harness_truthfulness"] > metrics["naive_truthfulness"] + 0.5
