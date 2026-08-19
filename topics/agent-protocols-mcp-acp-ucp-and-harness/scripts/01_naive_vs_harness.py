"""Сравнение голого агента и agent harness на задаче «upvote».

Скрипт показывает, почему обвязка важнее «промптить сильнее»:
- наивный агент кликает upvote на странице логина и врёт, что успех;
- harness логинит секретами вне промпта, режет бесконечный цикл
  и проверяет side effect, а не слова модели.

Ожидаемое поведение:
- у naive claimed_success=True при actually_done=False (ложь);
- у harness actually_done=True и claimed_success совпадает с фактом;
- доля правды (truthfulness) и доля реальных успехов у harness выше.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Page(str, Enum):
    LOGIN = "login"
    HOME = "home"


@dataclass
class World:
    """Минимальный «браузер»: страница, логин и флаг upvote."""

    page: Page = Page.LOGIN
    logged_in: bool = False
    upvoted: bool = False


class Action(str, Enum):
    CLICK_UPVOTE = "click_upvote"
    FILL_LOGIN = "fill_login"
    STOP = "stop"


@dataclass(frozen=True)
class ToolEvent:
    action: Action
    page_before: Page
    logged_in_before: bool
    upvoted_after: bool
    note: str = ""


@dataclass
class RunResult:
    claimed_success: bool
    actually_done: bool
    steps: int
    killed_by_guardrail: bool
    events: list[ToolEvent] = field(default_factory=list)

    @property
    def truthful(self) -> bool:
        return self.claimed_success == self.actually_done


def apply_action(world: World, action: Action, *, has_secrets: bool) -> str:
    """Применить действие к миру. Секреты доступны только harness, не агенту."""

    if action is Action.STOP:
        return "stop"
    if action is Action.FILL_LOGIN:
        if not has_secrets:
            return "login_failed_no_secrets"
        world.logged_in = True
        world.page = Page.HOME
        return "logged_in"
    if action is Action.CLICK_UPVOTE:
        if world.page is Page.LOGIN or not world.logged_in:
            return "upvote_on_login_wall"
        world.upvoted = True
        return "upvoted"
    raise ValueError(f"unknown action: {action}")


def naive_agent_policy(world: World, *, already_attempted: bool) -> Action:
    """Наивная политика: один клик upvote, затем «готово», даже если это login wall."""

    if world.upvoted or already_attempted:
        return Action.STOP
    return Action.CLICK_UPVOTE


def harness_agent_policy(world: World) -> Action:
    """Политика с учётом страницы: сначала логин, потом upvote."""

    if world.upvoted:
        return Action.STOP
    if world.page is Page.LOGIN or not world.logged_in:
        return Action.FILL_LOGIN
    return Action.CLICK_UPVOTE


def verify_upvote(world: World, events: list[ToolEvent]) -> bool:
    """Детерминированная проверка side effect, а не самоотчёт модели."""

    if not world.upvoted or not world.logged_in:
        return False
    return any(
        event.action is Action.CLICK_UPVOTE
        and event.logged_in_before
        and event.upvoted_after
        for event in events
    )


def run_agent(
    *,
    harnessed: bool,
    max_steps: int = 6,
    seed_world: World | None = None,
) -> RunResult:
    """Прогон агента. Harness: секреты, max_steps и verify."""

    world = seed_world if seed_world is not None else World()
    events: list[ToolEvent] = []
    has_secrets = harnessed
    killed = False

    for _step in range(1, max_steps + 1):
        if harnessed:
            action = harness_agent_policy(world)
        else:
            action = naive_agent_policy(world, already_attempted=bool(events))
        if action is Action.STOP:
            break
        page_before = world.page
        logged_in_before = world.logged_in
        note = apply_action(world, action, has_secrets=has_secrets)
        events.append(
            ToolEvent(
                action=action,
                page_before=page_before,
                logged_in_before=logged_in_before,
                upvoted_after=world.upvoted,
                note=note,
            )
        )
    else:
        killed = not world.upvoted

    actually_done = world.upvoted
    if harnessed:
        claimed_success = verify_upvote(world, events)
    else:
        claimed_success = True

    return RunResult(
        claimed_success=claimed_success,
        actually_done=actually_done,
        steps=len(events),
        killed_by_guardrail=killed,
        events=events,
    )


def compare_naive_vs_harness(repeats: int = 20) -> dict[str, float]:
    """Сводные метрики: доля реальных успехов и доля правдивых отчётов."""

    naive_done = 0
    naive_truth = 0
    harness_done = 0
    harness_truth = 0
    for _ in range(repeats):
        naive = run_agent(harnessed=False)
        harness = run_agent(harnessed=True)
        naive_done += int(naive.actually_done)
        naive_truth += int(naive.truthful)
        harness_done += int(harness.actually_done)
        harness_truth += int(harness.truthful)
    n = float(repeats)
    return {
        "naive_success_rate": naive_done / n,
        "naive_truthfulness": naive_truth / n,
        "harness_success_rate": harness_done / n,
        "harness_truthfulness": harness_truth / n,
    }


def main() -> None:
    metrics = compare_naive_vs_harness()
    naive = run_agent(harnessed=False)
    harness = run_agent(harnessed=True)
    print("naive:", naive)
    print("harness:", harness)
    print("metrics:", metrics)


if __name__ == "__main__":
    main()
