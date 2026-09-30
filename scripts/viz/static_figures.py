#!/usr/bin/env python3
"""Статические иллюстрации для README тем (matplotlib → PNG в topics/<slug>/assets/images/).

Заменяют хотлинки на внешние картинки: все фигуры рисуются с нуля, без сторонних файлов.
Запуск из корня репозитория (matplotlib не входит в зависимости проекта):

    uv run --with matplotlib --with numpy python scripts/viz/static_figures.py [--only venn_partition,...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TOPICS = REPO_ROOT / "topics"

BLUE, ORANGE, GREEN, RED, PURPLE, GRAY = "#2f6fdb", "#e8702a", "#2e9e5b", "#d63a5c", "#7b5cc7", "#6b7280"


def _np():
    import numpy

    return numpy


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def venn_partition(out: Path) -> None:
    """Разбиение Ω на гипотезы H1..H4 и событие A: формула полной вероятности."""
    plt = _mpl()
    from matplotlib.patches import Ellipse, Rectangle

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.add_patch(Rectangle((0, 0), 8, 4, fill=False, lw=2, color="black"))
    fills = ["#dbe7fb", "#fde4d3", "#d9f2e2", "#eadff7"]
    for i in range(4):
        ax.add_patch(Rectangle((2 * i, 0), 2, 4, color=fills[i], lw=0))
        ax.plot([2 * i, 2 * i], [0, 4], color="black", lw=1.2)
        ax.text(2 * i + 1, 3.55, f"$H_{i + 1}$", ha="center", fontsize=15)
    ax.add_patch(Ellipse((4, 1.7), 6.2, 1.9, fill=True, color=RED, alpha=0.18, lw=0))
    ax.add_patch(Ellipse((4, 1.7), 6.2, 1.9, fill=False, color=RED, lw=2.2))
    ax.text(4, 1.7, "$A$", ha="center", va="center", fontsize=18, color=RED)
    for i in range(4):
        ax.text(2 * i + 1, 0.35, f"$A\\cap H_{i + 1}$", ha="center", fontsize=11, color=GRAY)
    ax.text(0.15, 4.15, r"$\Omega = H_1 \cup H_2 \cup H_3 \cup H_4$,   $H_i \cap H_j = \varnothing$", fontsize=12)
    ax.text(4, -0.45, r"$P(A) = \sum_i P(A \mid H_i)\,P(H_i)$", ha="center", fontsize=14)
    ax.set_xlim(-0.2, 8.2)
    ax.set_ylim(-0.8, 4.6)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def venn_conditional(out: Path) -> None:
    """Условная вероятность: сужаем пространство исходов до B."""
    plt = _mpl()
    from matplotlib.patches import Circle, Rectangle

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.add_patch(Rectangle((0, 0), 8, 4.2, fill=True, color="#f3f4f6", lw=0))
    ax.add_patch(Rectangle((0, 0), 8, 4.2, fill=False, lw=2, color="black"))
    a, b = Circle((3.0, 2.1), 1.6), Circle((5.0, 2.1), 1.6)
    ax.add_patch(Circle((3.0, 2.1), 1.6, color=BLUE, alpha=0.25, lw=0))
    ax.add_patch(Circle((5.0, 2.1), 1.6, color=ORANGE, alpha=0.30, lw=0))
    ax.add_patch(Circle((3.0, 2.1), 1.6, fill=False, color=BLUE, lw=2.2))
    ax.add_patch(Circle((5.0, 2.1), 1.6, fill=False, color=ORANGE, lw=2.2))
    # пересечение — штриховкой (клиппинг круга A кругом B)
    inter = Circle((3.0, 2.1), 1.6, fill=True, color=RED, alpha=0.45, lw=0)
    ax.add_patch(inter)
    inter.set_clip_path(b)
    b.set_transform(ax.transData)
    ax.text(2.1, 2.1, "$A$", fontsize=18, color=BLUE, ha="center", va="center")
    ax.text(5.9, 2.1, "$B$", fontsize=18, color=ORANGE, ha="center", va="center")
    ax.text(4.0, 2.1, r"$A\cap B$", fontsize=13, color="black", ha="center", va="center")
    ax.text(0.15, 3.85, r"$\Omega$", fontsize=16)
    ax.text(4, -0.5, r"$P(A \mid B) = \dfrac{P(A \cap B)}{P(B)}$  — доля пересечения внутри $B$", ha="center", fontsize=13)
    ax.set_xlim(-0.2, 8.2)
    ax.set_ylim(-1.0, 4.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def normal_pdf(out: Path) -> None:
    """Плотности нормального распределения при разных μ и σ; область μ ± σ."""
    plt = _mpl()
    np = _np()
    x = np.linspace(-6, 6, 800)
    pdf = lambda x, m, s: np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))  # noqa: E731
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for m, s, c in [(0, 1, BLUE), (0, 0.5, ORANGE), (0, 2, GREEN), (-2, 1, PURPLE)]:
        ax.plot(x, pdf(x, m, s), color=c, lw=2.2, label=f"$\\mu={m},\\ \\sigma={s}$")
    mask = (x >= -1) & (x <= 1)
    ax.fill_between(x[mask], pdf(x[mask], 0, 1), color=BLUE, alpha=0.18)
    ax.text(0, 0.12, r"$\mu\pm\sigma$: $\approx 68\%$", ha="center", fontsize=11, color=BLUE)
    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$f(x) = \frac{1}{\sigma\sqrt{2\pi}}\,e^{-(x-\mu)^2/2\sigma^2}$")
    ax.set_ylim(0, 0.85)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def roc_curve(out: Path) -> None:
    """ROC-кривые: случайный, два классификатора разного качества, идеальный."""
    plt = _mpl()
    np = _np()
    fpr = np.linspace(0, 1, 400)
    # аналитические ROC для бимодальных гауссиан: TPR = Φ(d' + Φ⁻¹(FPR))
    from math import erf, sqrt

    phi = np.vectorize(lambda z: 0.5 * (1 + erf(z / sqrt(2))))
    inv = lambda p: np.sqrt(2) * _erfinv(2 * np.clip(p, 1e-9, 1 - 1e-9) - 1)  # noqa: E731
    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    ax.fill_between(fpr, 0, phi(1.2 + inv(fpr)), color=BLUE, alpha=0.08)
    for d, c, name in [(0.9, ORANGE, "модель A"), (1.2, BLUE, "модель B"), (2.2, GREEN, "модель C")]:
        tpr = phi(d + inv(fpr))
        auc = np.trapezoid(tpr, fpr)
        ax.plot(fpr, tpr, color=c, lw=2.4, label=f"{name}, AUC ≈ {auc:.2f}")
    ax.plot([0, 1], [0, 1], "--", color=GRAY, lw=1.6, label="случайный, AUC = 0.5")
    ax.plot([0, 0, 1], [0, 1, 1], color=RED, lw=1.4, alpha=0.7, label="идеальный, AUC = 1")
    ax.set_xlabel("FPR = FP / (FP + TN)")
    ax.set_ylabel("TPR = TP / (TP + FN)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    ax.grid(alpha=0.25)
    ax.set_title("ROC-кривая: TPR против FPR при движении порога")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _erfinv(y):
    np = _np()
    # аппроксимация Giles (достаточна для графика)
    w = -np.log((1.0 - y) * (1.0 + y))
    p = np.where(
        w < 5.0,
        (((((((2.81022636e-08 * (w - 2.5) + 3.43273939e-07) * (w - 2.5) - 3.5233877e-06) * (w - 2.5) - 4.39150654e-06) * (w - 2.5) + 0.00021858087) * (w - 2.5) - 0.00125372503) * (w - 2.5) - 0.00417768164) * (w - 2.5) + 0.246640727) * (w - 2.5) + 1.50140941,
        (((((((-0.000200214257 * (np.sqrt(w) - 3.0) + 0.000100950558) * (np.sqrt(w) - 3.0) + 0.00134934322) * (np.sqrt(w) - 3.0) - 0.00367342844) * (np.sqrt(w) - 3.0) + 0.00573950773) * (np.sqrt(w) - 3.0) - 0.0076224613) * (np.sqrt(w) - 3.0) + 0.00943887047) * (np.sqrt(w) - 3.0) + 1.00167406) * (np.sqrt(w) - 3.0) + 2.83297682,
    )
    return p * y


def grid_vs_random_search(out: Path) -> None:
    """Grid Search против Random Search при одном важном и одном неважном гиперпараметре."""
    plt = _mpl()
    np = _np()
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2))
    imp = lambda x: np.exp(-((x - 0.62) ** 2) / 0.03)  # noqa: E731  # важный параметр: узкий пик
    unimp = lambda y: 0.55 + 0.05 * np.sin(6 * y)  # noqa: E731  # неважный: почти плоско
    panels = [
        (axes[0], "Grid Search: 9 точек, 3 уникальных значения на ось", np.array([(i / 4, j / 4) for i in (1, 2, 3) for j in (1, 2, 3)])),
        (axes[1], "Random Search: 9 точек, 9 уникальных значений на ось", rng.uniform(0.06, 0.94, size=(9, 2))),
    ]
    for ax_, title, pts in panels:
        t = np.linspace(0, 1, 300)
        ax_.plot(t, 0.12 * imp(t) + 1.0, color=ORANGE, lw=2)  # маргинал важного параметра (сверху)
        ax_.plot(0.12 * unimp(t) + 1.0, t, color=GRAY, lw=2)  # маргинал неважного (справа)
        ax_.scatter(pts[:, 0], pts[:, 1], s=70, color=BLUE, zorder=3, edgecolor="white")
        ax_.vlines(pts[:, 0], 0.98, 1.0, color=ORANGE, lw=1, alpha=0.7)
        ax_.hlines(pts[:, 1], 0.98, 1.0, color=GRAY, lw=1, alpha=0.7)
        ax_.axvspan(0.5, 0.74, color=ORANGE, alpha=0.08)
        ax_.set_xlim(0, 1.16)
        ax_.set_ylim(0, 1.16)
        ax_.set_xlabel("важный гиперпараметр (напр. learning rate)")
        ax_.set_ylabel("неважный гиперпараметр")
        ax_.set_title(title, fontsize=11)
        ax_.set_xticks([])
        ax_.set_yticks([])
        ax_.text(0.62, 1.135, "качество", ha="center", fontsize=9, color=ORANGE)
    axes[0].text(0.62, 0.03, "оптимум пропущен", ha="center", fontsize=9, color=RED)
    fig.suptitle("Bergstra & Bengio (2012): при малом числе важных параметров случайный поиск покрывает их лучше", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


FIGURES = {
    "venn_partition": ("bayes-theorem-and-probability-foundations", venn_partition),
    "venn_conditional": ("bayes-theorem-and-probability-foundations", venn_conditional),
    "normal_pdf": ("gaussian-distribution", normal_pdf),
    "roc_curve": ("roc-curve-and-roc-auc", roc_curve),
    "grid_vs_random_search": ("hyperparameter-tuning", grid_vs_random_search),
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", help="Список фигур через запятую (по умолчанию все)")
    p.add_argument("--out-root", type=Path, default=TOPICS, help="Корень topics/ (для тестов)")
    args = p.parse_args(argv)
    names = args.only.split(",") if args.only else list(FIGURES)
    for name in names:
        slug, fn = FIGURES[name]
        out = args.out_root / slug / "assets" / "images" / f"{name}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fn(out)
        print(f"Записан {out.relative_to(REPO_ROOT) if out.is_relative_to(REPO_ROOT) else out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
