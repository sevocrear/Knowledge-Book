from __future__ import annotations

import math
import random

from manim import *


BG = "#0F1117"
BLUE_PRI = "#4A9EFF"
ORANGE = "#FF7A2F"
GREEN = "#3DBA7A"
RED_NEG = "#FF4D6D"
GRAY = "#6B7280"
WHITE = "#F0F4FF"
PURPLE = "#9B72CF"


def _rng(seed: int) -> random.Random:
    r = random.Random(seed)
    return r


def _make_points(seed: int, num_classes: int = 6, points_per_class: int = 14):
    r = _rng(seed)
    centers = []
    for i in range(num_classes):
        a = 2 * math.pi * i / num_classes
        centers.append((0.9 * math.cos(a), 0.9 * math.sin(a)))

    pts = []
    labels = []
    for c, (cx, cy) in enumerate(centers):
        for _ in range(points_per_class):
            x = cx + r.gauss(0, 0.35)
            y = cy + r.gauss(0, 0.35)
            pts.append((x, y))
            labels.append(c)
    return pts, labels


class ContrastiveEmbeddingSpace(Scene):
    def construct(self):
        self.camera.background_color = BG

        title = Text("Contrastive / Metric Learning", font_size=44, color=WHITE).to_edge(UP, buff=0.35)
        subtitle = Text("эмбеддинги: похожие ближе, разные дальше", font_size=22, color=GRAY).next_to(
            title, DOWN, buff=0.15
        )
        self.play(FadeIn(title), FadeIn(subtitle), run_time=1.0)
        self.wait(0.3)

        plane = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-2.5, 2.5, 1],
            background_line_style={"stroke_color": "#202533", "stroke_width": 1.0, "stroke_opacity": 0.85},
            axis_config={"stroke_color": "#31384A", "stroke_width": 2.0},
        ).scale(0.9).to_edge(DOWN, buff=0.4)
        self.play(FadeIn(plane), run_time=0.8)

        txt_before = Text("До обучения", font_size=28, color=WHITE).next_to(plane, UP, buff=0.2).shift(LEFT * 2.4)
        txt_after = Text("После обучения", font_size=28, color=WHITE).next_to(plane, UP, buff=0.2).shift(RIGHT * 2.4)
        self.play(FadeIn(txt_before), FadeIn(txt_after), run_time=0.6)

        # Two clouds: before (left) and after (right).
        pts_b, y_b = _make_points(seed=3)
        pts_a, y_a = _make_points(seed=3)

        palette = [BLUE_PRI, ORANGE, GREEN, PURPLE, RED_NEG, "#FFD166"]

        def dots_from(points, labels, x_shift: float):
            dots = VGroup()
            for (x, y), c in zip(points, labels, strict=True):
                d = Dot(point=plane.c2p(x + x_shift, y, 0), radius=0.06, color=palette[c % len(palette)])
                d.set_opacity(0.9)
                dots.add(d)
            return dots

        before = dots_from(pts_b, y_b, x_shift=-1.5)

        # "After" points: slightly pull each class toward its center and push centers apart.
        r = _rng(7)
        centers = []
        for i in range(6):
            a = 2 * math.pi * i / 6
            centers.append((1.25 * math.cos(a), 1.25 * math.sin(a)))

        pts_after = []
        for (x, y), c in zip(pts_a, y_a, strict=True):
            cx, cy = centers[c]
            x2 = 0.35 * x + 0.65 * cx + r.gauss(0, 0.08)
            y2 = 0.35 * y + 0.65 * cy + r.gauss(0, 0.08)
            pts_after.append((x2, y2))

        after = dots_from(pts_after, y_a, x_shift=+1.5)

        self.play(LaggedStart(*[FadeIn(d) for d in before], lag_ratio=0.02), run_time=1.2)
        self.play(LaggedStart(*[FadeIn(d) for d in after], lag_ratio=0.02), run_time=1.2)
        self.wait(0.4)

        # Show one anchor-positive-negative relation on the "before" side.
        a_idx = 3
        p_idx = 8
        n_idx = 25
        anchor = before[a_idx]
        positive = before[p_idx]
        negative = before[n_idx]

        self.play(Circumscribe(anchor, color=WHITE, time_width=0.8), run_time=0.9)

        ap = Line(anchor.get_center(), positive.get_center(), color=GREEN, stroke_width=4)
        an = Line(anchor.get_center(), negative.get_center(), color=RED_NEG, stroke_width=4)
        self.play(Create(ap), Create(an), run_time=0.6)
        self.play(Indicate(positive, color=GREEN, scale_factor=1.35), run_time=0.5)
        self.play(Indicate(negative, color=RED_NEG, scale_factor=1.35), run_time=0.5)

        eq = MathTex(r"L=\max(0,\ d(a,p)-d(a,n)+m)", font_size=34, color=WHITE).to_edge(DOWN, buff=0.2)
        self.play(FadeIn(eq), run_time=0.6)
        self.wait(1.2)

        self.play(FadeOut(ap), FadeOut(an), FadeOut(eq), run_time=0.5)
        self.wait(0.5)
