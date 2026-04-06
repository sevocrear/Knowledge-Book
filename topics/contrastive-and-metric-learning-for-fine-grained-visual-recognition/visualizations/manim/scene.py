"""
Contrastive / metric learning: embedding space before vs after + triplet intuition.
Target ~30–40 s @ 30fps: on-screen Russian labels, smooth pacing (3b1b-inspired).
"""

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
    return random.Random(seed)


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

        title = Text("Contrastive / metric learning", font_size=40, color=WHITE).to_edge(UP, buff=0.32)
        subtitle = Text("похожие ближе, разные дальше в пространстве эмбеддингов", font_size=22, color=GRAY).next_to(
            title, DOWN, buff=0.12
        )
        self.play(
            LaggedStart(FadeIn(title, shift=UP * 0.12), FadeIn(subtitle, shift=UP * 0.08), lag_ratio=0.45),
            run_time=1.0,
            rate_func=smooth,
        )
        self.wait(0.2)

        plane = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-2.5, 2.5, 1],
            background_line_style={"stroke_color": "#202533", "stroke_width": 1.0, "stroke_opacity": 0.85},
            axis_config={"stroke_color": "#31384A", "stroke_width": 2.0},
        ).scale(0.88).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(plane), run_time=0.7, rate_func=smooth)

        txt_before = Text("до обучения", font_size=24, color=WHITE).next_to(plane, UP, buff=0.15).shift(LEFT * 2.35)
        txt_after = Text("после обучения", font_size=24, color=WHITE).next_to(plane, UP, buff=0.15).shift(RIGHT * 2.35)
        self.play(FadeIn(txt_before), FadeIn(txt_after), run_time=0.5, rate_func=smooth)

        pts_b, y_b = _make_points(seed=3)
        pts_a, y_a = _make_points(seed=3)
        palette = [BLUE_PRI, ORANGE, GREEN, PURPLE, RED_NEG, "#FFD166"]

        def dots_from(points, labels, x_shift: float):
            dots = VGroup()
            for (x, y), c in zip(points, labels, strict=True):
                d = Dot(point=plane.c2p(x + x_shift, y, 0), radius=0.055, color=palette[c % len(palette)])
                d.set_opacity(0.92)
                dots.add(d)
            return dots

        before = dots_from(pts_b, y_b, x_shift=-1.5)

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

        self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in before], lag_ratio=0.03), run_time=1.0, rate_func=smooth)
        self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in after], lag_ratio=0.03), run_time=1.0, rate_func=smooth)
        self.wait(0.25)

        # Triplet on the left cloud
        a_idx, p_idx, n_idx = 3, 8, 25
        anchor = before[a_idx]
        positive = before[p_idx]
        negative = before[n_idx]

        leg_a = Text("якорь", font_size=20, color=WHITE)
        leg_p = Text("позитив (тот же класс)", font_size=20, color=GREEN)
        leg_n = Text("негатив (другой класс)", font_size=20, color=RED_NEG)
        legend = VGroup(leg_a, leg_p, leg_n).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        legend.to_corner(UR).shift(LEFT * 0.2 + DOWN * 0.35)
        self.play(FadeIn(legend, shift=LEFT * 0.1), run_time=0.5, rate_func=smooth)

        self.play(Circumscribe(anchor, color=WHITE, time_width=0.75), run_time=0.75, rate_func=smooth)

        ap = Line(anchor.get_center(), positive.get_center(), color=GREEN, stroke_width=4)
        an = Line(anchor.get_center(), negative.get_center(), color=RED_NEG, stroke_width=4)
        self.play(Create(ap), Create(an), run_time=0.55, rate_func=smooth)
        self.play(Indicate(positive, color=GREEN, scale_factor=1.28), run_time=0.45)
        self.play(Indicate(negative, color=RED_NEG, scale_factor=1.28), run_time=0.45)

        eq = MathTex(r"L=\max(0,\ d(a,p)-d(a,n)+m)", font_size=32, color=WHITE).to_edge(DOWN, buff=0.15)
        triplet_hint = Text("triplet: ближе к позитиву, дальше от негатива", font_size=20, color=GRAY).next_to(
            eq, UP, buff=0.1
        )
        self.play(FadeIn(triplet_hint), Write(eq), run_time=0.9, rate_func=smooth)
        self.wait(0.9)

        self.play(FadeOut(ap), FadeOut(an), FadeOut(eq), FadeOut(triplet_hint), FadeOut(legend), run_time=0.45)
        takeaway = Text("Итог: компактные кластеры по классам → лучше retrieval / verification", font_size=22, color=WHITE)
        takeaway.next_to(plane, UP, buff=0.05)
        self.play(FadeIn(takeaway, shift=UP * 0.08), run_time=0.7, rate_func=smooth)
        self.wait(0.85)
