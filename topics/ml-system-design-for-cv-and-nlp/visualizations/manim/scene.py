"""
Serving: клиенты → балансировщик → dynamic batching → GPU (~30–40 с).
"""

from __future__ import annotations

from manim import *

BG = "#0F1117"
WHITE = "#F0F4FF"
GRAY = "#6B7280"
BLUE = "#4A9EFF"
BLUE_DIM = "#1E3A5F"
ORANGE = "#FF7A2F"
GREEN = "#3DBA7A"
RED = "#FF4D6D"
PURPLE = "#9B72CF"


def _box(label: str, width: float, height: float, color: str, font_size: int = 22) -> VGroup:
    rect = RoundedRectangle(
        width=width,
        height=height,
        corner_radius=0.1,
        color=color,
        fill_color=BLUE_DIM,
        fill_opacity=0.85,
        stroke_width=3,
    )
    txt = Text(label, font_size=font_size, color=WHITE).move_to(rect)
    return VGroup(rect, txt)


class ServingLoadBalancerScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        title = Text("Serving модели", font_size=36, color=WHITE).to_edge(UP, buff=0.28)
        sub = Text("очередь, балансировщик, пачка на GPU", font_size=22, color=GRAY).next_to(title, DOWN, buff=0.08)
        self.play(
            FadeIn(title, shift=DOWN * 0.08),
            FadeIn(sub, shift=DOWN * 0.06),
            run_time=0.75,
            rate_func=smooth,
        )

        clients_ok = VGroup(
            *[Dot(radius=0.08, color=BLUE).shift(LEFT * 5.2 + UP * (0.7 - 0.18 * i) + RIGHT * (0.16 * (i % 3))) for i in range(12)]
        )
        cap100 = Text("100 клиентов", font_size=20, color=BLUE).next_to(clients_ok, DOWN, buff=0.18)
        gpu = _box("1 GPU", 1.6, 0.85, GREEN, 24).shift(RIGHT * 0.2 + UP * 0.15)
        rho_ok = Text("ρ = 0.8  очередь живая", font_size=22, color=GREEN).next_to(gpu, DOWN, buff=0.28)
        a1 = Arrow(clients_ok.get_right() + RIGHT * 0.15, gpu.get_left(), buff=0.12, color=GRAY, stroke_width=4)

        self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in clients_ok], lag_ratio=0.04), run_time=0.7, rate_func=smooth)
        self.play(FadeIn(cap100), GrowArrow(a1), FadeIn(gpu, shift=LEFT * 0.1), run_time=0.55, rate_func=smooth)
        self.play(FadeIn(rho_ok, shift=UP * 0.08), run_time=0.4, rate_func=smooth)
        self.wait(0.25)

        self.play(FadeOut(clients_ok), FadeOut(cap100), FadeOut(a1), FadeOut(rho_ok), run_time=0.35)

        flood = VGroup(
            *[
                Dot(radius=0.07, color=RED).shift(LEFT * 5.0 + UP * (1.15 - 0.13 * (i % 14)) + RIGHT * (0.18 * (i // 14)))
                for i in range(42)
            ]
        )
        cap1000 = Text("1000 клиентов", font_size=20, color=RED).next_to(flood, DOWN, buff=0.16)
        queue = VGroup(
            *[
                RoundedRectangle(width=0.28, height=0.22, corner_radius=0.04, color=RED, fill_opacity=0.8).shift(
                    LEFT * 1.8 + UP * (-0.55 + 0.16 * i)
                )
                for i in range(8)
            ]
        )
        q_lbl = Text("очередь растёт", font_size=20, color=RED).next_to(queue, DOWN, buff=0.18)
        gpu_bad = gpu.copy()
        gpu_bad[0].set_color(RED)
        a2 = Arrow(flood.get_right() + RIGHT * 0.1, queue.get_left(), buff=0.1, color=RED, stroke_width=4)
        a3 = Arrow(queue.get_right(), gpu_bad.get_left(), buff=0.1, color=RED, stroke_width=4)
        rho_bad = Text("ρ = 8  таймауты", font_size=22, color=RED).next_to(gpu_bad, DOWN, buff=0.28)

        self.play(LaggedStart(*[FadeIn(d, scale=0.4) for d in flood], lag_ratio=0.015), run_time=0.7, rate_func=smooth)
        self.play(FadeIn(cap1000), GrowArrow(a2), FadeIn(queue, shift=UP * 0.1), FadeIn(q_lbl), run_time=0.55, rate_func=smooth)
        self.play(Transform(gpu, gpu_bad), GrowArrow(a3), FadeIn(rho_bad), run_time=0.5, rate_func=smooth)
        self.play(Indicate(queue, color=ORANGE, scale_factor=1.08), run_time=0.45)
        self.wait(0.2)

        self.play(
            FadeOut(flood),
            FadeOut(cap1000),
            FadeOut(a2),
            FadeOut(queue),
            FadeOut(q_lbl),
            FadeOut(a3),
            FadeOut(gpu),
            FadeOut(rho_bad),
            run_time=0.4,
        )

        clients2 = VGroup(
            *[Dot(radius=0.075, color=BLUE).shift(LEFT * 5.4 + UP * (0.85 - 0.16 * i)) for i in range(10)]
        )
        lb = _box("балансировщик", 2.3, 0.8, ORANGE, 20).shift(LEFT * 2.55 + UP * 0.25)
        batcher = _box("пачка B=8", 1.9, 0.7, PURPLE, 20).shift(RIGHT * 0.15 + UP * 0.25)
        gpus = VGroup(
            _box("GPU 1", 1.35, 0.62, GREEN, 20).shift(RIGHT * 3.35 + UP * 1.15),
            _box("GPU 2", 1.35, 0.62, GREEN, 20).shift(RIGHT * 3.35 + UP * 0.25),
            _box("GPU 3", 1.35, 0.62, GREEN, 20).shift(RIGHT * 3.35 + DOWN * 0.65),
        )
        a_lb = Arrow(clients2.get_right(), lb.get_left(), buff=0.1, color=GRAY, stroke_width=4)
        a_b = Arrow(lb.get_right(), batcher.get_left(), buff=0.1, color=ORANGE, stroke_width=4)
        a_g = VGroup(
            *[Arrow(batcher.get_right(), g.get_left(), buff=0.1, color=GREEN, stroke_width=3) for g in gpus]
        )
        take = Text("ρ < 1  ·  свежие ответы  ·  GPU не простаивает", font_size=22, color=GREEN).to_edge(DOWN, buff=0.28)

        self.play(FadeIn(clients2, shift=RIGHT * 0.12), run_time=0.4, rate_func=smooth)
        self.play(GrowArrow(a_lb), FadeIn(lb, shift=LEFT * 0.08), run_time=0.5, rate_func=smooth)
        self.play(GrowArrow(a_b), FadeIn(batcher, shift=LEFT * 0.08), run_time=0.45, rate_func=smooth)
        self.play(
            LaggedStart(*[GrowArrow(a) for a in a_g], *[FadeIn(g, shift=LEFT * 0.08) for g in gpus], lag_ratio=0.12),
            run_time=0.85,
            rate_func=smooth,
        )
        self.play(Circumscribe(batcher, color=PURPLE, buff=0.06), run_time=0.55)
        self.play(FadeIn(take, shift=UP * 0.08), run_time=0.5, rate_func=smooth)
        self.wait(0.55)
