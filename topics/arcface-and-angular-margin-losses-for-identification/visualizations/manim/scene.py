"""
ArcFace: angular margin on the unit hypersphere (~30–40 s, labeled, smooth pacing).
"""

from __future__ import annotations

import math

from manim import *

BG = "#0F1117"
BLUE_DIM = "#1E3A5F"
ORANGE = "#FF7A2F"
GREEN = "#3DBA7A"
RED_NEG = "#FF4D6D"
GRAY = "#6B7280"
WHITE = "#F0F4FF"
PURPLE = "#9B72CF"


class ArcFaceAngularMarginScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        title = Text("ArcFace", font_size=48, color=WHITE).to_edge(UP, buff=0.28)
        subtitle = Text("добавочный угловой margin на гиперсфере", font_size=22, color=GRAY).next_to(title, DOWN, buff=0.08)
        self.play(FadeIn(title, shift=DOWN * 0.1), FadeIn(subtitle, shift=DOWN * 0.08), run_time=0.85, rate_func=smooth)

        center = DOWN * 0.42
        sphere = Circle(radius=2.45, color=BLUE_DIM, stroke_width=4).move_to(center)
        sphere_lbl = Text("L2-нормированные эмбеддинги и веса классов", font_size=20, color=GRAY).next_to(
            sphere, DOWN, buff=0.2
        )
        self.play(Create(sphere), FadeIn(sphere_lbl), run_time=0.75, rate_func=smooth)

        cos_m = MathTex(r"\cos\theta = x \cdot W", font_size=26, color=GRAY)
        cos_txt = Text("при L2-норме 1", font_size=20, color=GRAY).next_to(cos_m, DOWN, buff=0.06, aligned_edge=LEFT)
        cos_note = VGroup(cos_m, cos_txt).to_corner(UL).shift(DOWN * 0.5 + RIGHT * 0.12)
        self.play(FadeIn(cos_note, shift=RIGHT * 0.1), run_time=0.55, rate_func=smooth)

        theta_w1 = math.radians(28)
        theta_w2 = math.radians(122)
        r = sphere.radius
        c = sphere.get_center()

        w1_end = c + np.array([r * math.cos(theta_w1), r * math.sin(theta_w1), 0.0])
        w2_end = c + np.array([r * math.cos(theta_w2), r * math.sin(theta_w2), 0.0])
        w1 = Arrow(c, w1_end, buff=0.0, color=GREEN, stroke_width=6, max_stroke_width_to_length_ratio=15)
        w2 = Arrow(c, w2_end, buff=0.0, color=PURPLE, stroke_width=6, max_stroke_width_to_length_ratio=15)
        w1_t = MathTex(r"W_y", color=GREEN, font_size=28).next_to(w1_end, RIGHT, buff=0.1)
        w2_t = MathTex(r"W_j", color=PURPLE, font_size=28).next_to(w2_end, LEFT, buff=0.1)

        self.play(GrowArrow(w1), GrowArrow(w2), FadeIn(w1_t), FadeIn(w2_t), run_time=0.85, rate_func=smooth)
        self.wait(0.15)

        theta_x = math.radians(44)
        x_end = c + np.array([r * math.cos(theta_x), r * math.sin(theta_x), 0.0])
        x_vec = Arrow(c, x_end, buff=0.0, color=ORANGE, stroke_width=6, max_stroke_width_to_length_ratio=15)
        x_t = MathTex(r"x_i", color=ORANGE, font_size=28).next_to(x_end, UP, buff=0.08)
        self.play(GrowArrow(x_vec), FadeIn(x_t), run_time=0.75, rate_func=smooth)

        theta_arc = Arc(
            radius=0.92,
            start_angle=theta_w1,
            angle=theta_x - theta_w1,
            arc_center=c,
            color=WHITE,
            stroke_width=5,
        )
        theta_lbl = MathTex(r"\theta_y", color=WHITE, font_size=28).move_to(
            c + np.array([0.76 * math.cos((theta_w1 + theta_x) / 2), 0.76 * math.sin((theta_w1 + theta_x) / 2), 0.0])
        )
        self.play(Create(theta_arc), FadeIn(theta_lbl), run_time=0.65, rate_func=smooth)

        m = math.radians(16)
        margin_arc = Arc(
            radius=1.2,
            start_angle=theta_x,
            angle=m,
            arc_center=c,
            color=RED_NEG,
            stroke_width=6,
        )
        m_lbl = MathTex(r"m", color=RED_NEG, font_size=30).move_to(
            c + np.array([1.18 * math.cos(theta_x + m / 2), 1.18 * math.sin(theta_x + m / 2), 0.0])
        )
        self.play(Create(margin_arc), FadeIn(m_lbl), run_time=0.65, rate_func=smooth)

        eq_l = MathTex(r"s\cos(\theta_y + m)", font_size=34, color=WHITE)
        eq_mid = Text("вместо", font_size=26, color=GRAY)
        eq_r = MathTex(r"s\cos\theta_y", font_size=34, color=WHITE)
        eq = VGroup(eq_l, eq_mid, eq_r).arrange(RIGHT, buff=0.2).to_edge(DOWN, buff=0.12)
        self.play(LaggedStart(Write(eq_l), FadeIn(eq_mid), Write(eq_r), lag_ratio=0.35), run_time=0.95, rate_func=smooth)

        hint = Text("правильный класс должен «победить» с дополнительным угловым зазором", font_size=19, color=GRAY)
        hint.next_to(eq, UP, buff=0.1)
        self.play(FadeIn(hint), run_time=0.45, rate_func=smooth)

        new_theta_x = theta_x - math.radians(6)
        new_x_end = c + np.array([r * math.cos(new_theta_x), r * math.sin(new_theta_x), 0.0])
        new_x_vec = Arrow(c, new_x_end, buff=0.0, color=ORANGE, stroke_width=6, max_stroke_width_to_length_ratio=15)
        new_x_t = MathTex(r"x_i", color=ORANGE, font_size=28).next_to(new_x_end, UP, buff=0.06)
        self.play(Transform(x_vec, new_x_vec), Transform(x_t, new_x_t), run_time=0.95, rate_func=smooth)
        self.play(Indicate(w1, color=GREEN, scale_factor=1.04), run_time=0.5)

        compare = Text("компактнее внутри класса, шире между классами", font_size=20, color=GRAY).next_to(sphere, UP, buff=0.12)
        self.play(FadeIn(compare, shift=UP * 0.08), run_time=0.45, rate_func=smooth)
        self.wait(0.75)
