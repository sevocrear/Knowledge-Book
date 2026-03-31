from __future__ import annotations

import math

from manim import *

BG = "#0F1117"
BLUE_PRI = "#4A9EFF"
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

        title = Text("ArcFace", font_size=50, color=WHITE).to_edge(UP, buff=0.3)
        subtitle = Text("additive angular margin on hypersphere", font_size=24, color=GRAY).next_to(
            title, DOWN, buff=0.1
        )
        self.play(FadeIn(title), FadeIn(subtitle), run_time=1.0)

        center = DOWN * 0.45
        sphere = Circle(radius=2.5, color=BLUE_DIM, stroke_width=4).move_to(center)
        sphere_lbl = Text("L2-normalized embedding space", font_size=22, color=GRAY).next_to(
            sphere, DOWN, buff=0.25
        )
        self.play(Create(sphere), FadeIn(sphere_lbl), run_time=0.8)

        # Two class prototypes on the unit sphere.
        theta_w1 = math.radians(28)
        theta_w2 = math.radians(122)
        r = sphere.radius
        c = sphere.get_center()

        w1_end = c + np.array([r * math.cos(theta_w1), r * math.sin(theta_w1), 0.0])
        w2_end = c + np.array([r * math.cos(theta_w2), r * math.sin(theta_w2), 0.0])
        w1 = Arrow(c, w1_end, buff=0.0, color=GREEN, stroke_width=6, max_stroke_width_to_length_ratio=15)
        w2 = Arrow(c, w2_end, buff=0.0, color=PURPLE, stroke_width=6, max_stroke_width_to_length_ratio=15)
        w1_t = MathTex(r"W_y", color=GREEN, font_size=30).next_to(w1_end, RIGHT, buff=0.12)
        w2_t = MathTex(r"W_j", color=PURPLE, font_size=30).next_to(w2_end, LEFT, buff=0.12)

        self.play(GrowArrow(w1), GrowArrow(w2), FadeIn(w1_t), FadeIn(w2_t), run_time=1.0)
        self.wait(0.3)

        # Embedding vector x close to true class prototype.
        theta_x = math.radians(44)
        x_end = c + np.array([r * math.cos(theta_x), r * math.sin(theta_x), 0.0])
        x_vec = Arrow(c, x_end, buff=0.0, color=ORANGE, stroke_width=6, max_stroke_width_to_length_ratio=15)
        x_t = MathTex(r"x_i", color=ORANGE, font_size=30).next_to(x_end, UP, buff=0.1)
        self.play(GrowArrow(x_vec), FadeIn(x_t), run_time=0.9)

        # Angular distances and margin.
        theta_arc = Arc(
            radius=0.95,
            start_angle=theta_w1,
            angle=theta_x - theta_w1,
            arc_center=c,
            color=WHITE,
            stroke_width=5,
        )
        theta_lbl = MathTex(r"\theta_y", color=WHITE, font_size=30).move_to(
            c + np.array([0.78 * math.cos((theta_w1 + theta_x) / 2), 0.78 * math.sin((theta_w1 + theta_x) / 2), 0.0])
        )
        self.play(Create(theta_arc), FadeIn(theta_lbl), run_time=0.8)

        m = math.radians(18)
        margin_arc = Arc(
            radius=1.25,
            start_angle=theta_x,
            angle=m,
            arc_center=c,
            color=RED_NEG,
            stroke_width=6,
        )
        m_lbl = MathTex(r"m", color=RED_NEG, font_size=32).move_to(
            c + np.array(
                [1.22 * math.cos(theta_x + m / 2), 1.22 * math.sin(theta_x + m / 2), 0.0]
            )
        )
        self.play(Create(margin_arc), FadeIn(m_lbl), run_time=0.8)

        eq = MathTex(
            r"\text{ArcFace:}\quad s\cdot \cos(\theta_y + m)",
            color=WHITE,
            font_size=40,
        ).to_edge(DOWN, buff=0.18)
        self.play(Write(eq), run_time=1.0)

        hint = Text(
            "True class must be closer with extra angular gap",
            font_size=21,
            color=GRAY,
        ).next_to(eq, UP, buff=0.12)
        self.play(FadeIn(hint), run_time=0.5)

        # Visual effect: push x_i slightly closer to W_y.
        new_theta_x = theta_x - math.radians(7)
        new_x_end = c + np.array([r * math.cos(new_theta_x), r * math.sin(new_theta_x), 0.0])
        new_x_vec = Arrow(c, new_x_end, buff=0.0, color=ORANGE, stroke_width=6, max_stroke_width_to_length_ratio=15)
        new_x_t = MathTex(r"x_i", color=ORANGE, font_size=30).next_to(new_x_end, UP, buff=0.08)
        self.play(Transform(x_vec, new_x_vec), Transform(x_t, new_x_t), run_time=1.2)
        self.play(Indicate(w1, color=GREEN, scale_factor=1.05), run_time=0.6)

        compare = Text("better intra-class compactness + inter-class separation", font_size=20, color=GRAY).next_to(
            sphere, UP, buff=0.2
        )
        self.play(FadeIn(compare), run_time=0.5)
        self.wait(1.4)
