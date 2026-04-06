"""
ViT intuition: patches → tokens + CLS → one self-attention row (~30–40 s).
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
PURPLE = "#9B72CF"


class ViTPatchesAndAttentionScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        title = Text("Vision Transformer (идея)", font_size=40, color=WHITE).to_edge(UP, buff=0.3)
        hook = Text("Картинка → патчи → последовательность → attention", font_size=22, color=GRAY).next_to(title, DOWN, buff=0.1)
        self.play(FadeIn(title, shift=DOWN * 0.08), FadeIn(hook, shift=DOWN * 0.06), run_time=0.85, rate_func=smooth)
        self.wait(0.15)

        # Fake image as 2×3 patches
        img = Rectangle(width=3.6, height=2.4, color=BLUE_DIM, stroke_width=3, fill_color=BLUE_DIM, fill_opacity=0.35)
        img.shift(UP * 0.35)
        img_lbl = Text("изображение", font_size=22, color=GRAY).next_to(img, DOWN, buff=0.15)
        self.play(FadeIn(img), FadeIn(img_lbl), run_time=0.55, rate_func=smooth)

        vlines = VGroup(*[Line(img.get_corner(UL) + RIGHT * (i + 1) * 1.2, img.get_corner(DL) + RIGHT * (i + 1) * 1.2, color=WHITE, stroke_width=2) for i in range(2)])
        hlines = VGroup(*[Line(img.get_corner(UL) + DOWN * (j + 1) * 1.2, img.get_corner(UR) + DOWN * (j + 1) * 1.2, color=WHITE, stroke_width=2) for j in range(1)])
        self.play(Create(vlines), Create(hlines), run_time=0.65, rate_func=smooth)
        patch_txt = Text("патчи P×P", font_size=22, color=ORANGE).next_to(img, UP, buff=0.12)
        self.play(FadeIn(patch_txt, shift=UP * 0.08), run_time=0.4, rate_func=smooth)
        self.wait(0.2)

        self.play(FadeOut(vlines), FadeOut(hlines), FadeOut(patch_txt), FadeOut(img_lbl), run_time=0.35)

        # Row of patch tokens + CLS
        n_patches = 6
        cls_box = RoundedRectangle(width=0.65, height=0.55, corner_radius=0.06, color=PURPLE, fill_color=PURPLE, fill_opacity=0.45, stroke_width=2)
        cls_lbl = Text("CLS", font_size=18, color=WHITE).move_to(cls_box)
        cls_g = VGroup(cls_box, cls_lbl)

        patches = VGroup()
        for i in range(n_patches):
            b = Square(side_length=0.52, color=BLUE, fill_color=BLUE_DIM, fill_opacity=0.85, stroke_width=2)
            t = Text(str(i + 1), font_size=16, color=WHITE).move_to(b)
            patches.add(VGroup(b, t))

        row = VGroup(cls_g, *patches).arrange(RIGHT, buff=0.12).move_to(ORIGIN + UP * 0.55)
        tok_caption = Text("последовательность токенов (как слова в NLP)", font_size=20, color=GRAY).next_to(row, DOWN, buff=0.25)
        self.play(
            FadeOut(img),
            FadeIn(cls_g, shift=RIGHT * 0.2),
            LaggedStart(*[FadeIn(p, scale=0.6) for p in patches], lag_ratio=0.07),
            run_time=1.0,
            rate_func=smooth,
        )
        self.play(FadeIn(tok_caption), run_time=0.4, rate_func=smooth)
        self.wait(0.2)

        # Mini attention: query from patch 3 (index 3 in row after CLS is 0, so patch index 3 = 4th box)
        q_idx = 3
        query_g = row[q_idx]
        self.play(Indicate(query_g, color=ORANGE, scale_factor=1.12), run_time=0.45)

        weights = [0.08, 0.07, 0.52, 0.18, 0.08, 0.07]
        heat = VGroup()
        for i, w in enumerate(weights):
            sq = Square(side_length=0.42, fill_color=ORANGE, fill_opacity=0.15 + w * 1.1, color=ORANGE, stroke_width=1)
            lab = Text(f"{w:.2f}", font_size=12, color=WHITE).move_to(sq)
            heat.add(VGroup(sq, lab))
        heat.arrange(RIGHT, buff=0.08).next_to(row, DOWN, buff=0.85)
        hm_lbl = Text("веса softmax по одной строке (query одного патча)", font_size=18, color=GRAY).next_to(heat, UP, buff=0.12)
        self.play(FadeIn(hm_lbl, shift=UP * 0.06), LaggedStart(*[FadeIn(h) for h in heat], lag_ratio=0.06), run_time=0.85, rate_func=smooth)

        blend = Text("итог = взвешенная сумма Value-токенов", font_size=20, color=GREEN).to_edge(DOWN, buff=0.55)
        self.play(FadeIn(blend, shift=UP * 0.1), run_time=0.45, rate_func=smooth)

        eq = MathTex(r"\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V", font_size=30, color=WHITE).to_edge(DOWN, buff=0.12)
        self.play(Write(eq), run_time=0.85, rate_func=smooth)
        self.wait(0.55)

        out = Text("Дальше: L слоёв MSA + MLP, затем классификация по CLS", font_size=20, color=WHITE)
        out.to_edge(DOWN, buff=0.12)
        self.play(FadeOut(eq), FadeOut(blend), FadeIn(out), run_time=0.55, rate_func=smooth)
        self.wait(0.65)
