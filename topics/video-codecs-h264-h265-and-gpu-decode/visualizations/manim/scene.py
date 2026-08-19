"""
GOP: I-кадр хранит картинку, P/B кодируют только отличие.
Один клип ~30–40 с, подписи на русском.
"""

from __future__ import annotations

from manim import *

BG = "#0F1117"
WHITE = "#F0F4FF"
GRAY = "#6B7280"
ORANGE = "#FF7A2F"
GREEN = "#3DBA7A"
PURPLE = "#9B72CF"


def _frame_card(letter: str, color: str, subtitle: str) -> VGroup:
    box = RoundedRectangle(
        width=1.42,
        height=1.48,
        corner_radius=0.12,
        color=color,
        fill_color=color,
        fill_opacity=0.28,
        stroke_width=2.5,
    )
    lab = Text(letter, font_size=34, color=WHITE).move_to(box.get_center() + UP * 0.16)
    sub = Text(subtitle, font_size=13, color=GRAY).next_to(lab, DOWN, buff=0.1)
    return VGroup(box, lab, sub)


class GopPredictionScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        title = Text("Кодек пишет отличие, не каждый кадр целиком", font_size=30, color=WHITE)
        title.to_edge(UP, buff=0.28)
        hook = Text("GOP: I — якорь, P/B — предсказание", font_size=20, color=GRAY)
        hook.next_to(title, DOWN, buff=0.1)
        self.play(
            FadeIn(title, shift=DOWN * 0.08),
            FadeIn(hook, shift=DOWN * 0.05),
            run_time=0.8,
            rate_func=smooth,
        )

        gop = VGroup(
            _frame_card("I", GREEN, "весь кадр"),
            _frame_card("P", ORANGE, "из прошлого"),
            _frame_card("B", PURPLE, "с двух сторон"),
            _frame_card("P", ORANGE, "из прошлого"),
            _frame_card("P", ORANGE, "из прошлого"),
            _frame_card("I", GREEN, "новый якорь"),
        ).arrange(RIGHT, buff=0.18)
        gop.move_to(UP * 0.55)

        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.1) for c in gop], lag_ratio=0.08), run_time=0.95, rate_func=smooth)

        i_note = Text("I: DCT + квантование, как JPEG", font_size=18, color=GREEN)
        i_note.next_to(gop, DOWN, buff=0.28)
        self.play(Indicate(gop[0], color=GREEN, scale_factor=1.07), FadeIn(i_note), run_time=0.65, rate_func=smooth)
        self.wait(0.1)
        self.play(FadeOut(i_note), run_time=0.22)

        arrows = VGroup()
        for src, dst in [(0, 1), (1, 3), (3, 4)]:
            arrows.add(
                Arrow(
                    gop[src].get_top() + UP * 0.08,
                    gop[dst].get_top() + UP * 0.08,
                    buff=0.05,
                    stroke_width=3.5,
                    color=ORANGE,
                    max_tip_length_to_length_ratio=0.14,
                )
            )
        b_arrows = VGroup(
            Arrow(
                gop[1].get_right() + DOWN * 0.55,
                gop[2].get_left() + DOWN * 0.55,
                buff=0.04,
                stroke_width=2.5,
                color=PURPLE,
                max_tip_length_to_length_ratio=0.2,
            ),
            Arrow(
                gop[3].get_left() + DOWN * 0.55,
                gop[2].get_right() + DOWN * 0.55,
                buff=0.04,
                stroke_width=2.5,
                color=PURPLE,
                max_tip_length_to_length_ratio=0.2,
            ),
        )
        p_note = Text("P: движение + residual     B: смотрит в обе стороны", font_size=18, color=ORANGE)
        p_note.next_to(gop, DOWN, buff=0.28)
        self.play(
            LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.12),
            FadeIn(b_arrows),
            FadeIn(p_note),
            run_time=0.95,
            rate_func=smooth,
        )
        self.wait(0.12)

        heights = [1.05, 0.28, 0.16, 0.26, 0.24, 0.98]
        colors = [GREEN, ORANGE, PURPLE, ORANGE, ORANGE, GREEN]
        baseline_y = -2.35
        placed = VGroup()
        for h, col, card in zip(heights, colors, gop):
            bar = Rectangle(width=0.4, height=h, color=col, fill_color=col, fill_opacity=0.9, stroke_width=0)
            bar.move_to([card.get_center()[0], baseline_y + h / 2, 0])
            placed.add(bar)
        bit_lbl = Text("бюджет битов: I ≫ P/B", font_size=18, color=GRAY)
        bit_lbl.next_to(placed, UP, buff=0.12)

        self.play(FadeOut(p_note), run_time=0.18)
        self.play(
            LaggedStart(*[GrowFromEdge(b, DOWN) for b in placed], lag_ratio=0.07),
            FadeIn(bit_lbl),
            run_time=0.9,
            rate_func=smooth,
        )

        take = Text("H.265 делает тот же трюк лучше: −30…50% битрейта при том же качестве", font_size=18, color=WHITE)
        take.to_edge(DOWN, buff=0.2)
        self.play(FadeIn(take, shift=UP * 0.08), run_time=0.5, rate_func=smooth)
        self.wait(0.5)
