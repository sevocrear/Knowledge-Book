"""
RAG pipeline: query → retrieve Top-K → augment prompt → LLM (~30–40 s).
"""

from __future__ import annotations

from manim import *

BG = "#0F1117"
WHITE = "#F0F4FF"
GRAY = "#6B7280"
BLUE = "#4A9EFF"
ORANGE = "#FF7A2F"
GREEN = "#3DBA7A"
RED = "#FF4D6D"
PURPLE = "#9B72CF"


class RAGPipelineScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        title = Text("Retrieval-Augmented Generation", font_size=36, color=WHITE).to_edge(UP, buff=0.28)
        sub = Text("сначала факты из базы, потом генерация", font_size=22, color=GRAY).next_to(title, DOWN, buff=0.08)
        self.play(FadeIn(title, shift=DOWN * 0.08), FadeIn(sub, shift=DOWN * 0.06), run_time=0.8, rate_func=smooth)

        llm_only = RoundedRectangle(width=2.4, height=1.0, corner_radius=0.12, color=RED, stroke_width=3)
        llm_txt = Text("только LLM", font_size=24, color=WHITE).move_to(llm_only)
        risk = Text("риск галлюцинаций / устаревших фактов", font_size=18, color=GRAY).next_to(llm_only, DOWN, buff=0.15)
        g0 = VGroup(llm_only, llm_txt)
        g0.shift(UP * 0.35)
        self.play(FadeIn(g0), FadeIn(risk), run_time=0.55, rate_func=smooth)
        cross = Text("✗", font_size=56, color=RED).move_to(llm_only)
        self.play(FadeIn(cross, scale=0.5), run_time=0.35)
        self.wait(0.2)
        self.play(FadeOut(g0), FadeOut(risk), FadeOut(cross), run_time=0.4)

        # Pipeline left to right
        q_box = RoundedRectangle(width=1.5, height=0.65, corner_radius=0.08, color=BLUE, fill_color="#1E3A5F", fill_opacity=0.9)
        q_txt = Text("Query", font_size=22, color=WHITE).move_to(q_box)
        query = VGroup(q_box, q_txt).shift(LEFT * 4.2 + UP * 0.2)

        emb = Circle(radius=0.35, color=ORANGE, fill_color=ORANGE, fill_opacity=0.35, stroke_width=2)
        emb_l = Text("embed", font_size=18, color=WHITE).move_to(emb.get_center())
        embed_g = VGroup(emb, emb_l).shift(LEFT * 2.2 + UP * 0.2)

        a1 = Arrow(query.get_right(), embed_g.get_left(), buff=0.08, color=GRAY, stroke_width=4)

        docs = VGroup()
        for i in range(8):
            d = Dot(radius=0.09, color=GRAY).shift(RIGHT * (i % 4) * 0.22 + DOWN * (i // 4) * 0.22)
            docs.add(d)
        docs.move_to(RIGHT * 0.3 + UP * 0.15)
        store = Text("векторное хранилище\nчанков", font_size=18, color=GRAY).next_to(docs, DOWN, buff=0.2)
        chunk_g = VGroup(docs, store)

        a2 = Arrow(embed_g.get_right(), docs.get_left() + LEFT * 0.5, buff=0.1, color=GRAY, stroke_width=4)

        self.play(FadeIn(query, shift=RIGHT * 0.15), run_time=0.45, rate_func=smooth)
        self.play(GrowArrow(a1), FadeIn(embed_g, shift=RIGHT * 0.12), run_time=0.5, rate_func=smooth)
        self.play(GrowArrow(a2), FadeIn(chunk_g, shift=LEFT * 0.1), run_time=0.55, rate_func=smooth)

        # Highlight top-3 dots
        top = [docs[2], docs[5], docs[1]]
        self.play(*[Indicate(d, color=GREEN, scale_factor=1.8) for d in top], run_time=0.65)
        topk = Text("Top-K релевантных чанков", font_size=20, color=GREEN).next_to(chunk_g, UP, buff=0.35)
        self.play(FadeIn(topk, shift=UP * 0.08), run_time=0.4, rate_func=smooth)
        self.wait(0.15)

        ctx = RoundedRectangle(width=2.8, height=1.15, corner_radius=0.1, color=PURPLE, stroke_width=2, fill_opacity=0.2)
        ctx_t = Text("контекст в промпт", font_size=22, color=WHITE).move_to(ctx.get_center())
        ctx_g = VGroup(ctx, ctx_t).shift(RIGHT * 3.5 + UP * 0.2)
        a3 = Arrow(docs.get_right() + RIGHT * 0.15, ctx_g.get_left(), buff=0.08, color=PURPLE, stroke_width=4)
        self.play(GrowArrow(a3), FadeIn(ctx_g, shift=LEFT * 0.12), run_time=0.55, rate_func=smooth)

        llm = RoundedRectangle(width=1.6, height=0.75, corner_radius=0.1, color=BLUE, stroke_width=3)
        llm_l = Text("LLM", font_size=26, color=WHITE).move_to(llm)
        llm_g = VGroup(llm, llm_l).shift(RIGHT * 5.85 + UP * 0.2)
        a4 = Arrow(ctx_g.get_right(), llm_g.get_left(), buff=0.08, color=BLUE, stroke_width=4)
        self.play(GrowArrow(a4), FadeIn(llm_g, shift=LEFT * 0.1), run_time=0.5, rate_func=smooth)

        ans = Text("ответ с опорой\nна источники", font_size=20, color=GREEN).next_to(llm_g, DOWN, buff=0.35)
        self.play(FadeIn(ans, shift=UP * 0.08), run_time=0.45, rate_func=smooth)

        recap = Text("Retrieve → Augment → Generate", font_size=26, color=WHITE).to_edge(DOWN, buff=0.2)
        self.play(FadeIn(recap, shift=UP * 0.1), run_time=0.5, rate_func=smooth)
        self.wait(0.55)
