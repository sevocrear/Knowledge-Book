---
name: dl-viz
description: >
  Create professional animated visualizations (MP4 primary, optional GIF) of Deep Learning and Computer Vision concepts using Manim Community Edition.
  Use this skill whenever the user asks to visualize, animate, illustrate, or explain any DL/CV concept — including but not limited to:
  neural network architectures (CNN, Transformer, ViT, ResNet, YOLO), training algorithms (backpropagation, gradient descent, loss landscapes),
  CV operations (convolution, pooling, attention, feature maps, NMS), and inference/data pipelines.
  Trigger on phrases like "visualize X", "animate how X works", "make a gif of X", "explain X visually", "create a video showing X",
  or any time the user wants to explain a DL/CV concept to an audience. Produces downloadable MP4 and/or GIF files.
---

# DL/CV Visualization Skill

Produces high-quality animated visualizations using **Manim Community Edition**.

**Primary artifact:** **MP4** (full quality, embedded in topic README via HTML5 `<video>`).  
**Optional:** **GIF** (preview, Telegram, or Markdown viewers without `<video>`), generated with the shared repo script `scripts/viz/mp4_to_gif.py` (from repository root).

---

## Quick Start Workflow

1. **Understand the concept** — audience (beginner / engineer) and **one main idea per clip** (see Pedagogy below).
2. **Choose a visual approach** — `references/dl-patterns.md` + **3Blue1Brown-inspired polish** (below).
3. **Install Manim** — `uv sync --group viz` from repo root (see Setup).
4. **Write the Manim scene** — `topics/<topic-slug>/visualizations/manim/scene.py` only (no local `convert_gif.py`).
5. **Render MP4** → copy into `topics/<topic-slug>/assets/visualizations/<name>.mp4`.
6. **Optional GIF** — `uv run python scripts/viz/mp4_to_gif.py <path-to-mp4> -o <path-to-gif>`.
7. **Embed in README** — default: `<video src="./assets/visualizations/<name>.mp4" ...>`; fallback: `![](./assets/visualizations/<name>.gif)`.

---

## Topic-local output layout (required)

- `topics/<topic-slug>/visualizations/manim/` — Manim sources (`scene.py`, helpers).
- `topics/<topic-slug>/assets/visualizations/` — committed artifacts referenced from README:
  - **`<name>.mp4`** (required for default embed)
  - **`<name>.gif`** (optional fallback / preview)
- `topics/<topic-slug>/visualizations/manim/.media/` — Manim cache/output (do not commit; add to `.gitignore` if missing).

**Do not** duplicate conversion logic per topic — use **`scripts/viz/mp4_to_gif.py`** and **`scripts/viz/gif_to_mp4.py`** only.

---

## Setup (uv + optional system packages)

From the **knowledge-book repo root**:

```bash
uv sync --group viz
uv run python -c "import manim; print(manim.__version__)"
```

On Linux, if Cairo/Pango errors appear:

```bash
sudo apt-get install -y libcairo2-dev libpango1.0-dev ffmpeg
```

### Render MP4

```bash
TOPIC_SLUG="your-topic-slug"
TOPIC_DIR="topics/${TOPIC_SLUG}"
mkdir -p "${TOPIC_DIR}/visualizations/manim/.media" "${TOPIC_DIR}/assets/visualizations"
cd "${TOPIC_DIR}/visualizations/manim"

# Preview quality (fast)
uv run manim -ql scene.py YourSceneName -o "topic_viz" --media_dir "./.media"

# HD (final)
uv run manim -qh scene.py YourSceneName -o "topic_viz" --media_dir "./.media"
```

Copy the newest MP4 from `./.media/...` into `../../assets/visualizations/<descriptive_name>.mp4`, then optionally:

```bash
uv run python scripts/viz/mp4_to_gif.py \
  "topics/${TOPIC_SLUG}/assets/visualizations/<name>.mp4" \
  -o "topics/${TOPIC_SLUG}/assets/visualizations/<name>.gif"
```

(path to `scripts/` from repo root)

---

## Embedding in topic README

**Default (preferred):** HTML5 video with a **relative** `src` so GitHub / many viewers show controls:

```html
<video src="./assets/visualizations/<name>.mp4" controls muted loop playsinline width="100%"></video>
```

Add a **short line** under the video explaining what the clip shows (self-contained with the text).

**Fallback:** if the target viewer does not support `<video>` (some static site generators, plain MD preview), use:

```markdown
![](./assets/visualizations/<name>.gif)
```

You may commit **both** MP4 and GIF: README uses `<video>` first; GIF remains a portable fallback.

---

## Pedagogy, length, and clarity

- **Target length:** aim for **about 30–40 seconds maximum** per single MP4/GIF for one main idea. Tight pacing; avoid empty `wait()`s.
- **If you need more:** add a **second scene class** (second MP4) instead of one very long animation.
- **On-screen story (beats):** hook → “before” state → **one** central transform (labeled) → “after” → optional short `MathTex` (colors match diagram) → one-line takeaway.
- **Silent-friendly:** every important step has a **short label** (language aligned with the topic README).

---

## 3Blue1Brown-inspired polish

| Idea | Manim CE |
|------|-----------|
| Smooth motion | `rate_func=smooth` on `play()`; avoid linear-only pacing for the whole scene |
| Staggered reveals | `LaggedStart(..., lag_ratio=0.05–0.15)` |
| Entry | `FadeIn(obj, shift=UP*0.15)` |
| Focus | `Indicate`, `Circumscribe`, brief color pulses |
| Camera | `MovingCameraScene` only when it truly helps (zoom on formula / graph) |
| Math | Consistent colors for Q/K/V (or symbols matching geometry) |

Palette (teal/warm accents on dark):

```python
BG       = "#0F1117"
BLUE_PRI = "#4A9EFF"
BLUE_DIM = "#1E3A5F"
ORANGE   = "#FF7A2F"
GREEN    = "#3DBA7A"
RED_NEG  = "#FF4D6D"
GRAY     = "#6B7280"
WHITE    = "#F0F4FF"
PURPLE   = "#9B72CF"
TEAL     = "#2DD4BF"  # optional b1b-like accent
```

---

## Style Guide — scene template

```python
from manim import *

BG = "#0F1117"
WHITE = "#F0F4FF"
GRAY = "#6B7280"

class MyScene(Scene):
    def construct(self):
        self.camera.background_color = BG
        title = Text("Concept", font_size=42, color=WHITE).to_edge(UP, buff=0.4)
        self.play(FadeIn(title, shift=DOWN * 0.1), run_time=0.8, rate_func=smooth)
        # ...
```

---

## Coverage rule (not “every paragraph”)

- **One** canonical animation per **major** algorithm / pipeline / architecture in a topic.
- Before adding a new file, check whether `assets/visualizations/` already has a clip for that section; **extend** the existing scene when possible.

---

## Concept-Specific Patterns

See `references/dl-patterns.md` (convolution, attention, ViT-style blocks, RAG-style pipelines, etc.).

---

## Quality Checklist

- [ ] Dark background (`#0F1117`), labeled elements, ≤6–8 readable objects per beat
- [ ] Clear narrative arc; **~30–40 s** per clip (or split into two clips)
- [ ] `rate_func=smooth` (or equivalent) on major transitions; no harsh jumps
- [ ] MP4 in `assets/visualizations/` + README **`<video>`** default; GIF optional
- [ ] GIF under ~15 MB if committed (lower fps / stride via `mp4_to_gif.py`)
- [ ] No per-topic `convert_gif.py` — only `scripts/viz/*`

---

## Tests (CI)

Smoke tests for Manim scenes live under `tests/test_manim_topic_scenes.py` and are marked **`@pytest.mark.manim`**.

- Default (no viz group): `uv run pytest -m "not manim"`
- With Manim: `uv sync --group viz` then `uv run pytest -m manim` (requires Cairo/Pango/ffmpeg where applicable)
