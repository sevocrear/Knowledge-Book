---
name: dl-viz
description: >
  Create professional animated visualizations (MP4 video and GIF) of Deep Learning and Computer Vision concepts using Manim Community Edition.
  Use this skill whenever the user asks to visualize, animate, illustrate, or explain any DL/CV concept — including but not limited to:
  neural network architectures (CNN, Transformer, ViT, ResNet, YOLO), training algorithms (backpropagation, gradient descent, loss landscapes),
  CV operations (convolution, pooling, attention, feature maps, NMS), and inference/data pipelines.
  Trigger on phrases like "visualize X", "animate how X works", "make a gif of X", "explain X visually", "create a video showing X",
  or any time the user wants to explain a DL/CV concept to an audience. Produces downloadable MP4 and/or GIF files.
---

# DL/CV Visualization Skill

Produces high-quality animated visualizations of Deep Learning and Computer Vision concepts using **Manim Community Edition**.

Output formats: **MP4** (primary, full quality) + **GIF** (for Telegram/social, auto-generated from MP4).

---

## Quick Start Workflow

1. **Understand the concept** — clarify what to show and the target audience (beginner/researcher/engineer)
2. **Choose a visual approach** — consult `references/dl-patterns.md` for the right pattern
3. **Install Manim** if not present (see Setup below)
4. **Write the Manim scene** — follow the Style Guide section
5. **Render MP4**, then **convert to GIF** if needed
6. **Present files** to user

---

## Topic-local output layout (required)

This repository is a knowledge base with topic directories under `topics/<topic-slug>/`.

When you create visualizations for a topic, **all sources and outputs must live inside that topic**:

- `topics/<topic-slug>/visualizations/manim/` — Manim sources (e.g. `scene.py`, helper scripts)
- `topics/<topic-slug>/assets/visualizations/` — final artifacts to embed in Markdown:
  - `*.mp4` (primary)
  - `*.gif` (optional, derived from mp4)
- `topics/<topic-slug>/visualizations/manim/.media/` — Manim working directory (heavy intermediates)

Practical rule:

- Keep `assets/visualizations/` clean and stable (only final files referenced from `README.md`).
- Put Manim’s `--media_dir` inside `visualizations/manim/.media/` so intermediates don’t pollute the repo root.

---

## Setup

If there is parent `uv` project set up, install manim via `uv`

### Install Manim (run once per session)

```bash
pip install manim imageio imageio-ffmpeg --break-system-packages -q
```

Verify:
```bash
python -c "import manim; print(manim.__version__)"
```

If system deps missing (Cairo, Pango), install:
```bash
apt-get install -y libcairo2-dev libpango1.0-dev ffmpeg 2>/dev/null | tail -1
```

### Render command

```bash
# From the repo root, set your topic slug once:
TOPIC_SLUG="your-topic-slug"
TOPIC_DIR="topics/${TOPIC_SLUG}"

# Create the required dirs
mkdir -p "${TOPIC_DIR}/visualizations/manim" \
         "${TOPIC_DIR}/visualizations/manim/.media" \
         "${TOPIC_DIR}/assets/visualizations"

# Put your Manim scene into:
#   ${TOPIC_DIR}/visualizations/manim/scene.py
# and run commands from that directory:
cd "${TOPIC_DIR}/visualizations/manim"

# HD MP4 (1920x1080, recommended)
manim -qh scene.py SceneName -o "animation" --media_dir "./.media"

# Medium quality for faster iteration
manim -qm scene.py SceneName -o "animation" --media_dir "./.media"

# Then convert to GIF (keep under 15MB for Telegram)
python convert_gif.py
```

GIF conversion script (`convert_gif.py`):
```python
import imageio, os, glob

# Find the rendered MP4
topic_dir = os.environ.get("TOPIC_DIR", os.path.abspath(os.path.join(os.getcwd(), "..", "..")))
assets_dir = os.path.join(topic_dir, "assets", "visualizations")
media_dir = os.path.join(os.getcwd(), ".media")

mp4_files = glob.glob(os.path.join(media_dir, "**", "*.mp4"), recursive=True)
mp4_path = sorted(mp4_files)[-1]  # most recent

reader = imageio.get_reader(mp4_path)
fps = reader.get_meta_data()['fps']
gif_fps = min(fps, 15)  # cap at 15fps for size
stride = max(1, int(fps / gif_fps))

frames = [frame for i, frame in enumerate(reader) if i % stride == 0]
os.makedirs(assets_dir, exist_ok=True)
gif_path = os.path.join(assets_dir, "animation.gif")
imageio.mimsave(gif_path, frames, fps=gif_fps, loop=0)
print(f"GIF saved: {len(frames)} frames @ {gif_fps}fps")
```

---

## Style Guide

### Color Palette (use CSS variables as Manim colors)

```python
# Paste at top of every scene
BG       = "#0F1117"   # near-black background
BLUE_PRI = "#4A9EFF"   # primary: neurons, nodes, data flow
BLUE_DIM = "#1E3A5F"   # dim blue: inactive / background elements  
ORANGE   = "#FF7A2F"   # activations, highlights, attention weights
GREEN    = "#3DBA7A"   # weights, parameters, positive gradients
RED_NEG  = "#FF4D6D"   # negative gradients, loss, errors
GRAY     = "#6B7280"   # labels, axes, secondary text
WHITE    = "#F0F4FF"   # primary text, titles
PURPLE   = "#9B72CF"   # embeddings, latent space, features
```

### Scene Template

```python
from manim import *

# Color constants
BG       = "#0F1117"
BLUE_PRI = "#4A9EFF"
BLUE_DIM = "#1E3A5F"
ORANGE   = "#FF7A2F"
GREEN    = "#3DBA7A"
RED_NEG  = "#FF4D6D"
GRAY     = "#6B7280"
WHITE    = "#F0F4FF"
PURPLE   = "#9B72CF"

class MyScene(Scene):
    def construct(self):
        self.camera.background_color = BG
        
        # Title
        title = Text("Concept Name", font_size=42, color=WHITE).to_edge(UP, buff=0.4)
        subtitle = Text("subtitle here", font_size=22, color=GRAY).next_to(title, DOWN, buff=0.15)
        self.play(FadeIn(title), FadeIn(subtitle))
        self.wait(0.5)
        
        # ... main animation ...
        
        self.wait(1.5)
```

### Animation Principles

- **Pacing**: Add `self.wait(0.3–0.8)` between steps so viewers can follow
- **Duration**: Aim for 15–60 seconds (most concepts). Use `run_time=1.5` for key transforms
- **Labels**: Always label axes, tensors, and operations. Use `font_size=20–28`
- **Highlighting**: Use `Indicate()`, `Circumscribe()`, or color flashes to draw attention
- **Flow**: Show data/signal flowing left→right or top→bottom
- **Arrows**: Use `Arrow` for connections, `CurvedArrow` for skip connections
- **Tensors**: Represent as colored rectangles/cubes with dimension labels
- **Keep it dense but readable**: 1080p gives room — use it

### Common Manim Patterns

```python
# Animated arrow (data flow)
arrow = Arrow(start.get_right(), end.get_left(), color=BLUE_PRI, buff=0.1)
self.play(GrowArrow(arrow))

# Pulsing highlight
self.play(Indicate(obj, color=ORANGE, scale_factor=1.2))

# Color flash (activation firing)
self.play(obj.animate.set_fill(ORANGE, opacity=0.9), run_time=0.3)
self.play(obj.animate.set_fill(BLUE_PRI, opacity=0.7), run_time=0.3)

# Moving dot along path (signal propagation)
dot = Dot(color=ORANGE, radius=0.08)
dot.move_to(arrow.get_start())
self.play(MoveAlongPath(dot, arrow), run_time=1.0)

# Tensor reshape animation
self.play(Transform(tensor_3d, tensor_flat))

# Write equation
eq = MathTex(r"\sigma(Wx + b)", color=WHITE, font_size=36)
self.play(Write(eq))

# Fade and replace
self.play(FadeOut(old), FadeIn(new))
```

---

## Concept-Specific Patterns

Read `references/dl-patterns.md` for detailed scene blueprints for:

- **Convolution** — sliding kernel with sum animation
- **Self-Attention** — Q/K/V matrices, softmax heatmap, weighted sum
- **Backpropagation** — forward pass + gradient flow reverse
- **Gradient Descent** — loss landscape with rolling ball
- **CNN Architecture** — layer-by-layer feature map pipeline
- **Transformer Block** — full encoder block with all sub-components
- **YOLO/Detection** — grid cells, anchor boxes, NMS
- **Embedding Space** — 3D PCA/t-SNE scatter with clusters

Always check `dl-patterns.md` before writing a scene — it contains working code snippets.

---

## Output & File Delivery

After rendering:

```bash
# Copy the most recent MP4 from Manim media dir into topic assets
TOPIC_SLUG="your-topic-slug"
TOPIC_DIR="topics/${TOPIC_SLUG}"
cd "${TOPIC_DIR}/visualizations/manim"

latest_mp4="$(ls -t ./.media/**/*.mp4 ./.media/**/**/*.mp4 2>/dev/null | head -n 1)"
cp "${latest_mp4}" "../assets/visualizations/animation.mp4"
```

Now you should have:

- `topics/<topic-slug>/assets/visualizations/animation.mp4`
- `topics/<topic-slug>/assets/visualizations/animation.gif` (if generated)

Embed in Markdown using relative paths from the topic `README.md`, e.g.:

- `![](./assets/visualizations/animation.gif)`

---

## Quality Checklist

Before presenting to user:
- [ ] Background is dark (`#0F1117`), not white
- [ ] All elements are labeled
- [ ] Animation has a clear narrative arc (setup → show → highlight)
- [ ] No crowded frames — max 6–8 elements visible at once
- [ ] GIF is under 15MB (if not, reduce fps or duration)
- [ ] Title/subtitle present
- [ ] Smooth timing (no abrupt jumps)
