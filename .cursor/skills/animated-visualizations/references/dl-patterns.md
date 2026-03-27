# DL/CV Visualization Patterns

Working scene blueprints for the most common concepts. Copy and adapt as needed.

---

## Table of Contents

1. [Convolution Operation](#1-convolution-operation)
2. [Self-Attention Mechanism](#2-self-attention-mechanism)
3. [Gradient Descent on Loss Landscape](#3-gradient-descent-on-loss-landscape)
4. [Backpropagation](#4-backpropagation)
5. [CNN Feature Pipeline](#5-cnn-feature-pipeline)
6. [Transformer Encoder Block](#6-transformer-encoder-block)
7. [Non-Maximum Suppression (NMS)](#7-non-maximum-suppression-nms)
8. [Embedding / Latent Space](#8-embedding--latent-space)

---

## 1. Convolution Operation

**Visual approach**: 3×3 grid (input) + sliding kernel + output cell being filled one by one.

```python
class ConvolutionScene(Scene):
    def construct(self):
        self.camera.background_color = "#0F1117"
        
        # Build 5x5 input grid
        cell_size = 0.6
        input_grid = VGroup()
        values = [[1,2,0,1,0],[0,1,1,2,1],[1,0,2,1,0],[0,1,0,1,2],[1,2,1,0,1]]
        cells = {}
        for r in range(5):
            for c in range(5):
                sq = Square(cell_size, color="#1E3A5F", fill_color="#1E3A5F", fill_opacity=0.8)
                sq.move_to(LEFT*3 + RIGHT*c*cell_size + DOWN*r*cell_size)
                val = Text(str(values[r][c]), font_size=18, color="#F0F4FF")
                val.move_to(sq)
                cells[(r,c)] = VGroup(sq, val)
                input_grid.add(cells[(r,c)])
        
        # 3x3 kernel
        kernel_vals = [[1,0,-1],[0,1,0],[-1,0,1]]
        kernel = VGroup()
        for r in range(3):
            for c in range(3):
                sq = Square(cell_size, color="#FF7A2F", fill_color="#FF7A2F", fill_opacity=0.3,
                           stroke_width=2)
                sq.move_to(RIGHT*1.5 + RIGHT*c*cell_size + DOWN*r*cell_size)
                val = Text(str(kernel_vals[r][c]), font_size=16, color="#FF7A2F")
                val.move_to(sq)
                kernel.add(VGroup(sq, val))
        
        # Labels
        inp_label = Text("Input", font_size=24, color="#6B7280").next_to(input_grid, UP)
        k_label = Text("Kernel (3×3)", font_size=24, color="#FF7A2F").next_to(kernel, UP)
        
        title = Text("2D Convolution", font_size=40, color="#F0F4FF").to_edge(UP, buff=0.3)
        
        self.play(FadeIn(title), FadeIn(input_grid), FadeIn(inp_label))
        self.wait(0.3)
        self.play(FadeIn(kernel), FadeIn(k_label))
        self.wait(0.5)
        
        # Slide kernel over input (show 3 positions)
        output_label = Text("Output", font_size=24, color="#3DBA7A").move_to(RIGHT*4.5 + UP*1.5)
        self.play(FadeIn(output_label))
        
        for pos, (kr, kc) in enumerate([(0,0), (0,1), (1,1)]):
            # Move kernel highlight to position on input
            highlight = SurroundingRectangle(
                VGroup(*[cells[(kr+dr, kc+dc)] for dr in range(3) for dc in range(3)]),
                color="#FF7A2F", buff=0.02, stroke_width=3
            )
            self.play(Create(highlight), run_time=0.4)
            
            # Show dot product result appearing in output
            result_sq = Square(cell_size, color="#3DBA7A", fill_color="#3DBA7A", fill_opacity=0.5)
            result_sq.move_to(RIGHT*4.2 + RIGHT*kc*cell_size + DOWN*kr*cell_size)
            result_val = Text("✓", font_size=20, color="#3DBA7A").move_to(result_sq)
            self.play(FadeIn(result_sq), Write(result_val), run_time=0.5)
            self.wait(0.3)
            self.play(FadeOut(highlight))
        
        # Equation
        eq = MathTex(r"(I * K)[i,j] = \sum_{m}\sum_{n} I[i{+}m,\, j{+}n]\cdot K[m,n]",
                     font_size=28, color="#F0F4FF")
        eq.to_edge(DOWN, buff=0.5)
        self.play(Write(eq))
        self.wait(2)
```

---

## 2. Self-Attention Mechanism

**Visual approach**: Token sequence → Q/K/V projections → attention score heatmap → weighted sum output.

```python
class SelfAttentionScene(Scene):
    def construct(self):
        self.camera.background_color = "#0F1117"
        tokens = ["The", "cat", "sat", "mat"]
        n = len(tokens)
        
        title = Text("Self-Attention", font_size=40, color="#F0F4FF").to_edge(UP, buff=0.3)
        self.play(FadeIn(title))
        
        # Token boxes
        token_boxes = VGroup()
        for i, tok in enumerate(tokens):
            box = RoundedRectangle(width=1.2, height=0.6, corner_radius=0.1,
                                   color="#4A9EFF", fill_color="#1E3A5F", fill_opacity=0.9)
            box.move_to(LEFT*3.5 + RIGHT*i*1.5 + UP*1.5)
            label = Text(tok, font_size=20, color="#F0F4FF").move_to(box)
            token_boxes.add(VGroup(box, label))
        self.play(FadeIn(token_boxes))
        self.wait(0.3)
        
        # Q, K, V labels
        for label_txt, color, offset in [("Q", "#FF7A2F", -0.5), ("K", "#3DBA7A", 0), ("V", "#9B72CF", 0.5)]:
            lbl = Text(f"{label_txt}", font_size=22, color=color)
            lbl.move_to(RIGHT*3 + UP*(1.5 - offset))
            self.play(FadeIn(lbl), run_time=0.3)
        
        # Attention heatmap (4x4)
        attn_weights = [[0.7,0.1,0.1,0.1],[0.2,0.5,0.2,0.1],[0.1,0.3,0.4,0.2],[0.1,0.1,0.2,0.6]]
        heatmap = VGroup()
        for r in range(n):
            for c in range(n):
                w = attn_weights[r][c]
                alpha = 0.2 + w * 0.8
                sq = Square(0.7, fill_color="#FF7A2F", fill_opacity=alpha,
                           color="#FF7A2F", stroke_width=0.5)
                sq.move_to(DOWN*0.5 + LEFT*1.0 + RIGHT*c*0.72 + DOWN*r*0.72)
                wt = Text(f"{w:.1f}", font_size=12, color="#F0F4FF").move_to(sq)
                heatmap.add(VGroup(sq, wt))
        
        hm_label = Text("Attention Weights (softmax)", font_size=20, color="#6B7280")
        hm_label.next_to(heatmap, UP, buff=0.2)
        self.play(FadeIn(hm_label), FadeIn(heatmap), run_time=1.2)
        self.wait(0.5)
        
        # Highlight diagonal (attending to self)
        for i in range(n):
            diag = heatmap[i*n + i][0]
            self.play(Indicate(diag, color="#FF7A2F", scale_factor=1.3), run_time=0.3)
        
        # Equation
        eq = MathTex(r"\text{Attn}(Q,K,V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V",
                     font_size=30, color="#F0F4FF").to_edge(DOWN, buff=0.5)
        self.play(Write(eq))
        self.wait(2)
```

---

## 3. Gradient Descent on Loss Landscape

**Visual approach**: 3D parabolic surface (or 2D contour) with a ball rolling down to the minimum.

```python
class GradientDescentScene(Scene):
    def construct(self):
        self.camera.background_color = "#0F1117"
        
        title = Text("Gradient Descent", font_size=40, color="#F0F4FF").to_edge(UP, buff=0.3)
        subtitle = Text("Following the negative gradient to minimize loss",
                        font_size=22, color="#6B7280").next_to(title, DOWN, buff=0.15)
        self.play(FadeIn(title), FadeIn(subtitle))
        
        # Axes
        axes = Axes(
            x_range=[-3, 3, 1], y_range=[0, 5, 1],
            x_length=8, y_length=5,
            axis_config={"color": "#6B7280", "stroke_width": 2},
        ).shift(DOWN*0.3)
        x_label = axes.get_x_axis_label(Text("θ (parameter)", font_size=18, color="#6B7280"))
        y_label = axes.get_y_axis_label(Text("Loss", font_size=18, color="#6B7280"), direction=LEFT)
        self.play(Create(axes), FadeIn(x_label), FadeIn(y_label))
        
        # Loss curve: L(θ) = θ² + 0.5
        loss_curve = axes.plot(lambda x: x**2 + 0.5, color="#4A9EFF", stroke_width=3)
        self.play(Create(loss_curve), run_time=1.5)
        
        # Minimum label
        min_dot = Dot(axes.c2p(0, 0.5), color="#3DBA7A", radius=0.1)
        min_label = Text("minimum", font_size=18, color="#3DBA7A").next_to(min_dot, DOWN)
        self.play(FadeIn(min_dot), FadeIn(min_label))
        
        # Ball rolling down
        start_x = 2.5
        ball = Dot(axes.c2p(start_x, start_x**2 + 0.5), color="#FF7A2F", radius=0.15)
        self.play(FadeIn(ball))
        
        # Gradient steps
        lr = 0.4
        x = start_x
        for step in range(7):
            grad = 2 * x
            x_new = x - lr * grad
            x_new = max(x_new, 0.05)  # don't overshoot
            
            # Show gradient arrow
            grad_arrow = Arrow(
                axes.c2p(x, x**2 + 0.5),
                axes.c2p(x_new, x_new**2 + 0.5),
                color="#FF7A2F", buff=0, stroke_width=2,
                max_tip_length_to_length_ratio=0.3
            )
            self.play(GrowArrow(grad_arrow), run_time=0.3)
            self.play(
                ball.animate.move_to(axes.c2p(x_new, x_new**2 + 0.5)),
                run_time=0.4
            )
            self.play(FadeOut(grad_arrow), run_time=0.1)
            x = x_new
        
        # Update rule
        eq = MathTex(r"\theta \leftarrow \theta - \eta \cdot \nabla_\theta \mathcal{L}",
                     font_size=32, color="#F0F4FF").to_edge(DOWN, buff=0.5)
        self.play(Write(eq))
        self.wait(2)
```

---

## 4. Backpropagation

**Visual approach**: Simple 3-layer MLP, forward pass (blue signal), then reverse gradient flow (orange/red).

```python
class BackpropScene(Scene):
    def construct(self):
        self.camera.background_color = "#0F1117"
        
        title = Text("Backpropagation", font_size=40, color="#F0F4FF").to_edge(UP, buff=0.3)
        self.play(FadeIn(title))
        
        # Network layers: 3 → 4 → 3 → 1
        layer_sizes = [3, 4, 3, 1]
        layer_x = [-4, -1.5, 1, 3.5]
        node_spacing = 1.0
        
        all_nodes = []
        for l, (size, x) in enumerate(zip(layer_sizes, layer_x)):
            layer_nodes = []
            offset = (size - 1) * node_spacing / 2
            for n in range(size):
                y = offset - n * node_spacing
                node = Circle(radius=0.3, color="#4A9EFF",
                             fill_color="#1E3A5F", fill_opacity=0.9)
                node.move_to(np.array([x, y, 0]))
                layer_nodes.append(node)
            all_nodes.append(layer_nodes)
        
        layer_labels = ["Input\nx", "Hidden\nh₁", "Hidden\nh₂", "Output\nŷ"]
        for l, (nodes, lbl) in enumerate(zip(all_nodes, layer_labels)):
            grp = VGroup(*nodes)
            label = Text(lbl, font_size=16, color="#6B7280").next_to(grp, DOWN, buff=0.3)
            self.play(FadeIn(grp), FadeIn(label), run_time=0.4)
        
        # Draw connections
        edges = []
        for l in range(len(all_nodes) - 1):
            for n1 in all_nodes[l]:
                for n2 in all_nodes[l+1]:
                    edge = Line(n1.get_center(), n2.get_center(),
                               color="#1E3A5F", stroke_width=1.5, stroke_opacity=0.6)
                    edges.append(edge)
                    self.add(edge)
        
        self.wait(0.5)
        
        # Forward pass — blue signal
        fwd_label = Text("Forward Pass →", font_size=22, color="#4A9EFF").to_edge(LEFT).shift(DOWN*2.5)
        self.play(FadeIn(fwd_label))
        for l in range(len(all_nodes)):
            self.play(*[node.animate.set_fill("#4A9EFF", opacity=0.9)
                       for node in all_nodes[l]], run_time=0.4)
            self.wait(0.1)
        
        self.wait(0.4)
        
        # Backward pass — orange gradient
        bwd_label = Text("← Backward Pass", font_size=22, color="#FF7A2F").to_edge(RIGHT).shift(DOWN*2.5)
        self.play(FadeIn(bwd_label))
        for l in reversed(range(len(all_nodes))):
            self.play(*[node.animate.set_fill("#FF7A2F", opacity=0.9)
                       for node in all_nodes[l]], run_time=0.4)
            self.wait(0.1)
        
        eq = MathTex(r"\frac{\partial \mathcal{L}}{\partial W_l} = \delta_{l+1} \cdot a_l^T",
                     font_size=30, color="#F0F4FF").to_edge(DOWN, buff=0.4)
        self.play(Write(eq))
        self.wait(2)
```

---

## 5. CNN Feature Pipeline

**Visual approach**: Input image → Conv → ReLU → Pool → Feature maps shrinking and deepening left→right.

Key elements:
- Represent feature maps as stacked colored rectangles (width × height × depth)
- Show spatial dimensions decreasing, channel count increasing
- Animate a "slice" being extracted at each stage

```python
# Tensor block helper
def make_tensor_block(scene, x, y, w, h, d, color, label_text):
    """Draw a 3D-ish tensor block at position (x,y)"""
    # Front face
    front = Rectangle(width=w, height=h, fill_color=color, fill_opacity=0.7,
                      color=color, stroke_width=2)
    front.move_to(np.array([x, y, 0]))
    # Depth suggestion (offset lines)
    depth_lines = VGroup()
    for i in range(1, min(d+1, 4)):
        offset_rect = Rectangle(width=w, height=h, color=color,
                               fill_opacity=0.0, stroke_width=0.8, stroke_opacity=0.4)
        offset_rect.move_to(np.array([x + i*0.08, y + i*0.08, 0]))
        depth_lines.add(offset_rect)
    label = Text(label_text, font_size=14, color="#6B7280")
    label.next_to(front, DOWN, buff=0.2)
    return VGroup(depth_lines, front, label)

# In construct():
# stages = [
#   (2.0, 1.5, 1, "#4A9EFF", "224×224\n3ch"),   # input
#   (1.5, 1.5, 3, "#FF7A2F", "112×112\n64ch"),   # after conv1
#   (1.0, 1.5, 5, "#3DBA7A", "56×56\n128ch"),    # after conv2
#   (0.6, 1.5, 7, "#9B72CF", "28×28\n256ch"),    # after conv3
# ]
```

---

## 6. Transformer Encoder Block

**Visual approach**: Stacked boxes showing: Input Embeddings → Multi-Head Attention → Add & Norm → FFN → Add & Norm → Output.

Use `Rectangle` blocks with connecting arrows. Animate data flowing upward through the stack.

Key visual pattern:
- Each sub-block is a labeled rounded rectangle
- Skip connections shown as curved arrows bypassing the block
- Color-code: attention=orange, FFN=green, LayerNorm=gray, residual=blue

---

## 7. Non-Maximum Suppression (NMS)

**Visual approach**: Image background (rectangle) with overlapping bounding boxes → score bars → suppression animation.

Steps to animate:
1. Show multiple overlapping boxes (different transparencies by score)
2. Highlight highest-score box in green
3. Red X and FadeOut for suppressed boxes (IoU > threshold)
4. Repeat for remaining boxes

```python
# Box helper
def make_bbox(x, y, w, h, score, is_best=False):
    color = "#3DBA7A" if is_best else "#FF4D6D"
    box = Rectangle(width=w, height=h, color=color,
                   fill_color=color, fill_opacity=0.15, stroke_width=2.5)
    box.move_to(np.array([x, y, 0]))
    score_label = Text(f"{score:.2f}", font_size=16, color=color)
    score_label.next_to(box, UP, buff=0.05)
    return VGroup(box, score_label)
```

---

## 8. Embedding / Latent Space

**Visual approach**: 3D scatter (or 2D with NumberPlane) of points, colored by class, with cluster labels.

Use `Dot` objects on a `NumberPlane`. Animate points moving from random → clustered (showing representation learning).

```python
# Random → clustered animation
import numpy as np

# Cluster centers for 3 classes
centers = [np.array([-2, 1, 0]), np.array([2, 1, 0]), np.array([0, -2, 0])]
colors  = ["#4A9EFF", "#FF7A2F", "#3DBA7A"]

dots = []
for cls, (center, color) in enumerate(zip(centers, colors)):
    for _ in range(8):
        start_pos = np.random.uniform(-3, 3, 3); start_pos[2] = 0
        dot = Dot(point=start_pos, color=color, radius=0.1)
        dots.append((dot, center))

# Animate convergence
self.play(*[dot.animate.move_to(center + np.random.randn(3)*0.4*np.array([1,1,0]))
           for dot, center in dots], run_time=2.0)
```
