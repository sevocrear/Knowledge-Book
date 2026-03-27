"""
What this script demonstrates:
- A minimal metric learning pipeline for a fine-grained-like setting ("15 very similar classes").
- Training an embedding model with Triplet Margin Loss + simple semi-hard mining inside a P×K batch.
- Evaluating retrieval metrics (Recall@K) before/after training.

Expected behavior:
- Recall@1 and Recall@5 should improve after training (on a held-out set generated with the same class structure).
"""

from __future__ import annotations

from dataclasses import dataclass

import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SynthConfig:
    num_classes: int = 7
    input_dim: int = 16
    embed_dim: int = 8
    # "Fine-grained": class means are close to each other.
    class_separation: float = 0.75
    within_class_noise: float = 0.7
    # Train/eval sizes per class
    train_per_class: int = 140
    eval_per_class: int = 80


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_synth_dataset(cfg: SynthConfig, *, seed: int, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      x: (N, input_dim)
      y: (N,)
    """
    g = torch.Generator().manual_seed(seed + (0 if split == "train" else 10_000))

    # Create a "ring" of close class centers to simulate subtle differences.
    angles = torch.linspace(0.0, 2 * math.pi, cfg.num_classes + 1)[:-1]
    centers_2d = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * cfg.class_separation

    # Lift to input_dim with a random projection, to avoid too-trivial geometry.
    proj = torch.randn(2, cfg.input_dim, generator=g)
    centers = centers_2d @ proj  # (C, D)

    n_per_class = cfg.train_per_class if split == "train" else cfg.eval_per_class
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for c in range(cfg.num_classes):
        noise = torch.randn(n_per_class, cfg.input_dim, generator=g) * cfg.within_class_noise
        x_c = centers[c].unsqueeze(0) + noise
        y_c = torch.full((n_per_class,), c, dtype=torch.long)
        xs.append(x_c)
        ys.append(y_c)

    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)

    # Shuffle
    perm = torch.randperm(x.shape[0], generator=g)
    return x[perm], y[perm]


class SmallEmbedder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=-1)


def _pca_2d(x: torch.Tensor) -> torch.Tensor:
    """
    x: (N, D) float
    returns: (N, 2) float, PCA projection (CPU).
    """
    x = x.detach().float().cpu()
    x = x - x.mean(dim=0, keepdim=True)
    _, _, vT = torch.linalg.svd(x, full_matrices=False)
    w = vT[:2].T  # (D,2)
    return x @ w


def _projection_matrix(
    x: torch.Tensor,
    y: torch.Tensor | None = None,
    *,
    method: str = "pca",
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Returns W: (D, 2) projection matrix computed ONCE and reused for frames.

    - method="pca": unsupervised, stable if W is fixed
    - method="lda": supervised (uses labels), usually shows class separation more clearly
    """
    x = x.detach().float().cpu()
    if method == "pca":
        x0 = x - x.mean(dim=0, keepdim=True)
        _, _, vT = torch.linalg.svd(x0, full_matrices=False)
        return vT[:2].T  # (D,2)

    if method != "lda":
        raise ValueError(f"Unknown projection method: {method}")
    if y is None:
        raise ValueError("LDA projection requires labels y.")

    y = y.detach().cpu()
    classes = torch.unique(y)
    mu = x.mean(dim=0, keepdim=True)  # (1,D)

    # Between-class scatter Sb and within-class scatter Sw
    d = x.shape[1]
    Sb = torch.zeros(d, d)
    Sw = torch.zeros(d, d)
    for c in classes.tolist():
        xc = x[y == c]
        if xc.numel() == 0:
            continue
        muc = xc.mean(dim=0, keepdim=True)
        nc = xc.shape[0]
        dc = (muc - mu).T  # (D,1)
        Sb += float(nc) * (dc @ dc.T)

        xcw = xc - muc
        Sw += xcw.T @ xcw

    Sw += eps * torch.eye(d)
    # Solve generalized eigenproblem: Sb v = λ Sw v
    # Convert to standard eigenproblem via Sw^{-1} Sb.
    A = torch.linalg.solve(Sw, Sb)
    evals, evecs = torch.linalg.eig(A)
    evals = evals.real
    evecs = evecs.real
    top = torch.argsort(evals, descending=True)[:2]
    W = evecs[:, top]  # (D,2)
    # Orthonormalize for nicer scaling.
    W, _ = torch.linalg.qr(W)
    return W[:, :2]


def _project_2d(x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    x = x.detach().float().cpu()
    x = x - x.mean(dim=0, keepdim=True)
    return x @ W.detach().float().cpu()


def _save_ppm_scatter(
    path: Path,
    xy: torch.Tensor,
    y: torch.Tensor,
    *,
    width: int = 640,
    height: int = 360,
) -> None:
    """
    Save a tiny visualization without extra deps.
    Produces an ASCII PPM (P3) image with colored points.
    """
    xy = xy.detach().float().cpu()
    y = y.detach().cpu()

    mins = xy.min(dim=0).values
    maxs = xy.max(dim=0).values
    span = torch.clamp(maxs - mins, min=1e-6)
    norm = (xy - mins) / span
    pad = 0.06
    norm = norm * (1 - 2 * pad) + pad

    palette = [
        (74, 158, 255),  # blue
        (255, 122, 47),  # orange
        (61, 186, 122),  # green
        (155, 114, 207),  # purple
        (255, 77, 109),  # red/pink
        (255, 209, 102),  # yellow
        (142, 202, 230),  # light blue
        (255, 183, 3),  # amber
    ]
    bg = (15, 17, 23)
    img = [[bg for _ in range(width)] for _ in range(height)]

    r = 2
    for (u, v), cls in zip(norm.tolist(), y.tolist(), strict=True):
        cx = int(u * (width - 1))
        cy = int((1 - v) * (height - 1))
        col = palette[int(cls) % len(palette)]
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                xx = cx + dx
                yy = cy + dy
                if 0 <= xx < width and 0 <= yy < height:
                    img[yy][xx] = col

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as f:
        f.write(f"P3\n{width} {height}\n255\n")
        for row in img:
            f.write(" ".join(f"{rr} {gg} {bb}" for (rr, gg, bb) in row))
            f.write("\n")


def _save_ppm_similarity_heatmap(
    path: Path,
    emb: torch.Tensor,
    y: torch.Tensor,
    *,
    max_items: int = 240,
) -> None:
    """
    Save a cosine-similarity matrix heatmap as ASCII PPM (P3).

    This is much more stable than PCA scatter: if metric learning works, the
    matrix becomes block-diagonal when items are sorted by class label.
    """
    emb = emb.detach().float().cpu()
    y = y.detach().cpu()

    n = min(max_items, emb.shape[0])
    emb = emb[:n]
    y = y[:n]

    order = torch.argsort(y)
    emb = emb[order]
    y = y[order]

    sim = emb @ emb.T  # cosine sim since emb is normalized
    sim = torch.clamp(sim, -1.0, 1.0)

    # Map [-1,1] -> [0,255]
    img = (((sim + 1.0) * 0.5) * 255.0).to(torch.uint8)
    h, w = img.shape

    # Build RGB with a simple cold->hot palette (blue to orange).
    def col(v: int) -> tuple[int, int, int]:
        # v in [0,255]
        # dark blue -> gray -> orange
        if v < 128:
            t = v / 127.0
            r = int(15 + t * (120 - 15))
            g = int(17 + t * (120 - 17))
            b = int(45 + t * (180 - 45))
            return (r, g, b)
        t = (v - 128) / 127.0
        r = int(120 + t * (255 - 120))
        g = int(120 + t * (140 - 120))
        b = int(180 + t * (60 - 180))
        return (r, g, b)

    rgb = [[(0, 0, 0) for _ in range(w)] for _ in range(h)]
    for i in range(h):
        row = img[i].tolist()
        rgb[i] = [col(v) for v in row]

    # Draw class boundary lines (thin white) for readability.
    boundaries = []
    last = int(y[0].item())
    for i in range(1, n):
        cur = int(y[i].item())
        if cur != last:
            boundaries.append(i)
            last = cur
    for bnd in boundaries:
        if 0 <= bnd < n:
            for x in range(w):
                rgb[bnd][x] = (240, 244, 255)
            for ypix in range(h):
                rgb[ypix][bnd] = (240, 244, 255)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as f:
        f.write(f"P3\n{w} {h}\n255\n")
        for row in rgb:
            f.write(" ".join(f"{rr} {gg} {bb}" for (rr, gg, bb) in row))
            f.write("\n")


def _save_ppm_same_class_mask(
    path: Path,
    y: torch.Tensor,
    *,
    max_items: int = 240,
) -> None:
    """
    Save a ground-truth mask (same-class / different-class) as ASCII PPM (P3).
    Items are sorted by class label (same ordering as the similarity heatmap).

    - white pixels: same class
    - dark pixels: different class
    - thin white boundary lines: class boundaries (for readability)
    """
    y = y.detach().cpu()
    n = min(max_items, y.shape[0])
    y = y[:n]

    order = torch.argsort(y)
    y = y[order]

    same = (y.unsqueeze(0) == y.unsqueeze(1))  # (n,n) bool
    h = w = n

    bg = (15, 17, 23)
    fg = (240, 244, 255)
    mid = (80, 86, 98)

    rgb = [[bg for _ in range(w)] for _ in range(h)]
    for i in range(h):
        row = same[i].tolist()
        rgb[i] = [fg if v else mid for v in row]

    # Boundaries between class blocks.
    boundaries = []
    last = int(y[0].item())
    for i in range(1, n):
        cur = int(y[i].item())
        if cur != last:
            boundaries.append(i)
            last = cur
    for bnd in boundaries:
        if 0 <= bnd < n:
            for x in range(w):
                rgb[bnd][x] = fg
            for ypix in range(h):
                rgb[ypix][bnd] = fg

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as f:
        f.write(f"P3\n{w} {h}\n255\n")
        for row in rgb:
            f.write(" ".join(f"{rr} {gg} {bb}" for (rr, gg, bb) in row))
            f.write("\n")


def sample_pk_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    p: int,
    k: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    classes = torch.unique(y)
    chosen = classes[torch.randperm(classes.numel(), generator=generator)[:p]]

    idxs: list[torch.Tensor] = []
    for c in chosen.tolist():
        c_idx = torch.where(y == c)[0]
        # sample with replacement if needed
        pick = c_idx[torch.randint(0, c_idx.numel(), (k,), generator=generator)]
        idxs.append(pick)

    batch_idx = torch.cat(idxs, dim=0)
    return x[batch_idx], y[batch_idx]


def pairwise_cosine_distance(z: torch.Tensor) -> torch.Tensor:
    # z is normalized, so cosine similarity is dot product.
    sim = z @ z.T
    return 1.0 - sim  # (B,B)


def semi_hard_triplets_in_batch(z: torch.Tensor, y: torch.Tensor, margin: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Semi-hard mining:
    For each anchor, choose a positive (same class) and a negative such that:
      d(a, p) < d(a, n) < d(a, p) + margin
    If none exists, fall back to the closest negative (hard-ish) to keep gradients alive.
    """
    d = pairwise_cosine_distance(z).detach()  # mining doesn't backprop
    b = z.shape[0]

    a_idx: list[int] = []
    p_idx: list[int] = []
    n_idx: list[int] = []

    for a in range(b):
        same = torch.where(y == y[a])[0]
        diff = torch.where(y != y[a])[0]
        if same.numel() < 2 or diff.numel() < 1:
            continue

        # pick a positive that's not the anchor itself (random among same-class)
        same_wo_a = same[same != a]
        p = same_wo_a[torch.randint(0, same_wo_a.numel(), (1,)).item()].item()
        dap = d[a, p].item()

        dan = d[a, diff]  # distances to negatives
        # semi-hard condition
        mask = (dan > dap) & (dan < dap + margin)
        if mask.any():
            candidates = diff[mask]
            # choose the closest among semi-hard to focus learning
            n = candidates[dan[mask].argmin()].item()
        else:
            # fallback: closest negative
            n = diff[dan.argmin()].item()

        a_idx.append(a)
        p_idx.append(p)
        n_idx.append(n)

    if not a_idx:
        raise RuntimeError("No triplets could be mined. Try increasing batch size or K.")

    return (
        torch.tensor(a_idx, dtype=torch.long),
        torch.tensor(p_idx, dtype=torch.long),
        torch.tensor(n_idx, dtype=torch.long),
    )


@torch.no_grad()
def recall_at_k(emb_q: torch.Tensor, y_q: torch.Tensor, emb_db: torch.Tensor, y_db: torch.Tensor, k: int) -> float:
    # cosine similarity since normalized
    sim = emb_q @ emb_db.T  # (Nq, Ndb)
    topk = sim.topk(k=k, dim=1).indices  # (Nq,k)
    y_topk = y_db[topk]  # (Nq,k)
    hits = (y_topk == y_q.unsqueeze(1)).any(dim=1)
    return hits.float().mean().item()


@torch.no_grad()
def evaluate_retrieval(model: nn.Module, x_eval: torch.Tensor, y_eval: torch.Tensor) -> dict[str, float]:
    model.eval()
    emb = model(x_eval)
    # Use eval set as both queries and database, but exclude self-match.
    sim = emb @ emb.T
    sim.fill_diagonal_(-1e9)
    top1 = sim.topk(k=1, dim=1).indices
    top5 = sim.topk(k=5, dim=1).indices
    r1 = (y_eval[top1].squeeze(1) == y_eval).float().mean().item()
    r5 = (y_eval[top5] == y_eval.unsqueeze(1)).any(dim=1).float().mean().item()
    return {"recall@1": r1, "recall@5": r5}


def train_metric_learning(
    *,
    cfg: SynthConfig | None = None,
    seed: int,
    steps: int = 400,
    p: int = 10,
    k: int = 6,
    margin: float = 0.25,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    viz_dir: str | None = None,
    viz_every: int = 50,
    viz_max_points: int = 600,
    viz_method: str = "lda",
    viz_kind: str = "sim",
) -> tuple[dict[str, float], dict[str, float]]:
    set_seed(seed)
    cfg = cfg or SynthConfig()
    x_train, y_train = make_synth_dataset(cfg, seed=seed, split="train")
    x_eval, y_eval = make_synth_dataset(cfg, seed=seed, split="eval")

    x_train, y_train = x_train.to(device), y_train.to(device)
    x_eval, y_eval = x_eval.to(device), y_eval.to(device)

    model = SmallEmbedder(cfg.input_dim, cfg.embed_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    before = evaluate_retrieval(model, x_eval, y_eval)

    g = torch.Generator(device="cpu").manual_seed(seed + 123)
    triplet = nn.TripletMarginWithDistanceLoss(
        distance_function=lambda a, b: 1.0 - (a * b).sum(dim=-1),
        margin=margin,
        reduction="mean",
    )

    model.train()
    viz_path = Path(viz_dir) if viz_dir else None
    if viz_path is not None:
        # mkdir -p $viz_path
        viz_path.mkdir(parents=True, exist_ok=True)
    W: torch.Tensor | None = None
    for step in range(steps):
        xb, yb = sample_pk_batch(x_train, y_train, p=p, k=k, generator=g)
        xb, yb = xb.to(device), yb.to(device)

        z = model(xb)
        a_i, p_i, n_i = semi_hard_triplets_in_batch(z, yb, margin=margin)

        loss = triplet(z[a_i], z[p_i], z[n_i])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if viz_path is not None and (step % max(1, viz_every) == 0 or step == steps - 1):
            model.eval()
            n = min(viz_max_points, x_eval.shape[0])
            emb_eval = model(x_eval[:n])
            if viz_kind == "scatter":
                if W is None:
                    W = _projection_matrix(emb_eval, y_eval[:n], method=viz_method)
                xy = _project_2d(emb_eval, W)
                _save_ppm_scatter(viz_path / f"scatter_{step:05d}.ppm", xy, y_eval[:n])
            elif viz_kind == "sim":
                _save_ppm_similarity_heatmap(viz_path / f"sim_{step:05d}.ppm", emb_eval, y_eval[:n])
                _save_ppm_same_class_mask(viz_path / f"gt_{step:05d}.ppm", y_eval[:n])
            else:
                raise ValueError(f"Unknown viz_kind: {viz_kind}")
            model.train()

    after = evaluate_retrieval(model, x_eval, y_eval)
    return before, after


def main() -> None:
    cfg = SynthConfig()
    before, after = train_metric_learning(
        cfg=cfg,
        seed=123,
        steps=700,
        p=10,
        k=6,
        margin=0.25,
        lr=2e-3,
        weight_decay=1e-4,
        device="cpu",
        # Example: export frames during training (PPM images):
        viz_dir="topics/contrastive-and-metric-learning-for-fine-grained-visual-recognition/assets/visualizations/train_frames_ppm",
        viz_every=50,
        viz_method="lda",  # "lda" (default) or "pca"
        viz_kind="sim",  # "sim" (default) or "scatter"
    )
    print("Before:", before)
    print("After: ", after)


if __name__ == "__main__":
    main()
