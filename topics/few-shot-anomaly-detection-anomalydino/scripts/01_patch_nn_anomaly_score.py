"""Demonstrate AnomalyDINO-style patch NN scoring on synthetic features.

Shows:
- cosine distance between patch embeddings;
- memory bank from "normal" patches;
- per-patch NN distance map;
- image-level score = mean of top 1% patch distances (as in AnomalyDINO).

Expected behavior: a test grid with an injected "defect" region gets a higher
image score than a grid matching the reference distribution.
"""

from __future__ import annotations

import numpy as np


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance 1 - cos_sim between two vectors."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return 1.0 - float(np.dot(a, b) / denom)


def build_memory_bank(normal_patches: np.ndarray) -> np.ndarray:
    """Stack reference patch features (N_ref, D)."""
    return np.asarray(normal_patches, dtype=np.float64)


def patch_nn_distances(
    test_patches: np.ndarray, memory_bank: np.ndarray
) -> np.ndarray:
    """For each test patch, min cosine distance to any patch in M."""
    distances = np.empty(len(test_patches), dtype=np.float64)
    for i, p in enumerate(test_patches):
        d_min = min(cosine_distance(p, m) for m in memory_bank)
        distances[i] = d_min
    return distances


def image_anomaly_score(patch_distances: np.ndarray, top_fraction: float = 0.01) -> float:
    """Mean of top `top_fraction` patch distances (AnomalyDINO aggregation q)."""
    n = len(patch_distances)
    k = max(1, int(np.ceil(n * top_fraction)))
    topk = np.partition(patch_distances, -k)[-k:]
    return float(np.mean(topk))


def _make_patch_grid(
    rng: np.random.Generator, grid_h: int, grid_w: int, dim: int, mean: np.ndarray
) -> np.ndarray:
    """grid_h*grid_w patches around `mean` with small noise."""
    patches = []
    for _ in range(grid_h * grid_w):
        p = mean + 0.05 * rng.standard_normal(dim)
        patches.append(p)
    return np.stack(patches, axis=0)


def demo_scores(seed: int = 42) -> tuple[float, float]:
    """Return (score_normal_test, score_defect_test)."""
    rng = np.random.default_rng(seed)
    dim = 32
    grid_h, grid_w = 10, 10
    n_patches = grid_h * grid_w

    normal_mean = rng.standard_normal(dim)
    ref = _make_patch_grid(rng, grid_h, grid_w, dim, normal_mean)
    memory = build_memory_bank(ref)

    # Test: same distribution as reference
    test_ok = _make_patch_grid(rng, grid_h, grid_w, dim, normal_mean)
    dist_ok = patch_nn_distances(test_ok, memory)
    score_ok = image_anomaly_score(dist_ok)

    # Test: defect block (20 patches) shifted far in feature space
    test_bad = test_ok.copy()
    defect_mean = normal_mean + 2.5 * rng.standard_normal(dim)
    defect_idx = np.arange(40, 60)
    for j in defect_idx:
        test_bad[j] = defect_mean + 0.05 * rng.standard_normal(dim)
    dist_bad = patch_nn_distances(test_bad, memory)
    score_bad = image_anomaly_score(dist_bad)

    return score_ok, score_bad


def main() -> None:
    score_ok, score_bad = demo_scores()
    print(f"Normal-like test score: {score_ok:.4f}")
    print(f"Defect-injected score:  {score_bad:.4f}")
    print(f"Margin (bad - ok):      {score_bad - score_ok:.4f}")


if __name__ == "__main__":
    main()
