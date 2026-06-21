"""Subspace structure of physics categories (Step 1).

Classification accuracy alone does not prove categories live in *distinct directions* — a flexible
boundary can carve up entangled clouds. This module measures the geometry directly:

* **category directions** — the (standardised) mean latent of each category; their pairwise cosine
  similarity says which categories share a direction (overlap) vs. point elsewhere (distinct subspace).
* **separability** — Fisher-style between/within scatter ratio and silhouette: how cleanly the
  categories cluster, independent of any trained classifier.
* **principal angles** — between per-category PCA subspaces, to ask whether two categories (e.g. rigid
  motion vs. fluids) occupy genuinely different low-dimensional subspaces or merely shifted means.

All operate on clip-level pooled features ``(N, D)`` plus string labels (see
:func:`analysis.latent_geometry.pooled_features`).
"""

from __future__ import annotations

import numpy as np


def category_directions(features: np.ndarray, labels: list[str]) -> tuple[list[str], np.ndarray]:
    """Return ``(sorted_labels, means)`` where ``means[i]`` is the standardised mean of class ``i``.

    Features are z-scored over the dataset first so the cosine geometry is not dominated by a few
    high-variance dimensions.
    """
    f = (features - features.mean(0, keepdims=True)) / (features.std(0, keepdims=True) + 1e-8)
    classes = sorted(set(labels))
    lab = np.asarray(labels)
    means = np.stack([f[lab == c].mean(0) for c in classes], 0)
    return classes, means


def cosine_matrix(means: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity ``(C, C)`` between class-mean direction vectors (range ``[-1, 1]``)."""
    norm = means / (np.linalg.norm(means, axis=1, keepdims=True) + 1e-12)
    return norm @ norm.T


def separability(features: np.ndarray, labels: list[str]) -> dict[str, float]:
    """Classifier-free separability of the categories.

    * ``fisher_ratio`` = trace(between-class scatter) / trace(within-class scatter); >1 means classes
      are spread farther apart than they are internally diffuse.
    * ``silhouette`` = mean silhouette score (cosine) over clips; higher = tighter, better-separated.
    """
    f = (features - features.mean(0, keepdims=True)) / (features.std(0, keepdims=True) + 1e-8)
    lab = np.asarray(labels)
    classes = sorted(set(labels))
    mu = f.mean(0)
    sb = sw = 0.0
    for c in classes:
        fc = f[lab == c]
        mc = fc.mean(0)
        sb += len(fc) * float(((mc - mu) ** 2).sum())
        sw += float(((fc - mc) ** 2).sum())
    out = {"fisher_ratio": sb / (sw + 1e-12)}
    try:
        from sklearn.metrics import silhouette_score

        if len(classes) > 1 and len(f) > len(classes):
            out["silhouette"] = float(silhouette_score(f, lab, metric="cosine"))
    except Exception:
        pass
    return out


def principal_angles(features: np.ndarray, labels: list[str], k: int = 5) -> tuple[list[str], np.ndarray]:
    """Mean principal angle (radians) between each pair of categories' top-``k`` PCA subspaces.

    Small angles -> the two categories share a subspace (overlap); angles near ``pi/2`` -> orthogonal,
    distinct subspaces. Returns ``(sorted_labels, angle_matrix (C, C))`` with zeros on the diagonal.
    """
    f = (features - features.mean(0, keepdims=True)) / (features.std(0, keepdims=True) + 1e-8)
    lab = np.asarray(labels)
    classes = sorted(set(labels))

    bases: dict[str, np.ndarray] = {}
    for c in classes:
        fc = f[lab == c]
        kc = int(min(k, max(1, len(fc) - 1), fc.shape[1]))
        fc = fc - fc.mean(0, keepdims=True)
        _, _, vt = np.linalg.svd(fc, full_matrices=False)
        bases[c] = vt[:kc]  # (kc, D) orthonormal rows

    n = len(classes)
    ang = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            s = np.linalg.svd(bases[classes[i]] @ bases[classes[j]].T, compute_uv=False)
            mean_angle = float(np.arccos(np.clip(s, -1.0, 1.0)).mean())
            ang[i, j] = ang[j, i] = mean_angle
    return classes, ang
