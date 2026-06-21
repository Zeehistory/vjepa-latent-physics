"""Category-structure probing: are physics *categories* linearly separable in the latent space?

This implements **Step 1** of the physics-steering roadmap. Physics-IQ ships only coarse category
labels (solid mechanics, fluids, optics, ...) and *no* per-frame numeric physics state, so we cannot
regress quantities here (that is :mod:`train_probe`, Step 2). Instead we ask a structural question:

* Can a **linear** classifier separate the categories from clip-level pooled latents? If yes, each
  category occupies a (linearly) distinct region — and the classifier's per-class weight vector is a
  *candidate steering direction* we reuse downstream (Step 3).
* Does an **MLP** classifier do better? The linear-vs-MLP gap measures how much category information is
  present but *non-linearly* entangled.

Two controls keep the claim honest:

* **shuffled-label** — accuracy must collapse to the majority-class / chance rate.
* **appearance baseline** — the same probe on raw pooled *pixels* (and on shallow encoder layers). If
  shallow/pixel accuracy already matches deep-layer accuracy, the probe is reading appearance
  (lighting, colour, camera) rather than physics. Genuine physics structure should be *more* decodable
  in *deeper* layers — so we sweep all layers and look for accuracy that **rises with depth**.

Out-of-fold predictions (``cross_val_predict``) give an honest confusion matrix without a separate
held-out split, which matters because Physics-IQ category counts are small.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from ..analysis.latent_geometry import pooled_features
from ..encoders.feature_extractor import LatentDataset


def _pixel_features(latent_dir: str | Path) -> tuple[np.ndarray, list[str]]:
    """Clip-level mean-pooled *pixel* features (appearance baseline) + category labels.

    The latent cache stores the (downsampled) target frames alongside the latents, so we can build a
    pure-appearance control without re-reading the source videos.
    """
    ds = LatentDataset(latent_dir, layers="all")
    feats, cats = [], []
    for i in range(len(ds)):
        s = ds[i]
        frames = s["frames"].numpy()  # (T, C, H, W) in [0, 1]
        feats.append(frames.mean(axis=(0, 2, 3)))  # mean colour per channel -> crude appearance vector
        cats.append(s["category"])
    return np.stack(feats, 0), cats


def _n_splits(labels: np.ndarray, requested: int = 5) -> int:
    """Largest valid stratified-CV fold count (bounded by the smallest class)."""
    _, counts = np.unique(labels, return_counts=True)
    return int(max(2, min(requested, counts.min())))


def _fit_eval_clf(
    X: np.ndarray,
    y: np.ndarray,
    kind: str,
    seed: int,
    shuffle_labels: bool,
) -> dict[str, Any]:
    """Cross-validated classification accuracy + macro-F1, with out-of-fold confusion matrix.

    Returns ``accuracy``, ``macro_f1``, ``classes``, ``confusion`` (out-of-fold), and ``coef`` — the
    per-class linear weight directions ``(n_classes, D)`` for ``kind == "linear"`` (else ``None``).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = y.copy()
    rng = np.random.default_rng(seed)
    if shuffle_labels:
        y = y[rng.permutation(len(y))]

    classes = sorted(set(y.tolist()))
    if len(classes) < 2:
        return {"accuracy": float("nan"), "macro_f1": float("nan"),
                "classes": classes, "confusion": None, "coef": None}

    if kind == "linear":
        est = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    else:
        est = MLPClassifier(hidden_layer_sizes=(128, 128), max_iter=600, random_state=seed,
                            early_stopping=True)
    model = make_pipeline(StandardScaler(), est)

    splits = _n_splits(y, 5)
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    pred = cross_val_predict(model, X, y, cv=cv)
    acc = float(accuracy_score(y, pred))
    f1 = float(f1_score(y, pred, average="macro"))
    cm = confusion_matrix(y, pred, labels=classes)

    coef = None
    if kind == "linear" and not shuffle_labels:
        # Refit on all data to expose the per-class weight directions (the steering-direction bridge).
        full = make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=2000, C=1.0, class_weight="balanced")).fit(X, y)
        lr = full.named_steps["logisticregression"]
        coef = lr.coef_.astype(np.float32)  # (n_classes, D); binary -> (1, D)

    return {"accuracy": acc, "macro_f1": f1, "classes": classes, "confusion": cm, "coef": coef}


def classify_categories(
    latent_dir: str | Path,
    layers: list[int] | str = "all",
    seed: int = 0,
    output_csv: str | Path | None = None,
    pixel_baseline: bool = True,
) -> dict[str, Any]:
    """Run linear + MLP category classifiers per layer (with controls + appearance baseline).

    Returns a dict with ``records`` (one row per layer × probe, with controls), ``confusions`` (per
    layer, linear out-of-fold confusion matrix + class order), and ``directions`` (per layer, linear
    per-class weight vectors ``(n_classes, D)`` — reused as candidate steering directions in Step 3).
    """
    dataset = LatentDataset(latent_dir, layers="all")
    available = dataset.available_layers()
    layer_list = available if layers == "all" else [int(x) for x in layers]

    records: list[dict[str, Any]] = []
    confusions: dict[int, dict[str, Any]] = {}
    directions: dict[int, dict[str, Any]] = {}

    for layer in layer_list:
        X, cats = pooled_features(latent_dir, layer)
        y = np.asarray(cats)
        for kind in ("linear", "mlp"):
            real = _fit_eval_clf(X, y, kind, seed, shuffle_labels=False)
            ctrl = _fit_eval_clf(X, y, kind, seed, shuffle_labels=True)
            records.append({
                "layer": layer, "probe": kind,
                "accuracy": round(real["accuracy"], 4),
                "macro_f1": round(real["macro_f1"], 4),
                "ctrl_shuffled_label_accuracy": round(ctrl["accuracy"], 4),
                "n_classes": len(real["classes"]), "n_samples": int(len(y)),
            })
            if kind == "linear":
                confusions[layer] = {"matrix": real["confusion"], "classes": real["classes"]}
                if real["coef"] is not None:
                    directions[layer] = {"classes": real["classes"], "coef": real["coef"]}

    # Appearance baseline: pixel-only probe + shallowest-layer probe (already in the per-layer sweep).
    if pixel_baseline:
        Xp, cats = _pixel_features(latent_dir)
        yp = np.asarray(cats)
        for kind in ("linear", "mlp"):
            real = _fit_eval_clf(Xp, yp, kind, seed, shuffle_labels=False)
            ctrl = _fit_eval_clf(Xp, yp, kind, seed, shuffle_labels=True)
            records.append({
                "layer": -1, "probe": f"pixel_{kind}",
                "accuracy": round(real["accuracy"], 4),
                "macro_f1": round(real["macro_f1"], 4),
                "ctrl_shuffled_label_accuracy": round(ctrl["accuracy"], 4),
                "n_classes": len(real["classes"]), "n_samples": int(len(yp)),
            })

    if output_csv is not None and records:
        path = Path(output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    return {"records": records, "confusions": confusions, "directions": directions}
