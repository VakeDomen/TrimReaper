"""Epsilon sweep and Pareto plotting (PLAN.md sections 10, 15).

The proof-of-concept result is the Pareto frontier:

    number of channels removed   vs   divergence from original model

for each method (random, weight-norm, activation-magnitude, GA-no-rotations,
GA-with-rotations). This module builds a one-shot KL-vs-removed curve for the
non-evolutionary baselines and aggregates GA results into the same frontier.
"""

from __future__ import annotations

import os

import torch

from .baselines import (
    activation_magnitude_ranking,
    random_ranking,
    weight_norm_ranking,
)
from .config import Config
from .evaluate import (
    baseline_logits,
    evaluate_candidate,
    evaluate_pruned_on_set,
)
from .model import PrunedModel


def baseline_curves(
    pm: PrunedModel,
    cfg: Config,
    layer: int,
    streamer,
    widths: list[int] | None = None,
    seed: int = 0,
    val_batches: list | None = None,
    test_batches: list | None = None,
) -> dict[str, dict]:
    """Return per-method baseline curves evaluated on the SAME sets as the GA.

    Fix: previously the one-shot baselines were scored on a single fresh
    fitness batch, so they were NOT comparable to the GA's validation-KL. Now
    every method is measured on the same ``val_batches`` (and optionally the
    untouched ``test_batches``), giving apples-to-apples numbers.

    Returns: {method: {"val": [(w, kl), ...], "test": [(w, kl), ...]}}.
    """
    import random

    rng = random.Random(seed)
    seq_len = cfg.data.seq_len
    if widths is None:
        widths = [cfg.search.start_target, cfg.search.start_target + cfg.search.ratchet_step]

    # probe batches for activation magnitude (on the pristine model; not part
    # of the scored validation/test sets).
    probe_batches = [streamer.batch(4, seq_len) for _ in range(4)]

    wnorm = weight_norm_ranking(pm, layer)
    amag = activation_magnitude_ranking(pm, layer, probe_batches, seq_len)
    rand = random_ranking(pm.pristine[layer]["up_proj"].shape[0], rng)

    maps = {"random": rand, "weight_norm": wnorm, "activation_magnitude": amag}
    set_map = {"val": val_batches, "test": test_batches}
    results: dict[str, dict[str, list[tuple[int, float]]]] = {}
    for method, order in maps.items():
        results[method] = {"val": [], "test": []}
        for w in widths:
            pruned = order[:w]
            if val_batches:
                results[method]["val"].append(
                    (w, evaluate_pruned_on_set(pm, val_batches, layer, pruned, seq_len))
                )
            if test_batches:
                results[method]["test"].append(
                    (w, evaluate_pruned_on_set(pm, test_batches, layer, pruned, seq_len))
                )
    return results


def plot_frontiers(
    ga_points: list[tuple[int, float, str]],  # (removed, kl, label)
    baseline_points: dict[str, dict[str, list[tuple[int, float]]]],  # method -> {"val"/"test" -> pts}
    out_path: str,
    title: str = "MLP width-pruning Pareto frontier",
) -> str:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""  # plotting unavailable; caller can decide

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    markers = {"random": "x", "weight_norm": "s", "activation_magnitude": "D"}
    for name, sets in baseline_points.items():
        # plot the validation curve (same set as GA) by default
        pts = sets.get("val") or sets.get("test") or []
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker=markers.get(name, "o"), label=name, linestyle="--")

    # group GA labels
    ga_by_label: dict[str, list[tuple[int, float]]] = {}
    for removed, kl, label in ga_points:
        ga_by_label.setdefault(label, []).append((removed, kl))
    for label, pts in ga_by_label.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", label=label, linestyle="-", linewidth=2)

    ax.set_xlabel("channels removed")
    ax.set_ylabel("KL divergence from original")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
