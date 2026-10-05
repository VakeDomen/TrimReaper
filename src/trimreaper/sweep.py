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
from .evaluate import baseline_logits, evaluate_candidate
from .model import PrunedModel


def baseline_curves(
    pm: PrunedModel,
    cfg: Config,
    layer: int,
    streamer,
    widths: list[int] | None = None,
    seed: int = 0,
) -> dict[str, list[tuple[int, float]]]:
    """Return {method: [(channels_removed, kl), ...]} for non-GA baselines."""
    import random

    rng = random.Random(seed)
    seq_len = cfg.data.seq_len
    fit_batch = cfg.data.fit_batch
    if widths is None:
        widths = [cfg.search.start_target, cfg.search.start_target + cfg.search.ratchet_step]

    batch = streamer.batch(fit_batch, seq_len)
    ref = baseline_logits(pm, [batch], seq_len)[0]
    probe_batches = [streamer.batch(4, seq_len) for _ in range(4)]

    wnorm = weight_norm_ranking(pm, layer)
    amag = activation_magnitude_ranking(pm, layer, probe_batches, seq_len)
    rand = random_ranking(pm.pristine[layer]["up_proj"].shape[0], rng)

    results: dict[str, list[tuple[int, float]]] = {
        "random": [], "weight_norm": [], "activation_magnitude": [],
    }
    maps = {"random": rand, "weight_norm": wnorm, "activation_magnitude": amag}
    for method, order in maps.items():
        for w in widths:
            pruned = order[:w]
            kl = evaluate_candidate(pm, batch, ref, layer, [], pruned, seq_len)
            results[method].append((w, kl))
    return results


def plot_frontiers(
    ga_points: list[tuple[int, float, str]],  # (removed, kl, label)
    baseline_points: dict[str, list[tuple[int, float]]],
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
    for name, pts in baseline_points.items():
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
