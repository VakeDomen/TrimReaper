#!/usr/bin/env python
"""End-to-end smoke demo of the TrimReaper proof-of-concept on a tiny model.

Runs the complete pipeline without needing a GPU or a large Qwen3-4B
download: builds a tiny randomly-initialized gated-MLP Qwen3 model, runs the
ratcheting GA both WITH and WITHOUT rotations, computes the one-shot baseline
curves, and writes a Pareto frontier plot. Verifies archive persistence and
physical compaction.

Usage:
    .venv/bin/python scripts/run_smoke.py [--out DIR]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# make `import trimreaper` work when running from the repo root
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

import torch  # noqa: E402

from trimreaper.config import Config  # noqa: E402
from trimreaper.model import load_tiny_smoke_model  # noqa: E402
from trimreaper.pipeline import ratchet_search  # noqa: E402
from trimreaper.sweep import baseline_curves, plot_frontiers  # noqa: E402


class FakeStreamer:
    """Synthetic token streamer matching the WikiTextStreamer interface."""

    def __init__(self, vocab=3200, seed=0):
        self.vocab = vocab
        self.g = torch.Generator().manual_seed(seed)

    def batch(self, n, seq_len):
        return torch.randint(0, self.vocab, (n, seq_len), generator=self.g)

    def holdout_batches(self):
        # small fixed holdout for the demo
        return [self.batch(2, 32) for _ in range(2)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="runs/smoke", help="output directory")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg = Config.defaults()
    cfg.model.layers = [0]
    cfg.ga.population = 6
    cfg.ga.elitism = 1
    cfg.search.start_target = 16
    cfg.search.ratchet_step = 16
    cfg.search.max_target = 80   # cap so the smoke demo finishes fast on a busy CPU
    cfg.search.rounds = 2
    cfg.search.epsilon = 0.5
    cfg.search.archive_dir = os.path.join(args.out, "archive")
    cfg.genome.min_rotations = 2
    cfg.genome.max_rotations = 8
    cfg.data.seq_len = 32
    cfg.data.fit_batch = 2

    print(f"[smoke] building tiny gated-MLP model (no GPU/download)...")
    pm = load_tiny_smoke_model(cfg)
    streamer = FakeStreamer()
    holdout = streamer.holdout_batches()
    width = pm.pristine[0]["up_proj"].shape[0]
    print(f"[smoke] MLP width = {width}")

    t = time.time()
    res_rot = ratchet_search(pm, cfg, 0, streamer, holdout, use_rotations=True)
    res_norot = ratchet_search(pm, cfg, 0, streamer, holdout, use_rotations=False)
    print(f"[smoke] both GA runs took {time.time()-t:.2f}s")

    print(f"[smoke] GA+rotations archive:")
    for p in res_rot.points:
        print(f"   removed={p.removed:4d}  kl={p.kl:.5f}  holdout_kl={p.validated_kl:.5f}  "
              f"n_rot={len(p.genome.rotations)}")
    print(f"[smoke] GA-no-rotations archive:")
    for p in res_norot.points:
        print(f"   removed={p.removed:4d}  kl={p.kl:.5f}  holdout_kl={p.validated_kl:.5f}")

    widths = [16, 32]
    curves = baseline_curves(pm, cfg, 0, streamer, widths=widths)
    for name, pts in curves.items():
        print(f"[smoke] baseline {name}: " + ", ".join(f"{w}->{kl:.4f}" for w, kl in pts))

    ga_points = [(p.removed, p.validated_kl, "GA+rotations") for p in res_rot.points]
    ga_points += [(p.removed, p.validated_kl, "GA-no-rotations") for p in res_norot.points]
    plot_path = os.path.join(args.out, "smoke_frontier.png")
    plot_frontiers(ga_points, curves, plot_path,
                   title="Tiny-model MLP pruning frontier (smoke demo)")
    print(f"[smoke] frontier plot -> {plot_path}")

    # physical compaction on the best genome
    from trimreaper.compaction import compact_model_metrics
    if res_rot.best_genome is not None:
        metrics = compact_model_metrics(pm, 0, sorted(res_rot.best_genome.pruned),
                                        res_rot.best_genome.rotations)
        print(f"[smoke] compaction: channels {metrics['original_channels']} -> "
              f"{metrics['compact_channels']}  "
              f"params removed {metrics['params_removed']} "
              f"({metrics['fraction_removed']:.1%})")

    print(f"[smoke] archive files: {sorted(os.listdir(cfg.search.archive_dir))}")
    print("[smoke] DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
