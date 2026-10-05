#!/usr/bin/env python
"""Small sanity experiment on the CORRECTED pipeline.

Runs the REAL ratcheting GA (same code path as `trimreaper run`) on the tiny
gated-MLP model at a SINGLE fixed removal target, comparing GA-with-rotations
vs GA-without across several seeds, reporting BOTH validation KL (the archive
objective) and test KL (untouched set).

This is the reviewer's asked-for sanity check run on the corrected semantics:
  - rotation on the post-SwiGLU hidden activation + masked down' (invariant
    when unmasked),
  - real tournament selection,
  - archive decided strictly by validation KL,
  - the exact same validation/test sets for every method/seed,
  - FP32 KL.

Usage:
    .venv/bin/python scripts/sanity_compare.py [--seeds 0,1,2] [--target 56]
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

import torch  # noqa: E402

from trimreaper.config import Config, validate  # noqa: E402
from trimreaper.evaluate import evaluate_genome_on_set  # noqa: E402
from trimreaper.model import load_tiny_smoke_model  # noqa: E402
from trimreaper.pipeline import ratchet_search  # noqa: E402


class FakeStreamer:
    """Synthetic token streamer matching WikiTextStreamer.batch() (fitness only)."""

    def __init__(self, vocab=3200):
        self.vocab = vocab
        self.seed = 0

    def batch(self, n, seq_len):
        self.seed += 1
        g = torch.Generator().manual_seed(self.seed)
        return torch.randint(0, self.vocab, (n, seq_len), generator=g)


def run_variant(cfg, layer, val, test, target, use_rotations, seed):
    cfg.ga.seed = seed
    cfg.search.start_target = target
    cfg.search.max_target = target      # single fixed target, no ratchet
    cfg.search.rounds = 4
    cfg.ga.population = 8
    cfg.ga.tournament_size = 4
    cfg.search.valid_best_k = 3
    pm = load_tiny_smoke_model(cfg)
    res = ratchet_search(
        pm, cfg, layer, FakeStreamer(), val, use_rotations=use_rotations,
        archive_dir=None, variant="rot" if use_rotations else "norot",
    )
    if res.best_genome is not None:
        tkl = evaluate_genome_on_set(pm, test, layer, res.best_genome, cfg.data.seq_len)
    else:
        tkl = float("nan")
    vkl = res.points[0].validated_kl if res.points else float("nan")
    return vkl, tkl


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="0,1,2", help="comma-separated seeds")
    ap.add_argument("--target", type=int, default=56, help="channels to remove")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    cfg = Config.defaults()
    cfg.model.layers = [0]
    cfg.data.seq_len = 32
    cfg.data.fit_batch = 2
    cfg.genome.min_rotations = 2
    cfg.genome.max_rotations = 8
    validate(cfg)

    # canonical validation/test sets, FIXED across methods and seeds
    gv = torch.Generator().manual_seed(2024)
    val = [torch.randint(0, 3200, (2, 32), generator=gv) for _ in range(4)]
    gt = torch.Generator().manual_seed(777)
    test = [torch.randint(0, 3200, (2, 32), generator=gt) for _ in range(4)]

    print(f"sanity: fixed target removal = {args.target} of 128 channels, seeds={seeds}")
    print("corrected semantics: post-SwiGLU hidden rotation, tournament selection,")
    print("archive-by-validation-KL, FP32 KL, same val/test sets for all methods")
    print(f"{'seed':>5} {'method':<6} {'valid_kl':>10} {'test_kl':>10}")
    agg = {"rot": [], "norot": []}
    for seed in seeds:
        for use_rot, name in ((True, "rot"), (False, "norot")):
            vkl, tkl = run_variant(cfg, 0, val, test, args.target, use_rot, seed)
            agg[name].append(vkl)
            print(f"{seed:>5} {name:<6} {vkl:10.5f} {tkl:10.5f}")

    def _mean(k):
        xs = agg[k]
        return sum(xs) / len(xs) if xs else float("nan")

    print("-" * 38)
    print("mean valid_kl  GA+rot = %.5f   GA-norot = %.5f" % (_mean("rot"), _mean("norot")))
    print("(lower is better; tiny smoke-width sanity signal only, not a conclusion)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
