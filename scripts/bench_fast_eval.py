#!/usr/bin/env python
"""Benchmark the fast-eval candidate batching (reviewer optimization #3).

Measures, for candidate chunks {1, 2, 4, 8, whole}, the throughput of the
fast batched-tail evaluation (candidates/sec) and peak reserved VRAM, plus a
slow-path baseline for reference.

Run ON THE GPU target machine:

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    .venv/bin/python scripts/bench_fast_eval.py --config configs/cuda_full.yaml

By default it runs the tiny smoke model on CPU so you can sanity-check the
script anywhere; pass --config to bench the real Qwen3-4B on GPU.  Adjust the
candidate population with --pop.

Stop increasing the chunk size when throughput stops improving or VRAM gets
uncomfortable — that is the practical chunk for the real run.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

import torch  # noqa: E402

from trimreaper.config import Config, validate  # noqa: E402
from trimreaper.ga import Genome  # noqa: E402
from trimreaper.rotation import PairRotation  # noqa: E402


def _rand_genomes(n: int, width: int, rng) -> list[Genome]:
    """Deterministic random genomes (rotations + a pruned slice)."""
    gens = []
    for _ in range(n):
        g = Genome()
        g.pruned = set(sorted(rng.sample(range(width), max(4, width // 16))))
        rots = []
        for _ in range(rng.randint(2, 6)):
            a = rng.randrange(width)
            b = (a + 1 + rng.randrange(width - 1)) % width
            rots.append(PairRotation(a, b, rng.uniform(-1.5, 1.5)))
        g.rotations = rots
        gens.append(g)
    return gens


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="config for the real model (default: tiny smoke)")
    ap.add_argument("--pop", type=int, default=32, help="number of candidate genomes")
    ap.add_argument("--reps", type=int, default=3, help="repetitions per chunk size")
    ap.add_argument("--seq", type=int, default=256, help="sequence length (tiny default 32)")
    args = ap.parse_args()

    if args.config:
        cfg = Config.from_yaml(args.config)
        validate(cfg)
        from trimreaper.model import load_pruned_model

        pm = load_pruned_model(cfg)
        layer = cfg.model.layers[0]
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        cfg = Config.defaults()
        cfg.model.layers = [0]
        cfg.data.seq_len = 32
        validate(cfg)
        from trimreaper.model import load_tiny_smoke_model

        pm = load_tiny_smoke_model(cfg)
        layer = 0
        args.seq = 32
        dev = torch.device("cpu")

    from trimreaper.fast_eval import build_cache, eval_genomes_batched
    from trimreaper.pipeline import _intermediate_size

    width = _intermediate_size(pm, layer)
    b = cfg.data.fit_batch
    rng = __import__("random").Random(0)
    genomes = _rand_genomes(args.pop, width, rng)
    print(f"model: {'REAL ' + cfg.model.model_id if args.config else 'tiny smoke'}  "
          f"device={dev}  width={width}  batch={b}  seq={args.seq}  pop={args.pop}")

    batch = torch.randint(
        0, (152 * 1024) if args.config else 3200, (b, args.seq), device=dev
    )
    cache = build_cache(pm, batch, layer, args.seq)

    def _run(chunk, n_reps):
        t0 = time.perf_counter()
        for _ in range(n_reps):
            eval_genomes_batched(cache, pm, layer, genomes, args.seq, chunk=chunk)
        dt = time.perf_counter() - t0
        cands = args.pop * n_reps
        return cands / dt

    print("\nchunk   cands/sec   (vs whole)")
    whole = _run(None, args.reps)
    print(f"whole   {whole:9.0f}   (x1.00)")
    for chunk in (1, 2, 4, 8):
        rate = _run(chunk, args.reps)
        print(f"{chunk:<6} {rate:9.0f}   (x{rate / whole:.2f})")

    # slow full-model baseline (very expensive on the real model; tiny only)
    if not args.config:
        from trimreaper.evaluate import evaluate_genome_on_set

        t0 = time.perf_counter()
        for g in genomes:
            evaluate_genome_on_set(pm, [batch], layer, g, args.seq)
        dt = time.perf_counter() - t0
        print(f"\nslow   {args.pop / dt:9.0f}   (x{ (args.pop / dt) / (whole or 1):.2f})")

    if dev.type == "cuda":
        print(f"\npeak GPU VRAM reserved: {torch.cuda.max_memory_allocated(dev)/1e9:.2f} GB")
    print("\nTip: pick the largest chunk whose cands/sec is still growing "
          "(or before VRAM gets uncomfortable) and set it in configs as "
          "search.fast_eval_chunk.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
