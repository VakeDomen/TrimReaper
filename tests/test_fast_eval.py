"""Tests for the fast evaluation path (cache the pre-MLP prefix + batched tail).

The fast path must be numerically equivalent to the existing full-model path
(which mutates/restores weights per candidate). It should also satisfy the
unmasked-invariance invariant of the corrected rotation semantics.
"""

import torch

import pytest

from trimreaper.config import Config, validate
from trimreaper.evaluate import evaluate_genome_on_set
from trimreaper.fast_eval import build_cache, eval_genome_fast, eval_genomes_batched
from trimreaper.ga import Genome
from trimreaper.model import load_tiny_smoke_model
from trimreaper.rotation import PairRotation


def _cfg():
    cfg = Config.defaults()
    cfg.model.layers = [0]
    cfg.data.seq_len = 16
    validate(cfg)
    return cfg


def _genome(rots, pruned):
    g = Genome()
    g.rotations = [PairRotation(*t) for t in rots]
    g.pruned = set(pruned)
    return g


def _batch(seed=0, n=2, seq=16):
    return torch.randint(0, 3200, (n, seq), generator=torch.Generator().manual_seed(seed))


def test_fast_eval_matches_slow_single():
    cfg = _cfg()
    pm = load_tiny_smoke_model(cfg)
    batch = _batch()
    g = _genome([(10, 20, 0.7), (3, 40, -1.1)], [0, 5, 12, 30])
    fast = eval_genome_fast(build_cache(pm, batch, 0, 16), pm, 0, g, 16)
    slow = evaluate_genome_on_set(pm, [batch], 0, g, 16)
    assert abs(fast - slow) < 1e-6


def test_fast_eval_matches_slow_batched():
    cfg = _cfg()
    pm = load_tiny_smoke_model(cfg)
    batch = _batch()
    genomes = [
        _genome([(10, 20, 0.7), (3, 40, -1.1)], [0, 5, 12, 30]),
        _genome([(50, 5, 0.2), (80, 30, 2.0)], [1, 9, 22]),
        _genome([], [7, 8]),  # no rotation, like GA-no-rotations
    ]
    cache = build_cache(pm, batch, 0, 16)
    fasts = eval_genomes_batched(cache, pm, 0, genomes, 16)
    for f, g in zip(fasts, genomes):
        slow = evaluate_genome_on_set(pm, [batch], 0, g, 16)
        assert abs(f - slow) < 1e-6, (f, slow)


def test_fast_eval_no_rotation_is_exact():
    cfg = _cfg()
    pm = load_tiny_smoke_model(cfg)
    batch = _batch()
    g = _genome([], [0, 3, 9])
    fast = eval_genome_fast(build_cache(pm, batch, 0, 16), pm, 0, g, 16)
    slow = evaluate_genome_on_set(pm, [batch], 0, g, 16)
    assert abs(fast - slow) < 1e-12  # identical ops, no rotation


def test_fast_eval_unmasked_rotation_is_invariant():
    """With the mask emptied (no channels pruned), rotation + inverse
    compensation must leave logits ~unchanged (the corrected-semantics
    invariant), through the fast no-weight-rotation path."""
    cfg = _cfg()
    pm = load_tiny_smoke_model(cfg)
    batch = _batch()
    g = _genome([(10, 20, 0.7), (3, 40, -1.1), (50, 5, 0.2)], [])  # empty pruned
    kl = eval_genome_fast(build_cache(pm, batch, 0, 16), pm, 0, g, 16)
    assert kl < 1e-3, f"unmasked rotation should be invariant, KL={kl}"


def test_fast_eval_does_not_mutate_weights():
    """The fast path must leave the model's down_proj weights pristine (it never
    rotates them), unlike the slow path which counter-rotates + restores."""
    cfg = _cfg()
    pm = load_tiny_smoke_model(cfg)
    batch = _batch()
    down = pm.mlp_modules[0].down_proj
    before = down.weight.data.clone()
    g = _genome([(10, 20, 0.7), (3, 40, -1.1)], [0, 5])
    eval_genome_fast(build_cache(pm, batch, 0, 16), pm, 0, g, 16)
    assert torch.equal(down.weight.data, before)


def test_batched_chunk_size_invariant():
    """The chunk size must not change results: chunked vs whole-batch vs slow
    full-model must all agree. This is the tunable knob for GPU VRAM/throughput
    (reviewer's optimization #3)."""
    cfg = _cfg()
    pm = load_tiny_smoke_model(cfg)
    batch = _batch()
    genomes = [
        _genome([(10, 20, 0.7), (3, 40, -1.1)], [0, 5, 12, 30]),
        _genome([(50, 5, 0.2), (80, 30, 2.0)], [1, 9, 22]),
        _genome([], [7, 8]),
        _genome([(20, 7, 1.3)], [3, 11, 17]),
        _genome([(0, 99, 2.2), (15, 60, -0.8)], [0, 2, 4]),
    ]
    cache = build_cache(pm, batch, 0, 16)

    whole = eval_genomes_batched(cache, pm, 0, genomes, 16, chunk=None)
    for chunk in (1, 2, 3):
        chunked = eval_genomes_batched(cache, pm, 0, genomes, 16, chunk=chunk)
        assert len(chunked) == len(whole)
        for a, w in zip(chunked, whole):
            # Different GEMM batch shapes perturb fp accumulation slightly
            # (a few % relative on tiny KLs). Chunk size is functionally
            # equivalent: agree to 2% relative (or 1e-7 absolute), which is far
            # below any search-decision significance.
            assert abs(a - w) <= 1e-7 + 0.02 * abs(w), (chunk, a, w)

    # and whole vs slow full-model path
    for w, g in zip(whole, genomes):
        slow = evaluate_genome_on_set(pm, [batch], 0, g, 16)
        assert abs(w - slow) < 1e-6, (w, slow)


def test_pipeline_fast_vs_slow_multiseed_tiny():
    """End-to-end: the fast path must reproduce the slow path's search across
    several seeds (identical archived pruned sets)."""
    from trimreaper.pipeline import ratchet_search

    class _Fake:
        def __init__(self, seed=0):
            self.g = torch.Generator().manual_seed(seed)

        def batch(self, n, sl):
            return torch.randint(0, 3200, (n, sl), generator=self.g)

        def holdout_batches(self):
            return [self.batch(2, 16) for _ in range(2)]

    def _run(fast, seed):
        cfg = Config.defaults()
        cfg.model.layers = [0]
        cfg.ga.population = 6
        cfg.ga.elitism = 1
        cfg.ga.seed = seed
        cfg.search.start_target = 16
        cfg.search.ratchet_step = 16
        cfg.search.max_target = 32
        cfg.search.rounds = 2
        cfg.search.epsilon = 0.5
        cfg.search.valid_best_k = 3
        cfg.search.use_fast_eval = fast
        cfg.genome.min_rotations = 2
        cfg.genome.max_rotations = 4
        cfg.data.seq_len = 16
        cfg.data.fit_batch = 2
        validate(cfg)
        pm = load_tiny_smoke_model(cfg)
        st = _Fake(seed)
        res = ratchet_search(pm, cfg, 0, st, st.holdout_batches(), use_rotations=True)
        return [(p.removed, sorted(p.genome.pruned)) for p in res.points]

    for seed in (1, 2):
        fast = _run(True, seed)
        slow = _run(False, seed)
        assert fast == slow, (seed, fast, slow)
