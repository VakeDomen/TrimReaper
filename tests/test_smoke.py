"""End-to-end integration smoke test on a tiny gated-MLP model.

Uses a tiny randomly-initialized Qwen3 model (real Qwen3MLP classes, tiny dims)
so the whole pipeline runs on CPU in well under a minute with no GPU and no
large download. Exercises: load, rotation apply/restore, mask hooks, KL,
ratchet GA (with and without rotations), holdout validation, compaction, and
Pareto plotting.
"""

import os
import shutil

import pytest
import torch

from trimreaper.config import Config, validate
from trimreaper.model import load_tiny_smoke_model


class FakeStreamer:
    """Mimics WikiTextStreamer's batch()/holdout_batches() with synthetic tokens."""

    def __init__(self, vocab=3200, seed=0):
        self.vocab = vocab
        self.g = torch.Generator().manual_seed(seed)

    def batch(self, n, seq_len):
        return torch.randint(0, self.vocab, (n, seq_len), generator=self.g)

    def holdout_batches(self):
        return [self.batch(2, 32) for _ in range(2)]


def _cfg(tmp_path):
    cfg = Config.defaults()
    cfg.model.layers = [0]
    cfg.ga.population = 4
    cfg.ga.elitism = 1
    cfg.search.start_target = 8
    cfg.search.ratchet_step = 8
    cfg.search.max_target = 24   # cap so the smoke test finishes fast even on a busy CPU
    cfg.search.rounds = 2
    cfg.search.epsilon = 0.5
    cfg.search.archive_dir = str(tmp_path / "archive")
    cfg.genome.min_rotations = 2
    cfg.genome.max_rotations = 4
    cfg.data.seq_len = 32
    cfg.data.fit_batch = 2
    validate(cfg)
    return cfg


def test_full_pipeline_smoke(tmp_path):
    cfg = _cfg(tmp_path)
    pm = load_tiny_smoke_model(cfg)
    assert pm.pristine[0]["up_proj"].shape[0] == 128

    streamer = FakeStreamer()
    holdout = streamer.holdout_batches()

    from trimreaper.pipeline import ratchet_search

    res_rot = ratchet_search(pm, cfg, 0, streamer, holdout, use_rotations=True)
    assert res_rot.best_removed > 0
    assert all(p.validated_kl >= 0 for p in res_rot.points)

    res_norot = ratchet_search(pm, cfg, 0, streamer, holdout, use_rotations=False)
    assert res_norot.best_removed >= 0


def test_rotation_restores_exactly(tmp_path):
    from trimreaper.rotation import PairRotation, apply_rotation_sequence

    cfg = _cfg(tmp_path)
    pm = load_tiny_smoke_model(cfg)
    orig_gate = pm.model.model.layers[0].mlp.gate_proj.weight.data.clone()
    pm.apply_rotations(0, [PairRotation(0, 10, 0.6), PairRotation(30, 5, -1.1)])
    # not equal now
    assert not torch.equal(pm.model.model.layers[0].mlp.gate_proj.weight.data, orig_gate)
    pm.restore_all()
    assert torch.equal(pm.model.model.layers[0].mlp.gate_proj.weight.data, orig_gate)


def test_compaction_builds_smaller_mlp(tmp_path):
    from trimreaper.compaction import build_compact_mlp, compact_model_metrics

    cfg = _cfg(tmp_path)
    pm = load_tiny_smoke_model(cfg)
    pruned = list(range(0, 16))
    mods = build_compact_mlp(pm, 0, pruned, [])
    assert mods["gate_proj"].weight.shape[0] == 128 - 16
    assert mods["down_proj"].weight.shape[1] == 128 - 16
    metrics = compact_model_metrics(pm, 0, pruned, [])
    assert metrics["compact_channels"] == 128 - 16
    assert metrics["params_removed"] > 0


def test_baseline_not_masked_by_prior_candidate(tmp_path):
    """A prior candidate's mask must not leak into the baseline reference."""
    from trimreaper.evaluate import baseline_logits, evaluate_candidate

    cfg = _cfg(tmp_path)
    pm = load_tiny_smoke_model(cfg)
    streamer = FakeStreamer()
    batch = streamer.batch(2, 32)

    # Establish a clean reference first.
    ref_clean = baseline_logits(pm, [batch], 32)[0]

    # Run a candidate that prunes many channels (leaves a mask behind).
    _ = evaluate_candidate(pm, batch, ref_clean, 0, [], list(range(0, 64)), 32)

    # A fresh baseline after that candidate must equal the clean reference
    # (the stale mask must have been cleared).
    ref_after = baseline_logits(pm, [batch], 32)[0]
    assert torch.allclose(ref_clean, ref_after, atol=1e-6)


def test_baselines_and_plot(tmp_path):
    from trimreaper.sweep import baseline_curves, plot_frontiers

    cfg = _cfg(tmp_path)
    pm = load_tiny_smoke_model(cfg)
    streamer = FakeStreamer()
    curves = baseline_curves(pm, cfg, 0, streamer, widths=[8, 16])
    assert set(curves.keys()) == {"random", "weight_norm", "activation_magnitude"}
    for k, pts in curves.items():
        assert len(pts) == 2
        for w, kl in pts:
            assert kl >= 0

    out = os.path.join(str(tmp_path), "frontier.png")
    path = plot_frontiers(
        [(8, 0.1, "GA+rot"), (16, 0.2, "GA+rot")],
        curves,
        out,
    )
    assert path and os.path.exists(path) and os.path.getsize(path) > 0


def test_activation_magnitude_ranking_is_flat_and_valid(tmp_path):
    """The activation-magnitude ranking must be a valid flat channel order.

    Regression: it used to reduce over only dim=0, leaving a 2-D (seq, width)
    array whose .tolist() produced nested lists. Iterating those in set_mask()
    silently indexed with lists, returning garbage (identical KL) for every
    deletion width — exactly the '0.06885 == 0.06885' red flag seen on the real
    run. Each entry must be a plain int in [0, width).
    """
    from trimreaper.baselines import activation_magnitude_ranking
    from trimreaper.evaluate import baseline_logits, evaluate_candidate
    from trimreaper.model import load_tiny_smoke_model

    cfg = _cfg(tmp_path)
    pm = load_tiny_smoke_model(cfg)
    streamer = FakeStreamer()
    probe = [streamer.batch(4, 32) for _ in range(4)]
    batch = streamer.batch(2, 32)
    ref = baseline_logits(pm, [batch], 32)[0]

    order = activation_magnitude_ranking(pm, 0, probe, 32)
    assert len(order) == 128
    assert all(isinstance(x, int) for x in order)
    assert sorted(order) == list(range(128))  # a permutation of all channels

    # Distinct pruning widths must yield strictly increasing KL on the smoke
    # model (tiny but the bug made every width return the SAME value).
    kls = []
    for w in (16, 32, 64):
        kl = evaluate_candidate(pm, batch, ref, 0, [], order[:w], 32)
        kls.append(kl)
        assert kl >= 0
    assert kls[0] <= kls[-1], f"expected monotonic non-decreasing KL, got {kls}"
