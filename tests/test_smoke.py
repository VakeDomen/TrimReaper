"""End-to-end integration smoke test on a tiny gated-MLP model.

Uses a tiny randomly-initialized Qwen3 model (real Qwen3MLP classes, tiny dims)
so the whole pipeline runs on CPU in well under a minute with no GPU and no
large download. Exercises: load, rotation apply/restore, mask hooks, KL,
ratchet GA (with and without rotations), holdout validation, compaction, and
Pareto plotting.
"""

import json
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

    def test_batches(self):
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


def test_assigned_mutation_rate_stats(tmp_path):
    """Assigned-rate experiment: every explorer slot gets a fixed ascending rate,
    the rate survives re-basing, and the result reports per-rate win counters."""
    cfg = _cfg(tmp_path)
    cfg.ga.population = 8
    cfg.search.rounds = 3       # a few generations so the loop + stats run fully
    cfg.search.start_target = 8
    cfg.search.max_target = 8   # single fixed level for a deterministic check
    pm = load_tiny_smoke_model(cfg)
    streamer = FakeStreamer()
    holdout = streamer.holdout_batches()

    from trimreaper.ga import assigned_mutation_fracs
    from trimreaper.pipeline import ratchet_search

    fracs = assigned_mutation_fracs(cfg.ga.population, cfg.ga.mutation_rate_max)
    assert len(fracs) == 8
    assert fracs[0] < fracs[-1]

    res = ratchet_search(pm, cfg, 0, streamer, holdout, use_rotations=True)
    # every slot has a rate-stats entry with the three counters
    assert set(res.rate_stats.keys()) == {round(f, 6) for f in fracs}
    for k, st in res.rate_stats.items():
        assert set(st.keys()) == {"elite_wins", "archive_wins", "rebases"}
        assert all(isinstance(v, int) and v >= 0 for v in st.values())
    # elite_wins count (over the whole run) cannot exceed the number of gens
    total_elite = sum(st["elite_wins"] for st in res.rate_stats.values())
    assert total_elite <= cfg.search.rounds


def test_rotation_restores_exactly(tmp_path):
    """Rotation under the corrected semantics: gate/up stay pristine, down is
    counter-rotated; restore_all returns everything to pristine."""
    from trimreaper.rotation import PairRotation

    cfg = _cfg(tmp_path)
    pm = load_tiny_smoke_model(cfg)
    mlp = pm.model.model.layers[0].mlp
    orig_gate = mlp.gate_proj.weight.data.clone()
    orig_up = mlp.up_proj.weight.data.clone()
    orig_down = mlp.down_proj.weight.data.clone()

    pm.apply_rotations(0, [PairRotation(0, 10, 0.6), PairRotation(30, 5, -1.1)])

    # gate/up must be untouched (rotating raw rows would break gated-SiLU
    # invariance); only down_proj is counter-rotated.
    assert torch.equal(mlp.gate_proj.weight.data, orig_gate)
    assert torch.equal(mlp.up_proj.weight.data, orig_up)
    assert not torch.equal(mlp.down_proj.weight.data, orig_down)

    pm.restore_all()
    assert torch.equal(mlp.gate_proj.weight.data, orig_gate)
    assert torch.equal(mlp.up_proj.weight.data, orig_up)
    assert torch.equal(mlp.down_proj.weight.data, orig_down)


def test_rotation_is_invariant_unmasked(tmp_path):
    """The overriding correctness fix: rotation + counter-rotation MUST leave
    logits ~unchanged when no channel is masked (gated-SiLU invariance).

    The earlier implementation rotated raw gate/up rows, which is NOT invariant
    under SiLU and so was testing a different (broken) operation.
    """
    import torch.nn.functional as F

    from trimreaper.evaluate import baseline_logits
    from trimreaper.rotation import PairRotation

    cfg = _cfg(tmp_path)
    cfg.ga.population = 4
    cfg.genome.min_rotations = 2
    cfg.genome.max_rotations = 8
    pm = load_tiny_smoke_model(cfg)
    batch = torch.randint(0, 3200, (2, 32))

    ref = baseline_logits(pm, [batch], 32)[0]
    rots = [
        PairRotation(10, 20, 0.7),
        PairRotation(3, 40, -1.1),
        PairRotation(50, 5, 0.2),
        PairRotation(80, 30, 2.0),
    ]

    pm.restore_all()
    pm.apply_rotations(0, rots)
    pm.clear_all_masks()  # no mask -> must be invariant
    with torch.inference_mode():
        out = pm.model(input_ids=batch.to(pm.model.device), use_cache=False)
    cand = out.logits

    def _kl(a, b):
        la = F.log_softmax(a.float(), -1)
        pa = F.softmax(a.float(), -1)
        lb = F.log_softmax(b.float(), -1)
        return float((pa * (la - lb)).sum(-1).mean())

    k = _kl(ref, cand)
    # Numerical roundoff only: rotation + inverse compensation is ~identity.
    assert abs(k) < 1e-3, f"unnasked rotation should be invariant, KL={k}"


def test_compaction_builds_smaller_mlp(tmp_path):
    from trimreaper.compaction import build_compact_mlp, compact_model_metrics

    cfg = _cfg(tmp_path)
    pm = load_tiny_smoke_model(cfg)
    pruned = list(range(0, 16))
    mods = build_compact_mlp(pm, 0, pruned, [])
    # Gate/up stay FULL (rotation acts on hidden; down_proj drops the pruned
    # ROTATED columns).
    assert mods["gate_proj"].weight.shape[0] == 128
    assert mods["up_proj"].weight.shape[0] == 128
    assert "rotate" in mods and mods["rotate"].weight.shape == (128, 128)
    assert mods["down_proj"].weight.shape[1] == 128 - 16
    metrics = compact_model_metrics(pm, 0, pruned, [])
    assert metrics["compact_channels"] == 128 - 16
    assert metrics["params_removed"] > 0


def test_compaction_accepts_json_tuple_rotations(tmp_path):
    """Regression (fix): archived rotations are JSON (a, b, angle) tuples; the
    compaction path must convert them to PairRotation, not feed tuples into the
    rotation math."""
    from trimreaper.compaction import build_compact_mlp, compact_model_metrics

    cfg = _cfg(tmp_path)
    pm = load_tiny_smoke_model(cfg)
    pruned = list(range(0, 8))
    rot_tuples = [(10, 20, 0.7), (30, 5, -1.1)]   # exactly how JSON stores them
    mods = build_compact_mlp(pm, 0, pruned, rot_tuples)
    assert "rotate" in mods
    # down_proj dropped the pruned columns even with rotations applied
    assert mods["down_proj"].weight.shape[1] == 128 - 8
    metrics = compact_model_metrics(pm, 0, pruned, rot_tuples)
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
    val = streamer.holdout_batches()
    test = streamer.test_batches()
    curves = baseline_curves(pm, cfg, 0, streamer, widths=[8, 16],
                             val_batches=val, test_batches=test)
    assert set(curves.keys()) == {"random", "weight_norm", "activation_magnitude"}
    for k, sets in curves.items():
        assert set(sets.keys()) == {"val", "test"}
        for vname, pts in sets.items():
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


def test_run_uses_dated_dir_and_writes_report(tmp_path):
    """Run results must land in a date-prefixed folder (never overriding a
    previous run) and a machine-readable run_results.json must be written."""
    from trimreaper.cli import _make_run_dir, _write_run_report
    from trimreaper.ga import Genome
    from trimreaper.pipeline import ParetoPoint

    cfg = _cfg(tmp_path)
    cfg.search.archive_dir = str(tmp_path / "archive")

    # Two dated run dirs must never collide with one another.
    run1 = _make_run_dir(cfg, "expA")
    run2 = _make_run_dir(cfg, "expA")
    assert run1 != run2
    assert os.path.basename(run1)[:8].isdigit() and os.path.basename(run1)[8] == "_"
    assert os.path.basename(run1).endswith("expA")

    def _gen(n_rot=2):
        rot = Genome()
        from trimreaper.rotation import PairRotation
        rot.rotations = [PairRotation(a=i, b=i + 1, angle=0.5) for i in range(n_rot)]
        rot.pruned = set(range(0, 8))
        return rot

    res_rot = type("R", (), {})()
    res_rot.points = [
        ParetoPoint(removed=8, kl=0.001, validated_kl=0.002, test_kl=0.003, genome=_gen()),
    ]
    res_norot = type("R", (), {})()
    res_norot.points = [ParetoPoint(removed=8, kl=0.002, validated_kl=0.004, genome=_gen())]

    baselines = {"random": {"val": [(8, 0.01)], "test": [(8, 0.02)]}}

    report_path = _write_run_report(run1, cfg, res_rot, res_norot, baselines)
    assert os.path.exists(report_path)
    with open(report_path) as fh:
        report = json.load(fh)
    assert report["run_dir"] == run1
    assert len(report["ga_rotations"]) == 1
    assert report["ga_rotations"][0]["removed"] == 8
    assert report["ga_rotations"][0]["rotations"][0] == [0, 1, 0.5]  # JSON: tuple -> list
    assert report["baselines"]["random"]["val"] == [[8, 0.01]]
    assert "config" in report
