"""Tests for the analytical fixed-mask pre-analysis and GA seeding.

Covers:
  - channel_importance / importance_order / select_fixed_mask
  - local_pca_rotations (the greedy Givens builder) and its energy report
  - seed_population_from (GA seeding around an analytical template)
"""

from __future__ import annotations

import random

import torch

from src.trimreaper.analysis import (
    channel_importance,
    importance_order,
    local_pca_rotations,
    select_fixed_mask,
)
from src.trimreaper.config import Config
from src.trimreaper.ga import assigned_mutation_fracs, seed_population_from
from src.trimreaper.rotation import PairRotation, make_orthogonal_matrix


def _hidden(width: int = 8, n: int = 2000, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    H = torch.randn(n, width)
    # channel 0: tiny activation -> low importance
    H[:, 0] = 0.02 * torch.randn(n)
    return H


def test_importance_ranks_low_activity_channel_first():
    H = _hidden()
    dw = torch.randn(5, 8)          # down_proj (hidden_out, width)
    dw[:, 1] = 5.0                  # huge outgoing norm on channel 1
    score = channel_importance([H], dw)
    order = importance_order(score)
    # channel 0 (tiny activation) is least important => first
    assert order[0] == 0
    # channel 1 (huge down-norm) is most important => last
    assert order[-1] == 1
    # importance is a positive scalar per channel
    assert tuple(score.shape) == (8,)


def test_importance_multibatch_matches_single():
    H = _hidden()
    dw = torch.randn(5, 8)
    single = channel_importance([H], dw)
    multi = channel_importance([H, H, H], dw)
    assert torch.allclose(single, multi, atol=1e-6)


def test_select_fixed_mask_includes_weakest_and_respects_skip():
    score = torch.arange(8.0)       # ascending importance: 0 most removable
    sel = select_fixed_mask(score, target=3)
    assert sel == [0, 1, 2]         # weakest auto-included (skip_bottom=0)
    sel2 = select_fixed_mask(score, target=3, skip_bottom=2)
    assert sel2 == [2, 3, 4]        # skips the 2 very weakest


def test_local_pca_returns_one_cross_boundary_rotation_per_deleted():
    H = _hidden(width=8, n=3000)
    pruned = [1, 2]
    rots, rep = local_pca_rotations(H, pruned, max_rows=None)
    assert rep["n_rot"] == len(pruned) == len(rots)
    deleted = set(pruned)
    kept = set(range(8)) - deleted
    for r in rots:
        # exactly one endpoint is a deleted channel, the other is kept
        in_deleted = (r.a in deleted, r.b in deleted)
        assert in_deleted.count(True) == 1
        assert (r.a in kept) != (r.b in kept)
        assert r.a != r.b


def test_local_pca_reduces_deleted_energy_and_report_matches_composed():
    torch.manual_seed(0)
    H = _hidden(width=8, n=4000, seed=1)
    # make channels correlated with a natural partner so the rotation can help
    H[:, 3] = 0.8 * H[:, 1] + 0.6 * H[:, 3]
    H[:, 5] = 0.7 * H[:, 2] + 0.5 * H[:, 5]
    pruned = [1, 2]
    rots, rep = local_pca_rotations(H, pruned, max_rows=None)
    assert rep["energy_ratio"] < 1.0         # energy genuinely pushed into kept
    # report's composed residual matches applying Q to the ORIGINAL data
    Q = make_orthogonal_matrix(H.shape[-1], rots)
    Y = H @ Q
    composed_total = float((Y[:, pruned] ** 2).sum(dim=1).mean())
    assert abs(composed_total - rep["deleted_after_energy"]) < 1e-4


def test_local_pca_does_not_mutate_input():
    H = _hidden(width=8, n=2000, seed=2)
    pruned = [0, 4]
    clone = H.clone()
    local_pca_rotations(H, pruned, max_rows=None)
    assert torch.equal(H, clone)


def test_local_pca_angle_zeroes_shared_direction():
    # a cleanly-constructible case: channel d and a partner k are perfectly
    # correlated, so a single rotation should drive E[h_d^2] toward ~0.
    torch.manual_seed(3)
    n = 5000
    d, k = 2, 4
    base = torch.randn(n)
    H = torch.randn(n, 6)
    H[:, d] = 0.5 * base
    H[:, k] = 0.9 * base
    H[:, d] = H[:, d] + 0.05 * torch.randn(n)
    rots, rep = local_pca_rotations(H, [d], max_rows=None)
    Q = make_orthogonal_matrix(6, rots)
    Y = H @ Q
    assert float((Y[:, d] ** 2).mean()) < 0.05  # nearly all energy moved out


def test_seed_population_from_counts_tiers_and_frozen_mask():
    cfg = Config()
    cfg.ga.population = 20
    cfg.genome.min_rotations = 1
    cfg.genome.max_rotations = 64
    from src.trimreaper.ga import Genome
    template = Genome(rotations=[PairRotation(0, 1, 0.5), PairRotation(2, 3, 1.0)], pruned={0, 2})
    rng = random.Random(0)
    fracs = assigned_mutation_fracs(20, 0.003, 0.10)
    pop = seed_population_from(
        cfg, template, 20, fracs, rng, width=8, target=2, n_exact=1, n_small=5, n_large=5
    )
    assert len(pop) == 20
    # every individual shares the template's fixed mask
    for ind in pop:
        assert ind.genome.pruned == {0, 2}
    # slot 0 is the exact clone
    assert [r.to_tuple() for r in pop[0].genome.rotations] == [
        r.to_tuple() for r in template.rotations
    ]
    # small/large tiers got perturbed (some angle changed)
    small_rots = [r.to_tuple() for r in pop[1].genome.rotations]
    assert small_rots != [r.to_tuple() for r in template.rotations]
    # random slots build their OWN random rotations (different from the template)
    assert [r.to_tuple() for r in pop[11].genome.rotations] != [
        r.to_tuple() for r in template.rotations
    ]
    # per-slot mutation fracs are preserved ascending
    assert [round(ind.mutation_frac, 5) for ind in pop[:2]] == [round(fracs[0], 5), round(fracs[1], 5)]
