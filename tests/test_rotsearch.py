"""Tests for the rotation-isolation experiment (frozen mask, rotations only)."""

import torch

import pytest

from trimreaper.config import Config, validate
from trimreaper.ga import coverage_stats, make_random_genome, make_child_population, GARandom
from trimreaper.model import load_tiny_smoke_model
from trimreaper.pipeline import ratchet_search


class FakeStreamer:
    def __init__(self, seed=0):
        self.g = torch.Generator().manual_seed(seed)

    def batch(self, n, sl):
        return torch.randint(0, 3200, (n, sl), generator=self.g)

    def holdout_batches(self):
        return [self.batch(2, 16) for _ in range(2)]

    def test_batches(self):
        return [self.batch(2, 16) for _ in range(2)]


def _cfg():
    cfg = Config.defaults()
    cfg.model.layers = [0]
    cfg.ga.population = 6
    cfg.ga.elitism = 1
    cfg.ga.seed = 3
    cfg.search.start_target = 16
    cfg.search.ratchet_step = 16
    cfg.search.max_target = 16
    cfg.search.rounds = 3
    cfg.search.epsilon = 0.5
    cfg.search.valid_best_k = 3
    cfg.search.use_fast_eval = True
    cfg.search.freeze_mask = True
    cfg.genome.min_rotations = 2
    cfg.genome.max_rotations = 32
    cfg.genome.rotations_per_removed = 1.0
    cfg.data.seq_len = 16
    cfg.data.fit_batch = 2
    validate(cfg)
    return cfg


def test_frozen_mask_never_changes():
    """The whole point of rotation-isolation: the pruned mask must stay fixed
    while the GA optimizes rotations only."""
    cfg = _cfg()
    frozen = sorted([3, 7, 15, 22, 33, 40, 55, 61, 70, 82, 91, 100, 110, 115, 120, 127])
    cfg.search.frozen_pruned = frozen
    pm = load_tiny_smoke_model(cfg)
    st = FakeStreamer()
    res = ratchet_search(pm, cfg, 0, st, st.holdout_batches(), use_rotations=True,
                         archive_dir=None)
    assert res.points, "expected at least one archived genome"
    for p in res.points:
        assert sorted(p.genome.pruned) == frozen, "freeze_mask must pin the mask"
        assert p.genome.rotations, "rotation-isolation must keep some rotations"


def test_make_random_genome_frozen_mask():
    cfg = _cfg()
    frozen = sorted([0, 5, 12, 30, 44])
    cfg.search.frozen_pruned = frozen
    from trimreaper.ga import make_random_genome

    rng = GARandom(cfg).python
    width = 128
    for _ in range(20):
        g = make_random_genome(cfg, width, len(frozen), rng, set(frozen))
        assert sorted(g.pruned) == frozen
        assert g.rotations  # still optimizes rotations


def test_make_child_population_frozen_mask():
    from trimreaper.ga import Genome, Individual

    cfg = _cfg()
    frozen = sorted([0, 5, 12, 30, 44, 60, 77, 90])
    rng = GARandom(cfg).python
    width = 128
    parents = []
    for _ in range(6):
        g = make_random_genome(cfg, width, len(frozen), rng, set(frozen))
        parents.append(Individual(genome=g, fitness=rng.random()))
    children = make_child_population(cfg, parents, width, len(frozen), rng, set(frozen))
    for c in children:
        assert sorted(c.pruned) == frozen


def test_coverage_stats():
    from trimreaper.rotation import PairRotation

    g = __import__("trimreaper.ga", fromlist=["Genome"]).Genome()
    g.pruned = set([10, 20, 30])
    g.rotations = [PairRotation(10, 5, 0.7), PairRotation(30, 40, -1.1), PairRotation(10, 20, 0.3)]
    stats = coverage_stats(g)
    assert stats["n_rot"] == 3
    assert stats["unique_channels_touched"] == 5          # {10,5,30,40,20}
    assert stats["unique_pruned_touched"] == 3             # {10,30,20}
