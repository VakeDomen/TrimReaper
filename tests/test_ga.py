"""Unit tests for GA genome operators and config."""

import random

from trimreaper.config import Config
from trimreaper.ga import (
    Genome,
    Individual,
    crossover,
    fitness_value,
    make_random_genome,
    mutate,
    mutate_mask,
    rotation_budget,
)


def _cfg():
    c = Config.defaults()
    c.genome.min_rotations = 4
    c.genome.max_rotations = 8
    c.ga.population = 8
    c.search.epsilon = 0.01
    return c


def test_random_genome_respects_bounds():
    cfg = _cfg()
    rng = random.Random(0)
    g = make_random_genome(cfg, width=100, target=10, rng=rng)
    assert cfg.genome.min_rotations <= len(g.rotations) <= cfg.genome.max_rotations
    assert len(g.pruned) == 10
    for rot in g.rotations:
        assert 0 <= rot.a < 100 and 0 <= rot.b < 100 and rot.a != rot.b


def test_rotation_budget_scales_with_target():
    """Larger deletion targets must get more rotational freedom (clamped)."""
    cfg = _cfg()  # min=4, max=8, rotations_per_removed defaults to 0.5
    # With the scaled formula (0.5*target clamped to [4, 8]):
    assert rotation_budget(cfg, 8) == 4    # 0.5*8=4 -> floor
    assert rotation_budget(cfg, 16) == 8   # 0.5*16=8 -> cap at max_rotations
    # Explicit scaling check on a custom multiplier:
    cfg2 = Config.defaults()
    cfg2.genome.min_rotations = 0
    cfg2.genome.max_rotations = 1000
    cfg2.genome.rotations_per_removed = 0.5
    assert rotation_budget(cfg2, 256) == 128
    assert rotation_budget(cfg2, 512) == 256
    assert rotation_budget(cfg2, 1024) == 512
    assert rotation_budget(cfg2, 4096) == 1000  # clamped to ceiling


def test_with_target_pruned_resizes():
    cfg = _cfg()
    rng = random.Random(1)
    g = make_random_genome(cfg, width=100, target=5, rng=rng)
    g.with_target_pruned(12, rng, 100)
    assert len(g.pruned) == 12
    g.with_target_pruned(3, rng, 100)
    assert len(g.pruned) == 3


def test_mutate_produces_valid_genome():
    cfg = _cfg()
    rng = random.Random(2)
    base = make_random_genome(cfg, width=50, target=4, rng=rng)
    seen = set()
    for _ in range(200):
        m = mutate(cfg, base, 50, 4, rng)
        # deterministic-ish: never returns the identical object
        seen.add(tuple(rr.to_tuple() for rr in m.rotations))
        for rot in m.rotations:
            assert rot.a != rot.b
    assert len(seen) > 1  # actually mutates


def test_crossover_respects_target_count():
    cfg = _cfg()
    rng = random.Random(3)
    pa = make_random_genome(cfg, width=80, target=8, rng=rng)
    pb = make_random_genome(cfg, width=80, target=9, rng=rng)
    child = crossover(cfg, pa, pb, 80, target=7, rng=rng)
    assert len(child.pruned) == 7
    for rot in child.rotations:
        assert rot.a != rot.b


def test_fitness_value_penalty_ordering():
    cfg = _cfg()
    cfg.search.epsilon = 0.01
    cfg.search.penalty_scale = 10.0
    # over-epsilon should be worse than within-epsilon even if raw KL lower
    f_within = fitness_value(cfg, kl=0.005, removed=8)
    f_over = fitness_value(cfg, kl=0.02, removed=8)
    assert f_over > f_within
    assert f_within == 0.005


def test_config_override_and_validate():
    cfg = Config.defaults()
    cfg.override({"ga.population": 16, "data.seq_len": 128})
    assert cfg.ga.population == 16
    assert cfg.data.seq_len == 128
    from trimreaper.config import validate

    validate(cfg)  # should not raise
