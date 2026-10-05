"""Unit tests for GA genome operators and config."""

import random

from trimreaper.config import Config
from trimreaper.ga import (
    Genome,
    Individual,
    clone_genome,
    crossover,
    fitness_value,
    make_random_genome,
    mutate,
    mutate_from_self,
    mutate_mask,
    rebase_explorer,
    rotation_budget,
    tournament_parent,
)


def _cfg():
    c = Config.defaults()
    c.genome.min_rotations = 4
    c.genome.max_rotations = 8
    c.genome.rotations_per_removed = 0.5
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


def test_random_genome_mask_before_rotations():
    """Regression (fix): the pruning mask must exist BEFORE rotations are
    generated, otherwise delete<->survive-biased pairs have no known deleted
    channels and the 70% bias never fires on initial genomes."""
    cfg = _cfg()
    cfg.genome.delete_survive_bias = 1.0   # force the bias
    rng = random.Random(0)
    g = make_random_genome(cfg, width=100, target=10, rng=rng)
    assert len(g.pruned) == 10
    # Every rotation must pair a deleted channel with a survivor (bias=1.0).
    assert g.rotations, "expected at least one rotation"
    for rot in g.rotations:
        dead, alive = set(g.pruned), set(range(100)) - set(g.pruned)
        assert (rot.a in dead and rot.b in alive) or (rot.b in dead and rot.a in alive), \
            f"rotation ({rot.a},{rot.b}) not delete<->survive biased"


def test_tournament_parent_favors_fitness():
    """Selection pressure: a fitter individual must win tournaments more often
    than a worse one (best-of-k, chosen independently per parent)."""
    cfg = _cfg()
    rng = random.Random(0)
    pop = [Individual(genome=Genome(), fitness=f) for f in (1.0, 0.0, 10.0)]
    wins = {id(pop[i]): 0 for i in range(3)}
    for _ in range(200):
        p = tournament_parent(pop, tournament_size=3, rng=rng)
        wins[id(p)] += 1
    # The best (fitness 0.0) must win (nearly) every time with k=3.
    assert wins[id(pop[1])] == 200


def test_crossover_mask_keeps_intersection_and_target():
    """Controlled mask crossover: preserves the parental intersection and ends
    at exactly the target size."""
    cfg = _cfg()
    rng = random.Random(4)
    pa_pruned = {0, 1, 2, 3, 4, 5, 6, 7}
    pb_pruned = {2, 3, 4, 5, 8, 9, 10, 11}
    pa = Genome(rotations=[], pruned=set(pa_pruned))
    pb = Genome(rotations=[], pruned=set(pb_pruned))
    child = crossover(cfg, pa, pb, width=100, target=6, rng=rng)
    # Intersection {2,3,4,5} preserved entirely (it's within target).
    assert {2, 3, 4, 5} <= child.pruned
    assert len(child.pruned) == 6


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


def test_fail_limit_default():
    """Independent-explorer mode: a candidate re-bases after fail_limit gens."""
    assert Config.defaults().ga.fail_limit == 5


def test_clone_genome_keeps_or_refreshes_id():
    cfg = _cfg()
    rng = random.Random(9)
    base = make_random_genome(cfg, width=40, target=4, rng=rng)
    keep = clone_genome(base, keep_id=True)
    assert keep.id == base.id
    assert set(keep.pruned) == base.pruned
    assert [r.to_tuple() for r in keep.rotations] == [r.to_tuple() for r in base.rotations]
    # keep_id=False -> a fresh identity, independent memory
    fresh = clone_genome(base, keep_id=False)
    assert fresh.id != keep.id
    fresh.rotations = []
    assert base.rotations  # independent copy unaffected


def test_mutate_from_self_no_crossover_single_parent():
    """Each explorer mutates ONLY from itself (single self-parent, fresh id)."""
    cfg = _cfg()
    rng = random.Random(11)
    base = make_random_genome(cfg, width=40, target=4, rng=rng)
    child = mutate_from_self(cfg, base, 40, 4, rng)
    assert child.id != base.id
    assert child.parent_ids == [base.id]           # one self-parent, never two
    assert len(child.pruned) == 4                  # mask kept + target preserved


def test_rebase_explorer_copies_and_mutates_base():
    """A stuck explorer is re-based onto a base: copy + mutate, fresh id."""
    cfg = _cfg()
    rng = random.Random(13)
    base = make_random_genome(cfg, width=40, target=4, rng=rng)
    rebased = rebase_explorer(cfg, base, 40, 4, rng)
    assert rebased.id != base.id
    assert rebased.parent_ids == [base.id]
    assert len(rebased.pruned) == 4


def test_angle_mutate_frac_default():
    """Reviewer fix #3: independent-explorer mutation targets ~5% of angles per
    generation by default (angle_mutate_frac)."""
    assert Config.defaults().ga.angle_mutate_frac == 0.05


def test_mutate_from_self_mutation_is_guaranteed():
    """Reviewer fix #3: in independent-explorer mode mutation must happen 100%
    of the time (no mutation_rate gate), so EVERY child differs from its parent
    — not just on the flip of a coin. Across many draws every child must change
    at least its angles."""
    cfg = _cfg()
    cfg.ga.angle_mutate_frac = 0.5        # force many angle changes for a clean check
    cfg.ga.replace_a_p = 0.0              # isolate the angle path
    cfg.ga.replace_b_p = 0.0
    cfg.ga.add_rotation_p = 0.0
    cfg.ga.remove_rotation_p = 0.0
    cfg.ga.flip_prune_p = 0.0             # no mask swap; isolate angle mutation
    rng = random.Random(21)
    parent = make_random_genome(cfg, width=40, target=4, rng=rng)
    assert len(parent.rotations) >= 1
    for _ in range(200):
        child = mutate_from_self(cfg, parent, 40, 4, rng)
        # every child must have mutated its angles (100% guaranteed) even though
        # every other mutation operator is disabled.
        p_angles = [r.angle for r in parent.rotations]
        c_angles = [r.angle for r in child.rotations]
        assert c_angles != p_angles, "angle mutation must fire 100% of the time"


def test_mask_mutation_then_repair_keeps_cross_boundary():
    """Reviewer fix #4: mask mutation happens FIRST, then rotations are repaired
    against the new mask. With the flip gate forced on, a rotation straddling a
    swapped channel must end up back on the correct side (or remain valid), and
    every rotation must stay a != b."""
    cfg = _cfg()
    cfg.ga.flip_prune_p = 1.0             # ALWAYS swap one pruned -> surviving
    cfg.ga.delete_survive_bias = 1.0      # repair re-draws strictly cross-boundary
    rng = random.Random(31)
    for _ in range(100):
        parent = make_random_genome(cfg, width=64, target=8, rng=rng)
        child = mutate_from_self(cfg, parent, 64, 8, rng)
        assert len(child.pruned) == 8                  # exact target preserved
        for rot in child.rotations:
            assert rot.a != rot.b
            assert 0 <= rot.a < 64 and 0 <= rot.b < 64


def test_repair_rotations_after_mask_direct():
    """Direct check of the repair step: after swapping a channel, a rotation
    that ended up with both endpoints on the same side is re-drawn to cross the
    boundary (delete_survive_bias=1.0 makes the re-draw deterministic)."""
    from trimreaper.ga import _mask_swap, repair_rotations_after_mask
    from trimreaper.rotation import PairRotation

    cfg = _cfg()
    cfg.ga.delete_survive_bias = 1.0
    rng = random.Random(7)
    # pruned = {0,1}; rotation (0,5) starts cross-boundary (0 deleted, 5 kept).
    g = Genome(rotations=[PairRotation(0, 5, 0.3)], pruned={0, 1})
    # Force a swap via the mask mutator on a copy to learn the changed channels.
    from trimreaper.ga import mutate_mask
    child = mutate_mask(cfg, g, 64, rng)
    if child.pruned != {0, 1}:
        changed = set(g.pruned) ^ set(child.pruned)
    else:
        changed = set()
    repair_rotations_after_mask(cfg, child, 64, changed, rng)
    for rot in child.rotations:
        assert rot.a != rot.b
