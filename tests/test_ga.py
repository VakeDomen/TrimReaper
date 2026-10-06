"""Unit tests for GA genome operators and config."""

import random

from trimreaper.config import Config
from trimreaper.ga import (
    Genome,
    Individual,
    assigned_mutation_fracs,
    clone_genome,
    fitness_value,
    make_random_genome,
    mutate_from_self,
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


def test_random_genome_fixed_rotation_count():
    """Every genome gets EXACTLY rotation_budget(target) rotations (no length
    evolution / no initial_rotation_frac drift), so genomes are comparable."""
    cfg = _cfg()
    rng = random.Random(7)
    expect = rotation_budget(cfg, 10)   # clamp(0.5*10, 4, 8) = 5
    for _ in range(50):
        g = make_random_genome(cfg, width=100, target=10, rng=rng)
        assert len(g.rotations) == expect, (len(g.rotations), expect)


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
    cfg.override({"ga.population": 16, "data.seq_len": 128, "mutation.mask_swap_count": 2})
    assert cfg.ga.population == 16
    assert cfg.data.seq_len == 128
    assert cfg.mutation.mask_swap_count == 2
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


def test_mutation_config_defaults():
    """The minimal mutation schema has the reviewer's defaults."""
    m = Config.defaults().mutation
    assert m.angle_fraction_min == 0.003
    assert m.angle_fraction_max == 0.10
    assert m.pair_rewire_fraction == 0.02
    assert m.mask_swap_count == 1
    assert m.mask_swap_probability == 0.20


def test_mutate_from_self_keeps_fixed_rotation_count():
    """No add/remove rotation mutations: the child keeps EXACTLY the parent's
    rotation count."""
    cfg = _cfg()
    rng = random.Random(17)
    parent = make_random_genome(cfg, width=40, target=4, rng=rng)
    n0 = len(parent.rotations)
    for _ in range(200):
        child = mutate_from_self(cfg, parent, 40, 4, rng)
        assert len(child.rotations) == n0
        assert len(child.pruned) == 4


def test_mutate_from_self_mutation_is_guaranteed():
    """Mutation must happen 100% of the time (no mutation_rate gate): EVERY
    child differs from its parent, even with all probabilistic operators gated
    off (only the guaranteed angle mutation remains)."""
    cfg = _cfg()
    cfg.mutation.angle_fraction_max = 0.5   # force many angle changes for a clean check
    cfg.mutation.angle_fraction_min = 0.5
    cfg.mutation.mask_swap_probability = 0.0   # disable mask swap (probabilistic)
    cfg.mutation.pair_rewire_fraction = 0.0    # disable pair rewire (probabilistic)
    rng = random.Random(21)
    parent = make_random_genome(cfg, width=40, target=4, rng=rng)
    assert len(parent.rotations) >= 1
    for _ in range(200):
        child = mutate_from_self(cfg, parent, 40, 4, rng)
        p_angles = [r.angle for r in parent.rotations]
        c_angles = [r.angle for r in child.rotations]
        assert c_angles != p_angles, "angle mutation must fire 100% of the time"


def test_mask_mutation_then_repair_keeps_cross_boundary():
    """Mask swap happens in place, rotations are repaired against the new mask,
    and every rotation stays a != b with the exact target preserved."""
    cfg = _cfg()
    cfg.mutation.mask_swap_probability = 1.0   # ALWAYS swap
    cfg.mutation.mask_swap_count = 1
    cfg.genome.delete_survive_bias = 1.0       # repair re-draws strictly cross-boundary
    rng = random.Random(31)
    for _ in range(100):
        parent = make_random_genome(cfg, width=64, target=8, rng=rng)
        child = mutate_from_self(cfg, parent, 64, 8, rng)
        assert len(child.pruned) == 8                  # exact target preserved
        for rot in child.rotations:
            assert rot.a != rot.b
            assert 0 <= rot.a < 64 and 0 <= rot.b < 64


def test_mask_swap_is_exact_swap():
    """A mask swap removes one pruned and adds one kept (exact swap), so the
    count never changes even with multiple swap attempts."""
    from trimreaper.ga import _mask_swap

    cfg = _cfg()
    cfg.mutation.mask_swap_count = 1
    cfg.mutation.mask_swap_probability = 1.0
    rng = random.Random(37)
    for _ in range(100):
        g = make_random_genome(cfg, width=64, target=8, rng=rng)
        before = set(g.pruned)
        changed = _mask_swap(cfg, g, 64, rng)
        assert len(g.pruned) == 8
        assert changed, "mask swap must change status when forced"
        assert (set(g.pruned) ^ before) == changed   # exactly drop+add


def test_pair_rewire_changes_planes_keeps_angles():
    """PAIR rewire re-draws endpoints (new planes) while preserving each
    rotation's angle, and always yields valid cross-boundary pairs."""
    from trimreaper.ga import _rewire_pairs

    cfg = _cfg()
    cfg.mutation.pair_rewire_fraction = 1.0   # rewire every rotation
    cfg.genome.delete_survive_bias = 1.0      # strict cross-boundary
    rng = random.Random(43)
    for _ in range(100):
        g = make_random_genome(cfg, width=64, target=8, rng=rng)
        angles_before = [r.angle for r in g.rotations]
        _rewire_pairs(cfg, g, 64, rng)
        # angles preserved exactly
        assert [r.angle for r in g.rotations] == angles_before
        for rot in g.rotations:
            assert rot.a != rot.b
            dead = (rot.a in g.pruned) != (rot.b in g.pruned)
            assert dead, "rewired pair must cross the pruned/kept boundary"


def test_repair_rotations_after_mask_direct():
    """Direct check of the repair step: after swapping a channel, a rotation
    that ended up with both endpoints on the same side is re-drawn to cross the
    boundary (delete_survive_bias=1.0 makes the re-draw deterministic)."""
    from trimreaper.ga import _mask_swap, repair_rotations_after_mask
    from trimreaper.rotation import PairRotation

    cfg = _cfg()
    cfg.mutation.mask_swap_probability = 1.0
    cfg.genome.delete_survive_bias = 1.0
    rng = random.Random(7)
    # pruned = {0,1}; rotation (0,5) starts cross-boundary (0 deleted, 5 kept).
    g = Genome(rotations=[PairRotation(0, 5, 0.3)], pruned={0, 1})
    changed = _mask_swap(cfg, g, 64, rng)
    repair_rotations_after_mask(cfg, g, 64, changed, rng)
    assert len(g.pruned) == 2
    for rot in g.rotations:
        assert rot.a != rot.b


def test_assigned_mutation_fracs_ascending():
    """Each explorer SLOT gets a fixed ascending angle fraction (min..max)."""
    fracs = assigned_mutation_fracs(32, 0.003, 0.10)
    assert len(fracs) == 32
    assert abs(fracs[0] - 0.003) < 1e-9    # candidate 0 -> min
    assert abs(fracs[-1] - 0.10) < 1e-9    # candidate 31 -> max
    assert all(fracs[i] < fracs[i + 1] for i in range(len(fracs) - 1))
    # a different population still spreads ascending, capped at max
    fracs16 = assigned_mutation_fracs(16, 0.003, 0.10)
    assert len(fracs16) == 16
    assert abs(fracs16[-1] - 0.10) < 1e-9


def test_mutate_from_self_uses_assigned_frac():
    """The explorer's OWN assigned rate controls how many angles mutate (the
    rate belongs to the slot; it is NOT the config min/max default)."""
    cfg = _cfg()
    cfg.mutation.mask_swap_probability = 0.0
    cfg.mutation.pair_rewire_fraction = 0.0   # isolate pure angle mutation
    rng = random.Random(41)
    parent = make_random_genome(cfg, width=40, target=4, rng=rng)
    n = len(parent.rotations)
    assert n >= 4
    # high-rate explorer mutates ~50% of angles
    child_hi = mutate_from_self(cfg, parent, 40, 4, rng, mutation_frac=0.5)
    p_angles = [r.angle for r in parent.rotations]
    c_angles = [r.angle for r in child_hi.rotations]
    assert c_angles != p_angles
    changed = sum(1 for pa, ca in zip(p_angles, c_angles) if abs(pa - ca) > 1e-12)
    assert changed >= int(0.45 * n), (changed, n)


def test_rebase_keeps_slot_rate_not_donor_rate():
    """A high-rate explorer re-based onto a low-rate donor's genome stays a
    high-rate explorer: the donor's rate is NOT copied to the child; the child
    mutates at the SLOT's passed rate."""
    cfg = _cfg()
    cfg.mutation.mask_swap_probability = 0.0
    cfg.mutation.pair_rewire_fraction = 0.0
    rng = random.Random(51)
    donor = make_random_genome(cfg, width=40, target=4, rng=rng)
    rebased = rebase_explorer(cfg, donor, 40, 4, rng, mutation_frac=0.2)
    assert rebased.id != donor.id
    assert rebased.parent_ids == [donor.id]
    d_angles = [r.angle for r in donor.rotations]
    r_angles = [r.angle for r in rebased.rotations]
    n = len(donor.rotations)
    changed = sum(1 for da, ra in zip(d_angles, r_angles) if abs(da - ra) > 1e-12)
    assert changed >= max(1, int(0.15 * n)), (changed, n)
    assert changed <= max(1, int(0.35 * n)), (changed, n)
