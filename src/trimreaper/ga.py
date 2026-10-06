"""Genetic algorithm: genome representation, operators, and selection.

Implements PLAN.md sections 6, 7, and 10.

Genome
------
    rotations: list of PairRotation (sparse, 16-64 by default)
    pruned:    set[int] of removed MLP channels (in the target layer)

Fitness objective (section 10) is constrained:
    maximize channels removed      subject to  mean KL <= epsilon
among candidates removing the same count, lower divergence wins; candidates
over epsilon receive a strong penalty and are ranked by that penalty.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from .config import Config
from .rotation import PairRotation, rotation_distance

# Hazard guard for pathological angle gaps.


_GENOME_ID = 0


def _next_genome_id() -> int:
    global _GENOME_ID
    _GENOME_ID += 1
    return _GENOME_ID


@dataclass
class Genome:
    """A candidate sparse-rotation + prune genome for one layer."""

    rotations: list[PairRotation] = field(default_factory=list)
    pruned: set[int] = field(default_factory=set)

    # lineage tracking (diversity instrumentation): a stable id per genome and
    # the ids of the two genomes it was bred from. Empty parent_ids for a
    # freshly-random-initialized genome.
    id: int = None  # type: ignore[assignment]
    parent_ids: list[int] = field(default_factory=list)

    # ---- construction helpers ------------------------------------------
    def with_target_pruned(self, target: int, rng: random.Random, width: int) -> "Genome":
        """Truncate/extend pruned set to exactly ``target`` channels."""
        target = min(target, width)
        self.pruned = set(list(self.pruned)[:target])
        for _ in range(target - len(self.pruned)):
            candidates = [c for c in range(width) if c not in self.pruned]
            if not candidates:
                break
            self.pruned.add(rng.choice(candidates))
        return self

    def num_removed(self) -> int:
        return len(self.pruned)

    def to_dict(self) -> dict:
        return {
            "rotations": [r.to_tuple() for r in self.rotations],
            "pruned": sorted(self.pruned),
        }


@dataclass
class Individual:
    """A genome plus its measured fitness for a single-eval context."""

    genome: Genome
    fitness: float = float("inf")   # consistent use: LOWER is better (KL or penalized)
    kl: float = float("inf")
    removed: int = 0                # channels removed (for objective bookkeeping)
    validated_kl: float = float("nan")  # holdout-validated KL
    fail_count: int = 0             # consecutive gens without beating the global elite
    # Per-explorer ASSIGNED mutation rate (fraction of rotation angles changed
    # each generation). Belongs to the SLOT, not the genome: it is assigned once
    # at population creation (ascending across candidates) and is preserved
    # across re-basing — a 20% explorer re-based onto a 5% explorer's genome
    # stays a 20% explorer. 0.0 = unassigned (unit-test / no-rotation paths).
    mutation_frac: float = 0.0


def assigned_mutation_fracs(population_size: int, frac_min: float,
                            frac_max: float) -> list[float]:
    """Ascending, fixed per-slot assigned angle-mutation fractions.

    candidate i (0-based) is assigned a fraction linearly spaced from
    ``frac_min`` to ``frac_max`` across the population, so the explorers span
    cautious->aggressive. With population 32, frac_min 0.003, frac_max 0.10
    this is ~0.3%..10%. The list is in slot order, so slot i keeps its rate for
    the WHOLE run (even across re-basing). The assigned rate controls ONLY
    angle mutation; pair/mask rates are the same for every explorer.
    """
    if population_size <= 0:
        return []
    if population_size == 1:
        return [frac_max]
    step = (frac_max - frac_min) / (population_size - 1)
    return [frac_min + i * step for i in range(population_size)]


class GARandom:
    """Bundled RNG sources, all seeded from cfg for reproducibility."""

    def __init__(self, cfg: Config):
        seed = cfg.ga.seed
        self.python = random.Random(seed)
        self.rng = self.python  # convenience alias


def random_rotation_pair(width: int, genome: Genome, cfg: Config, rng: random.Random) -> tuple[int, int]:
    """Choose a (a, b) channel pair.

    Applies the delete<->survive bias from PLAN.md section 6: with probability
    ``genome.delete_survive_bias`` one endpoint comes from the to-delete set
    and the other from the surviving (non-deleted) set. The bias is strong
    (0.95 default) so most rotations genuinely perturb the removed subspace
    rather than being kept<->kept.
    """
    surviving = [c for c in range(width) if c not in genome.pruned]
    deleted = list(genome.pruned)
    if rng.random() < cfg.genome.delete_survive_bias and deleted and surviving:
        a = rng.choice(deleted)
        b = rng.choice(surviving)
        return a, b
    a = rng.randrange(width)
    b = rng.randrange(width)
    if a == b:
        b = (b + 1) % width
    return a, b


def replace_endpoint(width: int, genome: Genome, cfg: Config, rng: random.Random,
                     keep: int) -> int:
    """Pick a replacement endpoint for a rotation, biased cross-boundary.

    When replacing one endpoint of an existing rotation, prefer (with
    ``delete_survive_bias``) to draw the new endpoint from the OPPOSITE side of
    the deleted/kept boundary relative to ``keep``, so the rotation keeps
    perturbing the deleted subspace. ``keep`` is the endpoint being fixed.
    """
    keep_deleted = keep in genome.pruned
    target_pool = ("surviving" if keep_deleted else "deleted")
    surviving = [c for c in range(width) if c not in genome.pruned]
    deleted = list(genome.pruned)
    if rng.random() < cfg.genome.delete_survive_bias:
        pool = surviving if target_pool == "surviving" else deleted
        if pool:
            return rng.choice(pool)
    c = rng.randrange(width)
    while c == keep:
        c = rng.randrange(width)
    return c


def random_angle(cfg: Config, rng: random.Random) -> float:
    return rng.uniform(cfg.genome.min_angle, cfg.genome.max_angle)


def rotation_budget(cfg: Config, target: int) -> int:
    """Number of rotations to use for a given deletion target.

    Scales with ``target`` (larger prunes get more rotational freedom) but is
    clamped to [min_rotations, max_rotations]. Formula:
        budget = clamp(int(rotations_per_removed * target), min, max).
    """
    budget = int(cfg.genome.rotations_per_removed * target)
    return max(cfg.genome.min_rotations, min(cfg.genome.max_rotations, budget))


def make_random_genome(cfg: Config, width: int, target: int, rng: random.Random,
                       fixed_pruned: Optional[set] = None) -> Genome:
    g = Genome(id=_next_genome_id())
    # Choose the pruning mask FIRST so delete<->survive-biased rotation pairs
    # can actually pick from known deleted + surviving channels (fix: the old
    # code generated rotations before pruned, so the 70% bias never fired).
    if fixed_pruned is not None:
        # rotation-isolation mode: the mask is frozen; only rotations vary.
        g.pruned = set(fixed_pruned)
    else:
        target = min(target, width)
        g.pruned = set(rng.sample(range(width), target))

    # FIXED rotation count: every genome has EXACTLY rotation_budget(target)
    # rotations (never added/removed by mutation), so genomes are directly
    # comparable and we don't evolve rotation-count complexity at the same time
    # as the solution.
    n_rot = rotation_budget(cfg, target) if target > 0 else 0
    for _ in range(n_rot):
        a, b = random_rotation_pair(width, g, cfg, rng)
        g.rotations.append(PairRotation(a, b, random_angle(cfg, rng)))
    return g


def clone_genome(genome: Genome, keep_id: bool = True) -> Genome:
    """Return an independent copy of a genome.

    ``keep_id=True`` preserves the lineage id/parent_ids (used when a candidate
    is crowned the global elite: the snapshot keeps the id it was born with so
    ancestry stays traceable). ``keep_id=False`` gives a fresh identity.
    """
    g = Genome(rotations=list(genome.rotations), pruned=set(genome.pruned))
    g.id = genome.id if keep_id and genome.id is not None else _next_genome_id()
    g.parent_ids = list(genome.parent_ids)
    return g


def _mask_swap(cfg: Config, genome: Genome, width: int, rng: random.Random) -> Optional[set]:
    """Swap pruned channels for kept ones (mask mutation).

    Each of ``mask_swap_count`` iterations independently fires with
    ``mask_swap_probability`` and swaps ONE pruned channel for ONE kept
    channel (an exact swap, so the prune count stays exactly `target`). This is
    "try deleting a different dimension".

    Returns the set of channel indices whose pruned/kept STATUS changed (the
    dropped + added channels), or an empty set when nothing changed. The caller
    uses this to repair rotations that now point at flipped channels.
    """
    changed: set = set()
    if not genome.pruned:
        return changed
    count = max(1, int(cfg.mutation.mask_swap_count))
    for _ in range(count):
        if rng.random() < cfg.mutation.mask_swap_probability:
            drop = rng.choice(list(genome.pruned))
            survivors = [c for c in range(width) if c not in genome.pruned]
            if not survivors:
                continue
            add = rng.choice(survivors)
            genome.pruned.remove(drop)
            genome.pruned.add(add)
            changed.update((drop, add))
    return changed


def repair_rotations_after_mask(cfg: Config, genome: Genome, width: int,
                                changed: set, rng: random.Random) -> None:
    """Repair rotations AFTER a mask mutation (reviewer fix #4).

    A mask swap changes which channels are deleted/kept, so a rotation that
    crossed the deleted/kept boundary before may now sit entirely on one side
    (or become degenerate a==b). For every rotation touching a changed-status
    channel, if both endpoints now share the same deleted-state, re-draw one
    endpoint (cross-boundary biased) to restore the rotation's coverage of the
    removed subspace. Also guarantees a != b on every rotation.
    """
    for i, rot in enumerate(genome.rotations):
        a, b = rot.a, rot.b
        if a == b:
            # degenerate pair (shouldn't normally happen); re-pick b
            genome.rotations[i] = PairRotation(a, replace_endpoint(width, genome, cfg, rng, keep=a), rot.angle)
            continue
        if (a in changed) or (b in changed):
            a_del = a in genome.pruned
            b_del = b in genome.pruned
            if a_del == b_del:
                # both endpoints now on the same side of the boundary -> restore
                # the cross-boundary intent by re-drawing the changed endpoint.
                if a in changed and b not in changed:
                    a = replace_endpoint(width, genome, cfg, rng, keep=b)
                else:
                    b = replace_endpoint(width, genome, cfg, rng, keep=a)
                if a == b:
                    continue  # keep degenerate resolution from a width-1 pool; skip
                genome.rotations[i] = PairRotation(a, b, rot.angle)


def _mutate_angles_guaranteed(cfg: Config, genome: Genome, rng: random.Random,
                              frac: Optional[float] = None) -> None:
    """Mutate ~``frac`` of rotation angles — "try a different rotation in the
    same plane". GUARANTEED to run (independent-explorer mode mutates 100% of
    the time); the per-explorer assigned ``frac`` controls HOW MANY angles.

    Each selected angle gets the small Gaussian change (``mutation.angle_std``),
    with an occasional large jump (``mutation.large_angle_probability`` /
    ``mutation.large_angle_std``) for exploration. With ``frac`` given it is the
    explorer's assigned rate; otherwise it falls back to the midpoint of
    ``[angle_fraction_min, angle_fraction_max]`` (used by unit tests / legacy
    callers that don't carry a per-slot rate).
    """
    n = len(genome.rotations)
    if n == 0:
        return
    if frac is None:
        frac = (cfg.mutation.angle_fraction_min + cfg.mutation.angle_fraction_max) / 2.0
    n_angle = max(1, min(n, int(round(frac * n))))
    for i in rng.sample(range(n), n_angle):
        rot = genome.rotations[i]
        if rng.random() < cfg.mutation.large_angle_probability:
            new_angle = rot.angle + rng.gauss(0, cfg.mutation.large_angle_std)
        else:
            new_angle = rot.angle + rng.gauss(0, cfg.mutation.angle_std)
        genome.rotations[i] = PairRotation(rot.a, rot.b, new_angle)


def _rewire_pairs(cfg: Config, genome: Genome, width: int, rng: random.Random) -> None:
    """Rewire a fixed fraction of rotation pairs — "try a different plane".

    Each selected rotation gets BOTH endpoints re-drawn as a fresh,
    delete<->survive-biased (pruned<->kept) pair; its angle is preserved. This
    is ONE operation (not separate A/B endpoint mutations): every new pair is
    guaranteed valid and cross-boundary, so no separate repair is needed.
    """
    n = len(genome.rotations)
    if n == 0:
        return
    n_rewire = max(1, min(n, int(round(cfg.mutation.pair_rewire_fraction * n))))
    for i in rng.sample(range(n), n_rewire):
        a, b = random_rotation_pair(width, genome, cfg, rng)
        rot = genome.rotations[i]
        genome.rotations[i] = PairRotation(a, b, rot.angle)


def _mutate_explorer_core(cfg: Config, genome: Genome, width: int, target: int,
                          rng: random.Random,
                          fixed_pruned: Optional[set] = None,
                          mutation_frac: Optional[float] = None) -> Genome:
    """Independent-explorer mutation engine.

    Order (mask FIRST, then repair, then pairs, then angles):
        1. copy the parent genome
        2. mutate the pruned mask (probabilistic exact swaps)
        3. REPAIR rotations against the new mask (fix lost cross-boundary intent)
        4. rewire rotation PAIRS (fixed fraction, always valid pruned<->kept)
        5. mutate ANGLES (GUARANTEED, ~``mutation_frac`` every generation)

    ``mutation_frac`` is the assigned per-explorer angle rate. It is a property
    of the EXPLORER SLOT and stays with the caller regardless of which genome is
    being copied. Pair and mask rates are the same for every explorer.

    In rotation-isolation mode (``fixed_pruned``), the mask is frozen and only
    rotations evolve (mask mutation is skipped entirely).
    """
    g = Genome(rotations=list(genome.rotations), pruned=set(genome.pruned))
    changed: set = set()
    if fixed_pruned is not None:
        g.pruned = set(fixed_pruned)
    else:
        changed = _mask_swap(cfg, g, width, rng)   # exact swap keeps `target` removed
    # repair rotations against any mask change (only needed when the mask moved).
    repair_rotations_after_mask(cfg, g, width, changed, rng)
    # rewire a fixed fraction of rotation pairs (try different planes).
    _rewire_pairs(cfg, g, width, rng)
    # guaranteed angle mutation (100% of the time) at the explorer's rate.
    _mutate_angles_guaranteed(cfg, g, rng, frac=mutation_frac)
    if fixed_pruned is not None:
        g.pruned = set(fixed_pruned)   # rotations evolve; mask stays frozen
    g.id = _next_genome_id()
    return g


def mutate_from_self(cfg: Config, genome: Genome, width: int, target: int,
                     rng: random.Random, fixed_pruned: Optional[set] = None,
                     mutation_frac: Optional[float] = None) -> Genome:
    """Independent-explorer evolution: a candidate mutates ONLY from itself.

    No crossover, no other parent. The child inherits both the rotations and
    the pruned mask of its sole self-parent, then the FULL explorer mutation
    (mask-first + repair + pair rewire + guaranteed angle mutation at the
    explorer's own assigned rate) applies. Returns the new genome with a fresh
    identity whose single parent is ``genome``.

    ``mutation_frac`` is the assigned per-explorer angle rate (defaults to the
    midpoint of the angle-fraction range when not supplied).
    """
    g = _mutate_explorer_core(cfg, genome, width, target, rng, fixed_pruned,
                              mutation_frac=mutation_frac)
    g.parent_ids = [genome.id] if genome.id is not None else []
    return g


def rebase_explorer(cfg: Config, base: Genome, width: int, target: int,
                    rng: random.Random, fixed_pruned: Optional[set] = None,
                    mutation_frac: Optional[float] = None) -> Genome:
    """Rebase a stuck explorer onto a tournament-selected base and mutate it.

    The fresh genome copies ``base`` then runs the SAME guaranteed explorer
    mutation (mask-first + repair + pair rewire + angle), giving the rescued
    candidate a new path. Used only when a candidate has failed
    ``cfg.ga.fail_limit`` consecutive generations without beating the global
    elite.

    ``mutation_frac`` is the ASSIGNED rate of the SLOT being rescued (NOT the
    base's): the tournament donor's rate is deliberately not copied.
    """
    g = _mutate_explorer_core(cfg, base, width, target, rng, fixed_pruned,
                              mutation_frac=mutation_frac)
    g.parent_ids = [base.id] if base.id is not None else []
    return g


def coverage_stats(genome: Genome, width: int | None = None) -> dict:
    """Rotation-coverage metrics for an archived genome.

    Used by the rotation-isolation experiment to check whether the chosen
    rotations actually touch the removed channels: if ``unique_pruned_touched``
    is far below ``n_rot``/``unique_channels_touched``, much of the rotational
    freedom is being wasted outside the deleted subspace.
    """
    touched = set()
    pruned = set(genome.pruned)
    cross = 0                     # rotations whose endpoints straddle the boundary
    for r in genome.rotations:
        if (r.a in pruned) != (r.b in pruned):
            cross += 1
        touched.add(r.a)
        touched.add(r.b)
    n_pruned = len(pruned)
    up_touched = len(touched & pruned)
    return {
        "n_rot": len(genome.rotations),
        "unique_channels_touched": len(touched),
        "unique_pruned_touched": up_touched,
        # effective rotation count = rotations that actually cross the deleted/kept
        # boundary and can therefore move mass out of the removed subspace.
        "effective_rotations": cross,
        "cross_boundary_rotations": cross,
        # fraction of the deleted set touched by at least one rotation (0..1).
        "deleted_channel_coverage": (up_touched / n_pruned) if n_pruned else 0.0,
    }


def fitness_value(cfg: Config, kl: float, removed: int) -> float:
    """Constrained objective (PLAN.md section 10).

    Maximize channels removed subject to KL <= epsilon. Within a fixed removal
    count, lower KL wins. Candidates over epsilon receive a strong penalty so
    they sort below all feasible same-count candidates.
    """
    penalty = 0.0
    if kl > cfg.search.epsilon:
        penalty = cfg.search.penalty_scale * (kl - cfg.search.epsilon)
    return kl + penalty


def tournament_parent(
    pop: list[Individual], tournament_size: int, rng: random.Random
) -> Individual:
    """Pick one parent via tournament selection (best of ``tournament_size``)."""
    k = min(tournament_size, len(pop))
    contenders = rng.sample(pop, k)
    return min(contenders, key=lambda ind: ind.fitness)
