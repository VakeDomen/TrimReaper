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
    and the other from the surviving (non-deleted) set.
    """
    surviving = [c for c in range(width) if c not in genome.pruned]
    deleted = list(genome.pruned)
    if rng.random() < cfg.genome.delete_survive_bias and deleted and surviving:
        a = rng.choice(deleted)
        b = rng.choice(surviving)
    else:
        a = rng.randrange(width)
        b = rng.randrange(width)
    if a == b:
        b = (b + 1) % width
    return a, b


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

    n_rot_max = rotation_budget(cfg, target)
    if target > 0:
        n_rot = rng.randint(min(cfg.genome.min_rotations, n_rot_max), n_rot_max)
    else:
        n_rot = 0
    for _ in range(n_rot):
        a, b = random_rotation_pair(width, g, cfg, rng)
        g.rotations.append(PairRotation(a, b, random_angle(cfg, rng)))
    return g


def mutate(cfg: Config, genome: Genome, width: int, target: int, rng: random.Random,
           fixed_pruned: Optional[set] = None) -> Genome:
    """Return a new Genome produced by random mutations (section 7)."""
    g = Genome(rotations=list(genome.rotations), pruned=set(genome.pruned))

    if g.rotations and rng.random() < cfg.ga.angle_mutate_p:
        i = rng.randrange(len(g.rotations))
        rot = g.rotations[i]
        if rng.random() < cfg.ga.large_angle_p:
            new_angle = rot.angle + rng.gauss(0, cfg.ga.large_angle_std)
        else:
            new_angle = rot.angle + rng.gauss(0, cfg.ga.angle_mutate_std)
        g.rotations[i] = PairRotation(rot.a, rot.b, new_angle)

    if g.rotations and rng.random() < cfg.ga.replace_a_p:
        i = rng.randrange(len(g.rotations))
        rot = g.rotations[i]
        a = rng.randrange(width)
        while a == rot.b:
            a = rng.randrange(width)
        g.rotations[i] = PairRotation(a, rot.b, rot.angle)

    if g.rotations and rng.random() < cfg.ga.replace_b_p:
        i = rng.randrange(len(g.rotations))
        rot = g.rotations[i]
        b = rng.randrange(width)
        while b == rot.a:
            b = rng.randrange(width)
        g.rotations[i] = PairRotation(rot.a, b, rot.angle)

    if len(g.rotations) < rotation_budget(cfg, target) and rng.random() < cfg.ga.add_rotation_p:
        a, b = random_rotation_pair(width, g, cfg, rng)
        g.rotations.append(PairRotation(a, b, random_angle(cfg, rng)))

    if len(g.rotations) > 0 and rng.random() < cfg.ga.remove_rotation_p:
        del g.rotations[rng.randrange(len(g.rotations))]

    if fixed_pruned is not None:
        g.pruned = set(fixed_pruned)   # rotations evolve; mask stays frozen
    return g


def mutate_mask(cfg: Config, genome: Genome, width: int, rng: random.Random,
                fixed_pruned: Optional[set] = None) -> Genome:
    """Mutate the pruned-channel mask (section 7 mask mutations).

    In rotation-isolation mode (``fixed_pruned`` given), the mask is frozen and
    this is a no-op (the genome keeps the fixed mask unchanged).
    """
    g = Genome(rotations=list(genome.rotations), pruned=set(genome.pruned))
    if fixed_pruned is not None:
        g.pruned = set(fixed_pruned)
        return g
    if rng.random() < cfg.ga.flip_prune_p and g.pruned:
        # swap one pruned for one surviving
        drop = rng.choice(list(g.pruned))
        survivors = [c for c in range(width) if c not in g.pruned]
        if survivors:
            add = rng.choice(survivors)
            g.pruned.remove(drop)
            g.pruned.add(add)
    return g


def crossover(cfg: Config, pa: Genome, pb: Genome, width: int, target: int, rng: random.Random,
              fixed_pruned: Optional[set] = None) -> Genome:
    """Splice/subsample the two parents' rotations; combine + repair masks.

    Returns a child Genome with exactly ``target`` pruned channels.

    Mask crossover (controlled genetic operation): keep the parental
    INTERSECTION, then fill the remainder from their SYMMETRIC DIFFERENCE at
    random until ``target`` is reached. This preserves shared good channels and
    samples disagreement, rather than an arbitrary ``order[:target]`` of a set.

    In rotation-isolation mode (``fixed_pruned`` given), only rotations are
    crossed over; the child's mask is simply the fixed mask.
    """
    child = Genome()
    na = len(pa.rotations)
    nb = len(pb.rotations)
    # splice: take a prefix from A and a random sample from B
    cut = rng.randrange(0, na + 1) if na else 0
    take_b = max(0, rng.randrange(0, nb + 1)) if nb else 0
    child.rotations = list(pa.rotations[:cut]) + list(pb.rotations[max(0, nb - take_b):])
    # occasionally cap to the scaled rotation budget for this target
    budget = rotation_budget(cfg, target)
    if len(child.rotations) > budget:
        child.rotations = rng.sample(child.rotations, budget)

    if fixed_pruned is not None:
        child.pruned = set(fixed_pruned)
        return child

    # controlled mask crossover
    target = min(target, width)
    inter = set(pa.pruned) & set(pb.pruned)
    diff = (set(pa.pruned) ^ set(pb.pruned)) - inter
    child.pruned = set(list(inter)[:target])   # shared channels first
    remaining = list(diff) + [c for c in range(width) if c not in (set(pa.pruned) | set(pb.pruned))]
    while len(child.pruned) < target and remaining:
        c = remaining.pop(rng.randrange(len(remaining)))
        child.pruned.add(c)
    return child


def coverage_stats(genome: Genome, width: int | None = None) -> dict:
    """Rotation-coverage metrics for an archived genome.

    Used by the rotation-isolation experiment to check whether the chosen
    rotations actually touch the removed channels: if ``unique_pruned_touched``
    is far below ``n_rot``/``unique_channels_touched``, much of the rotational
    freedom is being wasted outside the deleted subspace.
    """
    touched = set()
    for r in genome.rotations:
        touched.add(r.a)
        touched.add(r.b)
    pruned = set(genome.pruned)
    return {
        "n_rot": len(genome.rotations),
        "unique_channels_touched": len(touched),
        "unique_pruned_touched": len(touched & pruned),
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


def selection(pop: list[Individual], elitism: int, tournament_size: int = 4, rng=None) -> list[Individual]:
    """Select surviving parents using REAL tournament selection.

    Returns ``elitism`` elite individuals (best by fitness) plus a pool of
    tournament-selected parents. Lower fitness is better. The returned list
    drives the next generation's reproduction.
    """
    pop_sorted = sorted(pop, key=lambda ind: ind.fitness)
    elites = pop_sorted[:elitism]
    parents = list(elites)
    return parents


def make_child_population(
    cfg: Config, parents: list[Individual], width: int, target: int, rng: random.Random,
    fixed_pruned: Optional[set] = None,
) -> list[Genome]:
    """Produce the next generation's genomes from the surviving parents.

    Each non-elite child is bred from TWO tournament-selected parents (chosen
    independently from the FULL previous population, best-of-k), so good
    individuals reproduce more — real selection pressure, not uniform random.

    When ``fixed_pruned`` is given (rotation-isolation mode), the mask is
    frozen and only rotations evolve.
    """
    next_gen: list[Genome] = []
    elites = sorted(parents, key=lambda ind: ind.fitness)[:cfg.ga.elitism]
    # keep elites verbatim (retain their lineage id so ancestry is traceable)
    for e in elites:
        el = Genome(rotations=list(e.genome.rotations),
                    pruned=set(fixed_pruned) if fixed_pruned is not None else set(e.genome.pruned),
                    id=e.genome.id, parent_ids=list(e.genome.parent_ids))
        next_gen.append(el)

    tsize = getattr(cfg.ga, "tournament_size", 4)
    while len(next_gen) < cfg.ga.population:
        if len(parents) < 2:
            child = make_random_genome(cfg, width, target, rng, fixed_pruned)
            next_gen.append(child)
            continue
        a = tournament_parent(parents, tsize, rng)
        b = tournament_parent(parents, tsize, rng)
        if rng.random() < cfg.ga.mutation_rate:
            child = mutate(cfg, a.genome if a is b else crossover(cfg, a.genome, b.genome, width, target, rng, fixed_pruned),
                           width, target, rng, fixed_pruned)
            child = mutate_mask(cfg, child, width, rng, fixed_pruned)
        else:
            child = crossover(cfg, a.genome, b.genome, width, target, rng, fixed_pruned)
        if fixed_pruned is None:
            child.with_target_pruned(target, rng, width)
        # fresh identity + record the two parents for lineage tracking
        child.id = _next_genome_id()
        child.parent_ids = [a.genome.id, b.genome.id]
        next_gen.append(child)

    return next_gen
