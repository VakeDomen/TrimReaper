"""Ratcheting evolutionary search (PLAN.md sections 11, 12, 13).

Core routine:
  - At each generation draw ONE random fitness batch; compute baseline logits
    from the pristine model once; evaluate every candidate against it on that
    same batch (fair comparisons).
  - Run G generations for the current deletion target.
  - When a candidate satisfies KL <= epsilon, validate on the fixed holdout;
    if it's a new record (or improves an existing one) accept and store it.
  - On success, ratchet the deletion target upward (32 -> 64 -> 96 ...).
  - Maintains an anytime archive: best validated removal level found so far.

Returns per-removal-level Pareto points: (channels_removed, kl) and the best
archive genomes.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Optional

import torch

from .config import Config
from .data import WikiTextStreamer
from .evaluate import (
    baseline_logits,
    evaluate_candidate,
    evaluate_candidate_holdout,
)
from .ga import (
    GARandom,
    Genome,
    Individual,
    fitness_value,
    make_child_population,
    make_random_genome,
)
from .model import PrunedModel


@dataclass
class ParetoPoint:
    removed: int
    kl: float
    validated_kl: float = float("nan")
    test_kl: float = float("nan")
    genome: Genome = field(default_factory=Genome)


@dataclass
class SearchResult:
    points: list[ParetoPoint] = field(default_factory=list)
    best_genome: Genome | None = None
    best_removed: int = 0
    n_generations: int = 0
    path: str = ""


def _intermediate_size(pm: PrunedModel, layer: int) -> int:
    return pm.pristine[layer]["up_proj"].shape[0]


def ratchet_search(
    pm: PrunedModel,
    cfg: Config,
    layer: int,
    streamer: WikiTextStreamer,
    holdout_batches: list[torch.Tensor],
    use_rotations: bool = True,
    progress=None,
) -> SearchResult:
    """Run the ratcheting GA for one layer with a given method.

    ``use_rotations=False`` implements the "GA selecting channels WITHOUT
    rotations" baseline (PLAN.md section 13): genomes carry only the mask.
    """
    width = _intermediate_size(pm, layer)
    seq_len = cfg.data.seq_len
    fit_batch = cfg.data.fit_batch

    rng_wrap = GARandom(cfg)
    rng = rng_wrap.python

    result = SearchResult()
    archive: dict[int, ParetoPoint] = {}

    target = cfg.search.start_target
    # Cap the ratchet at the MLP width (never remove more channels than exist).
    max_target = cfg.search.max_target if cfg.search.max_target > 0 else width
    max_target = min(max_target, width)

    while target <= max_target:
        # generations to run per ratchet target.
        # cfg.search.rounds > 0 -> bounded. If 0 (anytime) we still need a
        # finite per-target cap; use a large default (callers can set a real
        # budget via search.rounds on the CLI).
        gen_limit = cfg.search.rounds if cfg.search.rounds > 0 else 10_000_000
        # --- build initial population for this target ---
        population: list[Individual] = []
        for _ in range(cfg.ga.population):
            g = make_random_genome(cfg, width, target, rng)
            if not use_rotations:
                g.rotations = []
            population.append(Individual(genome=g))

        best_kl = float("inf")

        for gen in range(gen_limit):
            # NEW random batch every generation, same for all candidates.
            batch = streamer.batch(fit_batch, seq_len)
            ref = baseline_logits(pm, [batch], seq_len)[0]
            ref_safe = ref.clone() if ref is not None else None

            for ind in population:
                rots = ind.genome.rotations if use_rotations else []
                kl = evaluate_candidate(
                    pm, batch, ref_safe, layer, rots, sorted(ind.genome.pruned), seq_len
                ) if ref_safe is not None else float("inf")
                ind.kl = kl
                ind.removed = len(ind.genome.pruned)
                ind.fitness = fitness_value(cfg, kl, ind.removed)

            best = min(population, key=lambda i: i.fitness)
            best_kl = best.kl
            if progress:
                progress(target, gen, best.kl, best.removed)

            # Success driving the ratchet: a candidate is archived (below) whose
            # holdout-validated KL is within epsilon for this target.

            # holdout validate whenever the best candidate improves the KL
            # (prevents lucky fitness evals from being accepted, PLAN 12)
            cur_best_for_target = archive.get(target)
            if (cur_best_for_target is None
                    or best.kl < cur_best_for_target.kl) and best.removed > 0:
                holdout_kl = evaluate_candidate_holdout(
                    pm, holdout_batches, layer,
                    best.genome.rotations if use_rotations else [],
                    sorted(best.genome.pruned), seq_len,
                )
                if (cur_best_for_target is None
                        or holdout_kl < cur_best_for_target.validated_kl):
                    archived = ParetoPoint(
                        removed=best.removed,
                        kl=best.kl,
                        validated_kl=holdout_kl,
                        genome=Genome(
                            rotations=list(best.genome.rotations),
                            pruned=set(best.genome.pruned),
                        ),
                    )
                    archive[target] = archived
                    result.points.append(archived)
                    result.best_genome = archived.genome
                    result.best_removed = max(result.best_removed, best.removed)
                    _save_archive(cfg, layer, archived)

            # produce the next generation
            population = [
                Individual(genome=g)
                for g in make_child_population(cfg, population, width, target, rng)
            ]

        result.n_generations += gen_limit
        # whether to advance the ratchet
        prev = archive.get(target)
        if prev is not None and prev.validated_kl <= cfg.search.epsilon:
            # success -> ratchet up
            target += cfg.search.ratchet_step
        else:
            break  # fail to meet constraint at this level -> stop

    result.points = [archive[k] for k in sorted(archive)]
    return result


def _save_archive(cfg: Config, layer: int, point: ParetoPoint) -> None:
    """Persist a new record to the archive dir as JSON."""
    import json

    try:
        os.makedirs(cfg.search.archive_dir, exist_ok=True)
        path = os.path.join(cfg.search.archive_dir, f"layer{layer}_removed{point.removed}.json")
        data = {
            "layer": layer,
            "removed": point.removed,
            "kl": point.kl,
            "validated_kl": point.validated_kl,
            "rotations": [r.to_tuple() for r in point.genome.rotations],
            "pruned": sorted(point.genome.pruned),
        }
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)
    except Exception:
        pass
