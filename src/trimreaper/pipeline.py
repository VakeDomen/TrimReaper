"""Ratcheting evolutionary search (PLAN.md sections 11, 12, 13).

Core routine:
  - At each generation draw ONE random fitness batch; compute baseline logits
    from the pristine model once; evaluate every candidate against it on that
    same batch.
  - Take the top-K candidates by fitness and validate them on the fixed
    VALIDATION set (streamed, memory-safe). Archive decisions use VALIDATION
    KL ONLY -- never the single random fitness batch, whose KL is not
    comparable across generations.
  - Run G generations for the current deletion target.
  - On success (a validated candidate within epsilon) ratchet the deletion
    target upward (32 -> 64 -> 96 ...).
  - Maintains an anytime archive: best validated removal level found so far.
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
    evaluate_genome_on_set,
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
    val_batches: list[torch.Tensor],
    use_rotations: bool = True,
    progress=None,
    archive_dir: Optional[str] = None,
    variant: str = "",
) -> SearchResult:
    """Run the ratcheting GA for one layer with a given method.

    ``use_rotations=False`` implements the "GA selecting channels WITHOUT
    rotations" baseline (PLAN.md section 13): genomes carry only the mask.

    ``archive_dir`` overrides where per-record JSON archives are written
    (default: ``cfg.search.archive_dir``). ``variant`` is a short label
    (e.g. "rot" / "norot") embedded in the archive filenames so the two GA
    runs in a single ``run`` do not overwrite each other's records.
    """
    width = _intermediate_size(pm, layer)
    seq_len = cfg.data.seq_len
    fit_batch = cfg.data.fit_batch
    valid_best_k = max(1, cfg.search.valid_best_k)

    rng_wrap = GARandom(cfg)
    rng = rng_wrap.python

    result = SearchResult()
    archive: dict[int, ParetoPoint] = {}

    target = cfg.search.start_target
    # Cap the ratchet at the MLP width (never remove more channels than exist).
    max_target = cfg.search.max_target if cfg.search.max_target > 0 else width
    max_target = min(max_target, width)

    while target <= max_target:
        gen_limit = cfg.search.rounds if cfg.search.rounds > 0 else 10_000_000
        # --- build initial population for this target ---
        population: list[Individual] = []
        for _ in range(cfg.ga.population):
            g = make_random_genome(cfg, width, target, rng)
            if not use_rotations:
                g.rotations = []
            population.append(Individual(genome=g))

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

            # ---- stable objective: validate top-K on the VALIDATION set ----
            top = sorted(population, key=lambda i: i.fitness)[:valid_best_k]
            for ind in top:
                vkl = evaluate_genome_on_set(
                    pm, val_batches, layer, ind.genome, seq_len
                )
                ind.validated_kl = vkl
                cur = archive.get(target)
                if (cur is None or vkl < cur.validated_kl) and ind.removed > 0:
                    archived = ParetoPoint(
                        removed=ind.removed,
                        kl=ind.kl,
                        validated_kl=vkl,
                        genome=Genome(
                            rotations=list(ind.genome.rotations),
                            pruned=set(ind.genome.pruned),
                        ),
                    )
                    archive[target] = archived
                    result.points.append(archived)
                    result.best_genome = archived.genome
                    result.best_removed = max(result.best_removed, ind.removed)
                    _save_archive(cfg, layer, archived, archive_dir, variant)

            best = min(population, key=lambda i: i.fitness)
            if progress:
                av = archive.get(target)
                progress(target, gen, best.kl, best.removed,
                         arch_kl=av.validated_kl if av else float("nan"))

            # produce the next generation (tournament selection inside)
            population = [
                Individual(genome=g)
                for g in make_child_population(cfg, population, width, target, rng)
            ]
            if not use_rotations:
                for ind in population:
                    ind.genome.rotations = []

        result.n_generations += gen_limit
        # whether to advance the ratchet (validated record within epsilon)
        prev = archive.get(target)
        if prev is not None and prev.validated_kl <= cfg.search.epsilon:
            target += cfg.search.ratchet_step
        else:
            break  # fail to meet constraint at this level -> stop

    result.points = [archive[k] for k in sorted(archive)]
    return result


def _save_archive(cfg: Config, layer: int, point: ParetoPoint,
                  archive_dir: Optional[str] = None, variant: str = "") -> None:
    """Persist a new record to the archive dir as JSON."""
    import json

    try:
        arc_dir = archive_dir or cfg.search.archive_dir
        os.makedirs(arc_dir, exist_ok=True)
        prefix = f"{variant}_" if variant else ""
        path = os.path.join(arc_dir, f"{prefix}layer{layer}_removed{point.removed}.json")
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
