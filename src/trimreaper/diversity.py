"""Population diversity instrumentation for the GA.

These are PURE, GPU-free metrics computed once per generation from the decoded
genomes.  They answer the reviewer's central question: is the population still
exploring, or has it collapsed onto the elite?

Primary scalar to plot alongside best validation KL:
    ``mean_mask_jaccard`` -- mean pairwise Jaccard DISTANCE between pruning masks
        (1 = all distinct, 0 = all identical). Falls to ~0 under premature
        convergence while fitness may keep improving.

Mask metrics (per generation):
    unique_masks, mean_mask_jaccard,
    mean/min/max distance-to-elite, mean_elite_overlap,
    elite-overlap histogram bins {0-25,25-50,50-75,75-90,90-100}%.

Rotation metrics (per generation):
    mean_rotations, unique_pairs, mean_unique_channels_touched/genome,
    mean_pruned_channels_touched/genome, mean_rot_distance_to_elite,
    angle_std.

Lineage (over the last ``lookback`` generations):
    frac_descended_from_elite -- what fraction of the current population
    traces its ancestry back to the current elite's genome id.
"""

from __future__ import annotations

import math
import statistics
from typing import Iterable, Optional, Sequence


def jaccard_distance(a: Iterable[int], b: Iterable[int]) -> float:
    """Jaccard distance between two channel sets: 1 - |A∩B| / |A∪B|.

    Empty-vs-empty is defined as 0 (identical); empty-vs-nonempty is 1.
    """
    A = set(a)
    B = set(b)
    inter = len(A & B)
    union = len(A | B)
    if union == 0:
        return 0.0
    return 1.0 - inter / union


def _genome_of(item):
    """Unwrap an Individual to its Genome (or pass through a Genome)."""
    return item.genome if hasattr(item, "genome") else item


def _pruned_set(item) -> set:
    return set(_genome_of(item).pruned)


def _rotations(item) -> list:
    return list(_genome_of(item).rotations)


def mask_distance_to(target: Iterable[int], masks: Sequence[Iterable[int]]) -> list[float]:
    """Jaccard distance from each mask to ``target``."""
    return [jaccard_distance(target, m) for m in masks]


def elite_overlap(mask: Iterable[int], elite: Iterable[int]) -> float:
    """Fraction of ``mask`` that overlaps the elite's deleted channels: |M∩E|/|M|.

    Defined 0 if mask is empty (nothing to share). With every mask at 512
    channels this reads directly as "shares X% of elite's deleted channels".
    """
    M = set(mask)
    E = set(elite)
    if not M:
        return 0.0
    return len(M & E) / len(M)


def mask_metrics(population, elite_mask: Iterable[int]) -> dict:
    """All pruning-mask diversity numbers for one generation.

    ``population`` is a sequence of objects exposing ``.pruned`` (or ``.genome``).
    """
    masks = [_pruned_set(g) for g in population]
    if not masks:
        return {
            "unique_masks": 0, "mean_mask_jaccard": float("nan"),
            "mean_dist_to_elite": float("nan"), "min_dist_to_elite": float("nan"),
            "max_dist_to_elite": float("nan"), "mean_elite_overlap": float("nan"),
            "elite_overlap_hist": [0, 0, 0, 0, 0],
        }

    unique_masks = len({frozenset(m) for m in masks})

    # mean pairwise Jaccard distance (the one scalar to plot)
    pairs = 0
    total = 0.0
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            total += jaccard_distance(masks[i], masks[j])
            pairs += 1
    mean_pair = total / pairs if pairs else 0.0

    d_elite = mask_distance_to(elite_mask, masks)
    overlaps = [elite_overlap(m, elite_mask) for m in masks]

    # histogram bins: 0-25 / 25-50 / 50-75 / 75-90 / 90-100 %
    bins = [0, 0, 0, 0, 0]
    for o in overlaps:
        pct = o * 100.0
        if pct < 25:
            bins[0] += 1
        elif pct < 50:
            bins[1] += 1
        elif pct < 75:
            bins[2] += 1
        elif pct < 90:
            bins[3] += 1
        else:
            bins[4] += 1

    return {
        "unique_masks": unique_masks,
        "mean_mask_jaccard": mean_pair,
        "mean_dist_to_elite": statistics.mean(d_elite),
        "min_dist_to_elite": min(d_elite),
        "max_dist_to_elite": max(d_elite),
        "mean_elite_overlap": statistics.mean(overlaps),
        "elite_overlap_hist": bins,
    }


def rotation_metrics(population, elite_rotations: Sequence) -> dict:
    """Rotation-structure diversity for one generation.

    ``population`` exposes ``.rotations`` (list of PairRotation) or ``.genome``.
    ``elite_rotations`` is the elite genome's rotation list. Two genomes can share
    a mask yet evolve entirely different rotation structures -- this catches that.
    """
    rot_lists = [_rotations(g) for g in population]
    if not rot_lists:
        return {
            "mean_rotations": 0, "unique_pairs": 0,
            "mean_unique_channels_touched": float("nan"),
            "mean_pruned_channels_touched": float("nan"),
            "mean_rot_distance_to_elite": float("nan"), "angle_std": float("nan"),
        }

    n_rot = [len(rl) for rl in rot_lists]

    def _theta(r) -> float:
        return float(r[2] if isinstance(r, tuple) else r.angle)

    unique_pairs: set[tuple[int, int]] = set()
    chan_counts: list[set[int]] = []
    pruned_counts: list[int] = []
    for g, rl in zip(population, rot_lists):
        pruned = _pruned_set(g)
        chans = set()
        n_p = 0
        for r in rl:
            a, b = _norm_rot(r)
            unique_pairs.add(_norm_rot(r))
            chans.add(a)
            chans.add(b)
            if a in pruned or b in pruned:
                n_p += 1
        chan_counts.append(chans)
        pruned_counts.append(n_p)

    # rotation-pair Jaccard distance to elite (as unordered pair sets)
    elite_pairs = {_norm_rot(r) for r in elite_rotations}

    def _pairset_dist(pset):
        if not elite_pairs and not pset:
            return 0.0
        union = len(elite_pairs | pset)
        if union == 0:
            return 0.0
        return 1.0 - len(elite_pairs & pset) / union

    d_elite = [_pairset_dist({_norm_rot(r) for r in rl}) for rl in rot_lists]

    # angle standard deviation across all rotations in the population
    angles = [_theta(r) for rl in rot_lists for r in rl]
    angle_std = statistics.pstdev(angles) if len(angles) > 1 else 0.0

    return {
        "mean_rotations": statistics.mean(n_rot),
        "unique_pairs": len(unique_pairs),
        "mean_unique_channels_touched": statistics.mean(len(c) for c in chan_counts),
        "mean_pruned_channels_touched": statistics.mean(pruned_counts),
        "mean_rot_distance_to_elite": statistics.mean(d_elite),
        "angle_std": angle_std,
    }


def _norm_rot(r) -> tuple[int, int]:
    if isinstance(r, tuple):
        return (int(r[0]), int(r[1])) if r[0] <= r[1] else (int(r[1]), int(r[0]))
    a, b = int(r.a), int(r.b)
    return (a, b) if a <= b else (b, a)


def rot_jaccard(a: Sequence, b: Sequence) -> float:
    """Jaccard distance between two rotation-pair sets (public helper)."""
    A = {_norm_rot(r) for r in a}
    B = {_norm_rot(r) for r in b}
    union = len(A | B)
    if union == 0:
        return 0.0
    return 1.0 - len(A & B) / union


def lineage_total(population, elite_id: Optional[int], lookback_generations: int) -> dict:
    """Assemble parent-link maps and compute elite-lineage fraction.

    ``population`` exposes ``.genome`` with ``id`` and ``parent_ids``.  Returns
    a dict of ``parent_map`` (child_id -> set(parent_ids)), usable as the
    ancestry table for the CURRENT generation, plus a helper to accumulate
    across generations.  The caller maintains the rolling window of recent
    generations' parent maps (GA lineage crosses only one generation back).
    """
    parent_map: dict[int, set[int]] = {}
    for ind in population:
        g = ind.genome
        if g.id is not None:
            parent_map[int(g.id)] = {int(p) for p in (g.parent_ids or ())}
    return {"parent_map": parent_map}


def elite_lineage_frac(
    parent_maps: list[dict[int, set[int]]],  # newest last
    current_ids: Sequence[int],
    elite_id: int,
    lookback: int,
) -> float:
    """Fraction of ``current_ids`` whose ancestry (within the last ``lookback``
    generations of ``parent_maps``, newest last) reaches ``elite_id``.

    A genome counts as descended if its own id equals the elite id (elites are
    carried forward verbatim) OR if a walk through its parent links -- across
    the up-to-``lookback`` most recent generations -- reaches the elite id.
    """
    if not current_ids:
        return float("nan")
    recent = parent_maps[-(lookback):]  # newest last = links of the current pop
    descended = 0
    for cid in current_ids:
        if int(cid) == elite_id:
            descended += 1
            continue
        visited = set()
        frontier = {int(cid)}
        found = False
        for m in reversed(recent):  # newest generation's links first
            new_frontier: set[int] = set()
            for node in frontier:
                if node in visited:
                    continue
                visited.add(node)
                parents = m.get(node)
                if not parents:
                    continue
                if elite_id in parents:
                    found = True
                    break
                new_frontier |= parents
            if found:
                break
            frontier = new_frontier
        if found:
            descended += 1
    return descended / len(current_ids)

