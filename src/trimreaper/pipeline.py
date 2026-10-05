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
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import torch

from .config import Config
from .data import WikiTextStreamer
from .evaluate import (
    baseline_logits,
    evaluate_candidate,
)
from .ga import (
    GARandom,
    Genome,
    Individual,
    assigned_mutation_fracs,
    clone_genome,
    coverage_stats,
    fitness_value,
    make_random_genome,
    mutate_from_self,
    rebase_explorer,
    tournament_parent,
)
from .diversity import (
    elite_lineage_frac,
    mask_metrics,
    rotation_metrics,
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
    # Per-assigned-mutation-rate statistics for this whole search (one variant):
    #   rate (float) -> {"elite_wins", "archive_wins", "rebases"}.
    rate_stats: dict = field(default_factory=dict)


def _intermediate_size(pm: PrunedModel, layer: int) -> int:
    return pm.pristine[layer]["up_proj"].shape[0]


# previous archived validation KL per target level, for the arch ↓delta marker
_last_arch: dict[int, float] = {}


def _variant_tag(variant: str) -> str:
    """Derive a readable log tag from the ``variant`` label.

    "rot" -> "GA+rot", "norot" -> "GA-norot", "rot0.5" -> "rot0.5",
    "" -> "GA". Falls back to the variant verbatim when it contains '-'.
    """
    v = variant or ""
    if v == "rot":
        return "GA+rot"
    if v == "norot":
        return "GA-norot"
    if v == "":
        return "GA"
    if v.startswith("rot") and len(v) > 3:
        return v  # e.g. rot0.5, rot1.0 (budget label)
    return v or "GA"


def _fmt(v, nd=4):
    return f"{v:.{nd}f}" if v == v else "  n/a"


# ---- optional rich colorization of the per-generation log block ----
# Degrades gracefully to plain text when `rich` is unavailable (or a flag
# disables color). Color scheme follows the reviewer's spec:
#   GA+rot tag -> cyan | fit -> white | arch improved->bright green,
#   arch unchanged -> dim | MASK -> blue | ROT -> magenta |
#   dangerously low diversity/coverage -> yellow/red.
_COLOR = True              # set False at runtime to force plain output
_console_cache = None
_console_stdout = None
_MARKUP_TAG = re.compile(r"\[/?[a-zA-Z_]+(?: [^\]]*)?\]")


def _console():
    """Lazily build the rich Console bound to the CURRENT sys.stdout.

    Recreated if sys.stdout changes (e.g. under redirect_stdout in tests) so
    capture works; returns None when color is disabled or rich is unavailable.
    """
    import sys
    global _console_cache, _console_stdout
    cur = sys.stdout
    if _console_cache is not None and _console_stdout is cur:
        return _console_cache
    _console_cache = None
    _console_stdout = cur
    if _COLOR:
        try:
            from rich.console import Console
            _console_cache = Console(file=cur, highlight=False, soft_wrap=True)
        except Exception:
            _console_cache = None
    return _console_cache


def _emit_line(text: str) -> None:
    """Print one (markup-tagged) line, stripping tags when rich is off."""
    c = _console()
    if c is None:
        print(_MARKUP_TAG.sub("", text), flush=True)
    else:
        c.print(text)


def _c(text, style, cond=True):
    """Wrap ``text`` in a rich markup tag of ``style`` (no-op if not cond or
    style is empty -- avoids producing malformed ``[]text[/]`` tags)."""
    if not cond or not _COLOR or not style:
        return text
    return f"[{style}]{text}[/{style}]"


def _danger(v, low_thresholds=None, high_thresholds=None):
    """Return a rich style ("red"/"yellow"/"") for a value.

    ``low_thresholds=(yellow, red)`` marks LOW values risky: yellow when
    ``v < yellow``, red when ``v < red``.  ``high_thresholds=(yellow, red)``
    marks HIGH values risky: yellow when ``v > yellow``, red when ``v > red``.
    Both may be provided; the stricter of the two wins.  ``v`` nan -> "".
    """
    if v != v:
        return ""
    if low_thresholds:
        y, r = low_thresholds
        if v < r:
            return "red"
        if v < y:
            return "yellow"
    if high_thresholds:
        y, r = high_thresholds
        if v > r:
            return "red"
        if v > y:
            return "yellow"
    return ""


def _fmt_mut_line(winner_frac, rate_stats: dict) -> str:
    """Format the per-generation MUT line (assigned-mutation-rate tracking).

    When an explorer became the new elite this generation:
        MUT winner=17%  elite_wins=3  archive_wins=1
    (winner's cumulative counts). Otherwise:
        MUT winner=—  top_rates=17%:3  9%:2  24%:2
    (top-3 rates by elite_wins). Best-effort; never raises.
    """
    try:
        def _pct(f):
            return int(round(f * 100.0))
        if winner_frac is not None:
            k = round(winner_frac, 6)
            st = rate_stats.get(k, {})
            return (f"  {_c('MUT', 'bold')}  winner={_c(f'{_pct(winner_frac)}%', 'green')}"
                    f"  elite_wins={st.get('elite_wins', 0)}  archive_wins={st.get('archive_wins', 0)}")
        ranked = sorted(
            ((k, st.get("elite_wins", 0)) for k, st in rate_stats.items()),
            key=lambda kv: (-kv[1], kv[0]),
        )
        top = [f"{_pct(k)}%:{w}" for k, w in ranked[:3] if w > 0]
        if not top:
            return f"  {_c('MUT', 'bold')}  winner=—"
        return f"  {_c('MUT', 'bold')}  winner=—  top_rates={_c(' '.join(top), 'dim')}"
    except Exception:
        return ""


def _emit_generation(target, gen, total, best_kl, arch_kl, removed, div,
                     tag="GA", prev_arch_kl=None, pop_size=0, debug_every=10,
                     gen_time=0.0, mut=None):
    """Emit the reviewer's compact per-generation log block:
        1. is fitness/archive improving?
        2. is the pruning population collapsing?
        3. are rotations covering the channels we care about?
    plus an optional 4th DBG line every ``debug_every`` generations with the
    verbose diagnostics (overlap histogram / angle_std / elite dist range).
    Best-effort: every metric handled independently -- never raises.
    """
    try:
        m = div.get("mask", {})
        r = div.get("rot", {})
        hist = div.get("elite_overlap_hist", [])
        uniq = int(m.get("unique_masks", 0))
        pop = pop_size or uniq or 0

        # ---- line 1: fitness / archive ----
        tag_label = _c(tag, "cyan")
        fitv = _c(f"{best_kl:.5f}", "white")
        removed_s = _c(str(removed), "white")
        if arch_kl != arch_kl:
            a = ""                     # no archived record yet
        elif prev_arch_kl not in (None, float("nan")):
            d = prev_arch_kl - arch_kl  # positive = improvement
            if d > 1e-9:
                a = f"  arch={_c(f'{arch_kl:.5f} \u2193{d:.5f}', 'bold green')}"
            elif d < -1e-9:
                a = f"  arch={_c(f'{arch_kl:.5f} \u2191{-d:.5f}', 'red')}"
            else:
                a = f"  arch={_c(f'{arch_kl:.5f} \u2014', 'dim')}"
        else:
            a = f"  arch={_c(f'{arch_kl:.5f}', 'dim')}"
        gt_s = _c(f"{gen_time:.1f}s", "yellow") if gen_time > 0.0 else ""
        _emit_line(
            f"[{tag_label}  t={target:4d}  g={gen:02d}/{total}] fit={fitv}{a}  "
            f"removed={removed_s}{('  gen=%s' % gt_s) if gt_s else ''}"
        )

        # ---- line 2: mask diversity / collapse ----
        md = m.get("mean_mask_jaccard", float("nan"))
        eo = m.get("mean_elite_overlap", float("nan"))
        hi = hist[4] if len(hist) == 5 else None
        hi_frac = (hi / pop) if (isinstance(hi, int) and pop) else 1.0
        mdist_s = _c(_fmt(md), _danger(md, low_thresholds=(0.4, 0.2)))
        eo_s = _c(f"{100.0 * eo:.1f}%", _danger(eo, high_thresholds=(0.75, 0.9)))
        hi_s = _c(str(hi), _danger(hi_frac, high_thresholds=(0.75, 0.9))) if hi is not None else "n/a"
        _emit_line(
            f"  {_c('MASK', 'blue')}  uniq={uniq}/{pop}  dist={mdist_s}  "
            f"{_c('elite_overlap', 'bold')}={eo_s}  >90%={hi_s}"
        )

        # ---- line 3: rotation coverage ----
        pruned = r.get("mean_pruned_channels_touched", float("nan"))
        pfrac = f" ({100.0*pruned/removed:.1f}%)" if pruned == pruned and removed else ""
        pct = (100.0 * pruned / removed) if (pruned == pruned and removed) else float("nan")
        pruned_s = _c(_fmt(pruned, 1), _danger(pct / 100.0, low_thresholds=(0.15, 0.05)))
        eff = r.get("mean_effective_rotations", float("nan"))
        cov = r.get("mean_deleted_channel_coverage", float("nan"))   # 0..1
        cov_s = _c(f"{100.0*cov:.0f}%", _danger(cov, low_thresholds=(0.15, 0.05)))
        rd = r.get("mean_rot_distance_to_elite", float("nan"))
        lin = div.get("lineage_frac_elite", float("nan"))
        lin_s = _c(f"{100.0*lin:.1f}%", _danger(lin, high_thresholds=(0.75, 0.9)))
        _emit_line(
            f"  {_c('ROT', 'magenta')}   n={_fmt(r.get('mean_rotations'), 1)}  "
            f"pairs={r.get('unique_pairs','n/a')}  chans={_fmt(r.get('mean_unique_channels_touched'), 1)}  "
            f"{_c('pruned', 'bold')}={pruned_s}/{removed}{pfrac}  "
            f"{_c('cov', 'bold')}={cov_s}  {_c('eff', 'bold')}={_fmt(eff, 1)}  "
            f"dist={_fmt(rd)}  lineage={lin_s}"
        )

        # ---- optional line 4: verbose diagnostics (periodic) ----
        dbg = (debug_every > 0) and (gen % debug_every == 0)
        if dbg:
            h = ",".join(str(b) for b in hist) if len(hist) == 5 else "n/a"
            er = f"{_fmt(m.get('min_dist_to_elite'))}..{_fmt(m.get('max_dist_to_elite'))}"
            _emit_line(
                f"  {_c('DBG', 'dim')}   overlap=[{h}]  "
                f"{_c('angle_std', 'dim')}={_fmt(r.get('angle_std'))}  "
                f"{_c('elite_dist', 'dim')}={er}"
            )

        # ---- assigned-mutation-rate line (which explorer rates are winning) ----
        if mut:
            line = _fmt_mut_line(mut.get("winner"), mut.get("rate_stats", {}))
            if line:
                _emit_line(line)

        # blank line between generations (easier to scan a long run)
        _emit_line("")
    except Exception:
        pass


def _emit_legend(tag: str = "GA") -> None:
    """Print a one-time legend explaining the per-generation terms, colored to
    match the live output. Fenced by blank lines."""
    try:
        _emit_line("")
        _emit_line(f"{_c('legend', 'bold')}   one 3-line block per generation:")
        _emit_line(f"  {_c('fit', 'white')}       = best fitness (KL) this generation")
        _emit_line(f"  {_c('arch', 'bold green')}      = best held-out KL (\u2193 improved / \u2014 unchanged)")
        _emit_line(f"  {_c('removed', 'white')}    = channels pruned")
        _emit_line(
            f"  {_c('MASK', 'blue')}: {_c('uniq', 'bold')} unique masks / {_c('dist', 'bold')} "
            f"mean pairwise mask Jaccard distance ({_c('0=all identical', 'red')}) / "
            f"{_c('elite_overlap', 'bold')} % of mask shared with best / "
            f"{_c('>90%', 'bold')} # genomes \u226590% overlapping best"
        )
        _emit_line(
            f"  {_c('ROT', 'magenta')}: {_c('n', 'bold')} mean rotations / "
            f"{_c('pairs', 'bold')} unique rotation pairs / "
            f"{_c('chans', 'bold')} mean channels touched"
        )
        _emit_line(
            f"  {_c('pruned', 'bold')} X/512  unique deleted channels touched "
            f"({_c('low \u2192 red', 'red')} = rotations miss the deleted set)"
        )
        _emit_line(
            f"  {_c('cov', 'bold')}  deleted-channel coverage % "
            f"(want \u2265450-512/512, not {_c('13/512', 'red')}) / "
            f"{_c('eff', 'bold')} effective cross-boundary rotations / "
            f"{_c('dist', 'bold')} rotation Jaccard dist to best / "
            f"{_c('lineage', 'bold')} % descended from best"
        )
        _emit_line("")
    except Exception:
        pass


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

    # Rotation-isolation mode: a fixed pruning mask is frozen and the GA
    # optimizes ONLY rotations around it (see rotsearch CLI).
    fixed_pruned: Optional[set] = None
    if cfg.search.freeze_mask:
        fixed_pruned = set(cfg.search.frozen_pruned)
        fixed_pruned = {int(c) for c in fixed_pruned if 0 <= int(c) < width}
        if not fixed_pruned:
            raise ValueError("freeze_mask=True requires a non-empty search.frozen_pruned mask")

    rng_wrap = GARandom(cfg)
    rng = rng_wrap.python

    result = SearchResult()
    archive: dict[int, ParetoPoint] = {}

    target = cfg.search.start_target
    # Cap the ratchet at the MLP width (never remove more channels than exist).
    max_target = cfg.search.max_target if cfg.search.max_target > 0 else width
    max_target = min(max_target, width)

    # Per-assigned-mutation-rate statistics for this WHOLE variant (accumulate
    # across ratchet levels). Each explorer SLOT has a fixed rate (1%..32% by
    # default); we count how often each rate produced an elite/archive win or
    # got re-based. Keys are the assigned fracs (floats like 0.01..0.32).
    rate_max = getattr(cfg.ga, "mutation_rate_max", 0.32)
    rate_stats: dict[float, dict] = {}
    for f in assigned_mutation_fracs(cfg.ga.population, rate_max):
        rate_stats.setdefault(round(f, 6), {"elite_wins": 0, "archive_wins": 0, "rebases": 0})

    while target <= max_target:
        gen_limit = cfg.search.rounds if cfg.search.rounds > 0 else 10_000_000
        # --- build initial population for this target ---
        population: list[Individual] = []
        fracs = assigned_mutation_fracs(cfg.ga.population, rate_max)
        for i in range(cfg.ga.population):
            g = make_random_genome(cfg, width, target, rng, fixed_pruned)
            if not use_rotations:
                g.rotations = []
            population.append(Individual(genome=g, mutation_frac=fracs[i]))

        # rolling window of per-generation parent-link maps (newest last), for
        # the elite-lineage fraction over the last ``lineage_lookback`` gens.
        parent_maps: list[dict[int, set[int]]] = []

        # Independent-explorer mode (no crossover, no tournament every gen).
        # The population is NOT a breeding pool: each candidate is an explorer
        # that mutates ONLY from itself, generation after generation. A global
        # elite (snapshot of the best genome seen) is the comparison reference.
        # A candidate that goes `fail_limit` consecutive gens without beating
        # the elite gets re-based onto a tournament-selected base + mutated.
        global_elite: Optional[Individual] = None
        tsize = getattr(cfg.ga, "tournament_size", 4)
        fail_limit = getattr(cfg.ga, "fail_limit", 5)

        # print the legend once, before the first generation
        _emit_legend(tag=_variant_tag(variant))

        for gen in range(gen_limit):
            t_gen = time.time()      # for the per-generation time on the log line
            # NEW random batch every generation, same for all candidates.
            batch = streamer.batch(fit_batch, seq_len)

            # ---- evaluate every explorer's CURRENT genome on today's batch.
            # The global-elite reference (if any) is evaluated in the SAME call
            # so the elite-KL and each candidate's KL come from the identical
            # evaluator/rounding — a fair same-batch comparison (this is what
            # makes the fast and slow paths deterministic and defines "beats the
            # elite" without ambiguity).
            eval_genomes = [ind.genome for ind in population]
            have_elite = global_elite is not None
            if have_elite:
                eval_genomes = [global_elite.genome] + eval_genomes
            if cfg.search.use_fast_eval:
                # fast path: cache the pre-MLP prefix ONCE, then score ALL
                # genomes in ONE batched tail forward (no weight mutation).
                # build_cache() itself performs the single pristine full-model
                # forward and stores ref_logits, so we SKIP baseline_logits()
                # entirely (reviewer fix #5 — one forward per generation, not 2).
                from .fast_eval import build_cache, eval_genomes_batched

                cache = build_cache(pm, batch, layer, seq_len)
                ref_safe = None   # fast path compares against cache.ref_logits internally
                kls = eval_genomes_batched(
                    cache, pm, layer, eval_genomes, seq_len,
                    chunk=cfg.search.fast_eval_chunk,  # 0 = auto-size from free VRAM
                )
            else:
                # slow path: pristine baseline on this batch, then per-candidate
                # weight-rotation + masked forward.
                ref = baseline_logits(pm, [batch], seq_len)[0]
                ref_safe = ref.clone() if ref is not None else None
                kls = [
                    evaluate_candidate(
                        pm, batch, ref_safe, layer,
                        (g.rotations if use_rotations else []),
                        sorted(g.pruned), seq_len,
                    ) if ref_safe is not None else float("inf")
                    for g in eval_genomes
                ]

            if have_elite:
                elite_kl = kls[0]
                kls = kls[1:]
            else:
                elite_kl = float("inf")

            for ind, kl in zip(population, kls):
                ind.kl = kl
                ind.removed = len(ind.genome.pruned)
                ind.fitness = fitness_value(cfg, kl, ind.removed)

            # ---- fair same-batch comparison against the global elite ----
            # ``elite_winner`` records which explorer (by mutation_frac) became
            # the new global elite THIS generation, for the per-rate win stats
            # and the live MUT log line. None when the elite did not change.
            elite_winner: Optional[Individual] = None
            if global_elite is None:
                # first generation: crown the best founder as the elite, so
                # there is a reference to beat from gen 1 on. NOT counted as a
                # rate win: no mutation produced it (assigned-rate rule #6).
                best_ind = min(population, key=lambda i: i.fitness)
                global_elite = Individual(
                    genome=clone_genome(best_ind.genome, keep_id=True),
                    fitness=best_ind.fitness, kl=best_ind.kl, removed=best_ind.removed,
                    mutation_frac=best_ind.mutation_frac,
                )
                for ind in population:
                    ind.fail_count = 0
            else:
                # Fix #2: find ALL candidates that beat the old elite on this
                # same batch and reset their fail counters, then promote only
                # the SINGLE best of them. We must NOT let the last-beater
                # overwrite the elite, or a later (worse) winner could displace
                # an earlier (better) one.
                beaters = [ind for ind in population if ind.kl < elite_kl]
                for ind in population:
                    if ind.kl < elite_kl:
                        ind.fail_count = 0
                    else:
                        ind.fail_count += 1
                if beaters:
                    best_beater = min(beaters, key=lambda i: i.fitness)
                    global_elite = Individual(
                        genome=clone_genome(best_beater.genome, keep_id=True),
                        fitness=best_beater.fitness, kl=best_beater.kl, removed=best_beater.removed,
                        mutation_frac=best_beater.mutation_frac,
                    )
                    elite_winner = best_beater

            # Mutations that drove the wins: elite_wins are only counted after
            # generation 0 (a founder is not a mutation-driven win).
            if elite_winner is not None and gen >= 1:
                k = round(elite_winner.mutation_frac, 6)
                if k in rate_stats:
                    rate_stats[k]["elite_wins"] += 1

            # ---- stable objective: validate top-K on the VALIDATION set ----
            # NOTE: the archive decision MUST use ONE authoritative evaluator so
            # that the fast and slow search paths agree on WHICH genome is best.
            # The batched fast evaluator is the production/GPU validator; the
            # per-genome fitness KLs may differ from it at the ~1e-8 level, which
            # is fine for ranking within a generation but would flip archive
            # records that are validated by different evaluators across paths.
            top = sorted(population, key=lambda i: i.fitness)[:valid_best_k]
            if top:
                from .fast_eval import build_cache, eval_genomes_batched

                top_genomes = [ind.genome for ind in top]
                acc = [0.0] * len(top_genomes)
                n_b = 0
                for vb in val_batches:
                    cache = build_cache(pm, vb, layer, seq_len)
                    kls = eval_genomes_batched(cache, pm, layer, top_genomes, seq_len,
                                               chunk=cfg.search.fast_eval_chunk)  # 0 = auto
                    for i, k in enumerate(kls):
                        if k == k:  # not nan
                            acc[i] += k
                    n_b += 1
                for ind, a in zip(top, acc):
                    ind.validated_kl = (a / n_b) if n_b else float("nan")

            for ind in top:
                vkl = ind.validated_kl
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
                    # Assigned-rate win #6: a new best VALIDATION record is the
                    # strongest evidence a rate works. Skip generation 0 (the
                    # founder was random, not mutation-driven).
                    if gen >= 1:
                        k = round(ind.mutation_frac, 6)
                        if k in rate_stats:
                            rate_stats[k]["archive_wins"] += 1

            best = global_elite if global_elite is not None else min(population, key=lambda i: i.fitness)
            av = archive.get(target)

            # ---- compact per-generation log (reviewer's format) ----
            div = {}
            try:
                m = mask_metrics(population, best.genome.pruned)
                r = rotation_metrics(population, best.genome.rotations)
                lookback = max(2, getattr(cfg.search, "lineage_lookback", 5) or 1)
                elite_id = best.genome.id
                lf = float("nan")
                if elite_id is not None:
                    lf = elite_lineage_frac(
                        parent_maps, [ind.genome.id for ind in population],
                        int(elite_id), lookback,
                    )
                div = {
                    "mask": m,
                    "rot": r,
                    "elite_overlap_hist": m.get("elite_overlap_hist", []),
                    "unique_masks": m.get("unique_masks", 0),
                    "mean_mask_jaccard": m.get("mean_mask_jaccard", float("nan")),
                    "mean_dist_to_elite": m.get("mean_dist_to_elite", float("nan")),
                    "min_dist_to_elite": m.get("min_dist_to_elite", float("nan")),
                    "max_dist_to_elite": m.get("max_dist_to_elite", float("nan")),
                    "mean_elite_overlap": m.get("mean_elite_overlap", float("nan")),
                    "lineage_frac_elite": lf,
                    "parent_maps": parent_maps,
                }
                _prev = _last_arch.get(target)
                _emit_generation(
                    target, gen, gen_limit, best.kl, av.validated_kl if av else float("nan"),
                    best.removed, div, tag=_variant_tag(variant),
                    prev_arch_kl=_prev, pop_size=len(population),
                    debug_every=getattr(cfg.search, "diversity_debug_every", 10),
                    gen_time=(time.time() - t_gen),
                    mut={"winner": elite_winner.mutation_frac if elite_winner is not None else None,
                         "rate_stats": rate_stats},
                )
                _last_arch[target] = av.validated_kl if av else float("nan")
            except Exception:
                # diagnostics are best-effort; never break the search
                div = {}

            if progress:
                progress(target, gen, best.kl, best.removed,
                         arch_kl=av.validated_kl if av else float("nan"),
                         diversity=div)

            # ---- evolve each explorer ONCE for the next generation ----
            # Fix #1: decide ALL rebases BEFORE mutating anyone. Re-basing one
            # explorer mutates it and resets its fitness to inf in-place; if we
            # tournament-selected later, we'd draw from an already-half-mutated
            # population. So snapshot the evaluated population (frozen fitness +
            # pre-evolution genomes) and use THAT snapshot for every tournament,
            # then mutate/re-base everybody.
            snapshot = [
                Individual(genome=ind.genome, fitness=ind.fitness, kl=ind.kl,
                           removed=ind.removed, fail_count=ind.fail_count,
                           mutation_frac=ind.mutation_frac)
                for ind in population
            ]
            rebase_base_of: dict[int, Individual] = {}
            for i, ind in enumerate(population):
                if ind.fail_count >= fail_limit:
                    rebase_base_of[i] = tournament_parent(snapshot, tsize, rng)

            for i, ind in enumerate(population):
                # The explorer's OWN assigned rate stays put: re-basing copies a
                # donor genome but the slot keeps its rate (donor rate NOT copied).
                frac = ind.mutation_frac
                if i in rebase_base_of:
                    # a stuck explorer: copy the frozen-snapshot-selected base
                    # and mutate it to start a fresh search path.
                    base = rebase_base_of[i]
                    ind.genome = rebase_explorer(
                        cfg, base.genome, width, target, rng, fixed_pruned,
                        mutation_frac=frac)
                    if not use_rotations:
                        ind.genome.rotations = []
                    ind.fail_count = 0
                    ind.kl = float("inf")
                    ind.fitness = float("inf")
                    k = round(frac, 6)
                    if k in rate_stats:
                        rate_stats[k]["rebases"] += 1
                else:
                    ind.genome = mutate_from_self(
                        cfg, ind.genome, width, target, rng, fixed_pruned,
                        mutation_frac=frac)
                    if not use_rotations:
                        ind.genome.rotations = []
                    ind.kl = float("inf")
                    ind.fitness = float("inf")

            # record this generation's parent links (after evolution) for lineage
            parent_maps.append({})
            for ind in population:
                g = ind.genome
                if g.id is not None:
                    parent_maps[-1][int(g.id)] = {int(p) for p in (g.parent_ids or ())}
            # keep only the most recent `lineage_lookback` generations
            plb = max(2, getattr(cfg.search, "lineage_lookback", 5) or 1)
            if len(parent_maps) > plb:
                parent_maps = parent_maps[-plb:]

        result.n_generations += gen_limit
        # whether to advance the ratchet (validated record within epsilon)
        prev = archive.get(target)
        if prev is not None and prev.validated_kl <= cfg.search.epsilon:
            target += cfg.search.ratchet_step
        else:
            break  # fail to meet constraint at this level -> stop

    result.points = [archive[k] for k in sorted(archive)]
    result.rate_stats = {round(k, 6): dict(v) for k, v in rate_stats.items()}
    _emit_mutation_rate_table(variant, result.rate_stats)
    return result


def _emit_mutation_rate_table(variant: str, rate_stats: dict) -> None:
    """Print the end-of-run mutation-rate results table for one variant.

        mutation rate results (GA+rot)
         rate   elite wins   archive wins   rebases
          1%       0             0            3
          2%       1             0            2
           ...
         17%       6             3            1
           ...
         32%       1             0            4

    Best-effort; never raises. The same data is also written to
    ``run_results.json`` via ``SearchResult.rate_stats``.
    """
    try:
        if not rate_stats:
            return
        _emit_line("")
        _emit_line(f"{_c('mutation rate results', 'bold')} ({_c(_variant_tag(variant), 'cyan')}):")
        _emit_line(f"  {_c('rate', 'bold'):>5}  {_c('elite wins', 'bold'):>11}  "
                   f"{_c('archive wins', 'bold'):>12}  {_c('rebases', 'bold'):>8}")
        for k in sorted(rate_stats):
            st = rate_stats[k]
            _emit_line(
                f"  {int(round(k*100)):>3}%  {st.get('elite_wins', 0):>11}  "
                f"{st.get('archive_wins', 0):>12}  {st.get('rebases', 0):>8}"
            )
        _emit_line("")
    except Exception:
        pass


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
            **coverage_stats(point.genome),
        }
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)
    except Exception:
        pass
