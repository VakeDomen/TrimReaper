"""Tests for the population-diversity instrumentation.

Covers the pure GPU-free metrics in ``trimreaper.diversity`` plus genome
lineage-id assignment in ``ga``.
"""

import random

from trimreaper.config import Config, validate
from trimreaper.diversity import (
    elite_lineage_frac,
    jaccard_distance,
    mask_metrics,
    rot_jaccard,
    rotation_metrics,
)
from trimreaper.ga import (
    Genome,
    Individual,
    make_child_population,
    make_random_genome,
)
from trimreaper.rotation import PairRotation


def test_jaccard_distance():
    assert jaccard_distance([], []) == 0.0          # identical empty
    assert jaccard_distance([], [1, 2]) == 1.0      # disjoint
    assert jaccard_distance([1, 2], [2, 3]) == 1 - 1 / 3
    assert jaccard_distance([1, 2, 3], [1, 2, 3]) == 0.0  # identical


def _cfg():
    cfg = Config.defaults()
    cfg.ga.population = 8
    cfg.ga.elitism = 1
    cfg.ga.seed = 0
    cfg.ga.mutation_rate = 0.8
    cfg.ga.tournament_size = 2
    cfg.search.start_target = 6
    cfg.search.freeze_mask = False
    validate(cfg)
    return cfg


def _genome(pruned, rots=()):
    g = Genome()
    g.pruned = set(pruned)
    g.rotations = [PairRotation(*t) for t in rots]
    return g


def test_mask_metrics_identical_population():
    # all masks identical -> pairwise jaccard 0, unique_masks 1, dist-to-elite 0
    pop = [_genome([1, 2, 3]) for _ in range(5)]
    m = mask_metrics(pop, [1, 2, 3])
    assert m["unique_masks"] == 1
    assert m["mean_mask_jaccard"] == 0.0
    assert m["mean_dist_to_elite"] == 0.0
    assert m["mean_elite_overlap"] == 1.0
    # all >=90% overlap -> last histogram bin has all
    assert m["elite_overlap_hist"][4] == 5


def test_mask_metrics_diverse_population():
    pop = [
        _genome([1, 2, 3, 4, 5, 6]),
        _genome([10, 11, 12, 13, 14, 15]),
        _genome([1, 2, 3, 20, 21, 22]),
    ]
    m = mask_metrics(pop, [1, 2, 3, 4, 5, 6])
    assert m["unique_masks"] == 3
    assert m["mean_mask_jaccard"] > 0.5       # clearly distinct masks
    assert m["max_dist_to_elite"] == 1.0      # disjoint mask -> distance 1
    assert m["min_dist_to_elite"] == 0.0      # first == elite


def test_rotation_metrics():
    pop = [
        _genome([1, 2, 3], [(0, 10, 0.5), (20, 30, -0.2), (1, 40, 1.1)]),
        _genome([1, 2, 3], [(0, 10, 0.9), (50, 60, 0.0)]),   # shares (0,10), pruned-touch 1
        _genome([1, 2, 3], []),
    ]
    r = rotation_metrics(pop, [(0, 10, 0.5), (20, 30, -0.2), (1, 40, 1.1)])
    assert r["unique_pairs"] == 4          # (0,10),(20,30),(1,40),(50,60)
    assert r["mean_rotations"] == 5 / 3
    assert r["angle_std"] >= 0.0
    assert r["mean_pruned_channels_touched"] >= 0.0


def test_rot_jaccard_unordered_pairs():
    # (a,b) and (b,a) are the same pair
    assert rot_jaccard([(0, 10, 0.5)], [(10, 0, -1.0)]) == 0.0


def test_genome_lineage_ids_across_generations():
    cfg = _cfg()
    rng = random.Random(42)
    width = 64
    pop = [Individual(genome=make_random_genome(cfg, width, 6, rng)) for _ in range(8)]
    ids0 = {g.genome.id for g in pop}
    assert len(ids0) == 8 and all(i is not None for i in ids0)
    # elites carry the same id forward; children get fresh ids + parent links
    children = [Individual(genome=g) for g in make_child_population(cfg, pop, width, 6, rng)]
    cids = [g.genome.id for g in children]
    assert len(set(cids)) == len(cids)         # all children unique
    elites_id = pop[0].genome.id               # best-fitness elite kept verbatim
    assert any(g.genome.id == elites_id for g in children)
    for ind in children:
        if ind.genome.id != elites_id:
            assert len(ind.genome.parent_ids) == 2
            assert set(ind.genome.parent_ids) <= ids0


def test_lineage_frac_detects_elite_descent():
    # 2 generations of parent maps: current id 100 has parents {99, 98}, and
    # 99 descends from elite 50 two generations earlier -> elite_lineage_frac=1
    parent_maps = [
        {50: set(), 51: set()},          # oldest
        {99: {50}, 98: {50}},            # middle: 99 & 98 from elite 50
        {100: {99, 98}},                 # newest (below current)
    ]
    frac = elite_lineage_frac(parent_maps, [100], elite_id=50, lookback=3)
    assert frac == 1.0


def test_lineage_frac_unrelated():
    parent_maps = [
        {50: set(), 60: set()},
        {99: {60}},
        {100: {99}},
    ]
    frac = elite_lineage_frac(parent_maps, [100], elite_id=50, lookback=3)
    assert frac == 0.0


def test_emit_generation_compact_format():
    """The per-generation log must be 3 lines normally (4 with the periodic DBG
    line), with the reviewer's names: MASK dist=, elite_overlap=%, >90%=N, and
    ROT pruned=X/512 (Y%). Dropped per-gen: [min,max], angle_std, histogram."""
    import contextlib
    from io import StringIO

    from trimreaper.pipeline import _emit_generation

    div = {
        "mask": {"unique_masks": 32, "mean_mask_jaccard": 0.972,
                 "mean_elite_overlap": 0.082, "min_dist_to_elite": 0.0,
                 "max_dist_to_elite": 0.982},
        "rot": {"mean_rotations": 67, "unique_pairs": 598,
                "mean_unique_channels_touched": 130, "mean_pruned_channels_touched": 7,
                "mean_rot_distance_to_elite": 0.856, "angle_std": 3.66},
        "elite_overlap_hist": [31, 0, 0, 0, 1],
        "lineage_frac_elite": 0.031,
    }

    def emit(gen, debug_every=10):
        buf = StringIO()
        with contextlib.redirect_stdout(buf):
            _emit_generation(512, gen, 50, 0.00486, 0.00406, 512, div,
                             tag="GA+rot", prev_arch_kl=0.00435,
                             pop_size=32, debug_every=debug_every)
        raw = buf.getvalue()
        # every generation block is followed by one blank separator line
        assert raw.endswith("\n\n")
        return [l for l in raw.splitlines() if l.strip()]   # drop blank separators

    lines = emit(4)
    assert len(lines) == 3                       # no DBG line at gen 4
    # line 1: gen/total, fit, arch with improvement arrow
    assert lines[0].startswith("[GA+rot  t= 512  g=04/50]")
    assert "fit=0.00486" in lines[0]
    assert "arch=0.00406" in lines[0]
    assert "\u21930.00029" in lines[0]           # ↓ improvement marker
    # line 2: mask metrics, renamed 'dist='
    assert "MASK" in lines[1]
    assert "uniq=32/32" in lines[1]
    assert "dist=0.9720" in lines[1]
    assert "elite_overlap=8.2%" in lines[1]
    assert ">90%=1" in lines[1]
    # line 3: rotation coverage with fraction
    assert "ROT" in lines[2]
    assert "pairs=598" in lines[2]
    assert "chans=130.0" in lines[2]
    assert "pruned=7.0/512 (1.4%)" in lines[2]
    assert "lineage=3.1%" in lines[2]
    # dropped verbose diagnostics from per-gen lines
    assert "angle_std" not in lines[2]
    assert "overlap=" not in lines[2]

    # every-10th generation adds the DBG line
    lines10 = emit(10)
    assert len(lines10) == 4
    assert lines10[3].startswith("  DBG   overlap=[31,0,0,0,1]")
    assert "angle_std=3.6600" in lines10[3]
    assert "elite_dist=0.0000..0.9820" in lines10[3]

    # debug_every=0 disables the DBG line entirely
    assert len(emit(10, debug_every=0)) == 3

    # unchanged arch draws a dash, not an arrow
    buf = StringIO()
    with contextlib.redirect_stdout(buf):
        _emit_generation(512, 5, 50, 0.00486, 0.00406, 512, div, tag="GA+rot",
                         prev_arch_kl=0.00406, pop_size=32, debug_every=10)
    assert "\u2014" in buf.getvalue()            # '—'


def test_danger_styles_and_colored_output():
    """The danger helper must mark low diversity/coverage and high
    collapse-to-elite values red/yellow, and the rich-colored emitter must wrap
    the important numbers in valid rich markup (verified by forcing ANSI)."""
    import io
    from rich.console import Console

    from trimreaper import pipeline as P

    # _danger unit checks
    assert P._danger(0.97, low_thresholds=(0.4, 0.2)) == ""       # healthy
    assert P._danger(0.30, low_thresholds=(0.4, 0.2)) == "yellow"
    assert P._danger(0.10, low_thresholds=(0.4, 0.2)) == "red"
    assert P._danger(0.10, high_thresholds=(0.75, 0.9)) == ""      # low not risky high-side
    assert P._danger(0.85, high_thresholds=(0.75, 0.9)) == "yellow"
    assert P._danger(0.95, high_thresholds=(0.75, 0.9)) == "red"

    div_collapsed = {
        "mask": {"unique_masks": 3, "mean_mask_jaccard": 0.07,
                 "mean_elite_overlap": 0.97, "min_dist_to_elite": 0.0,
                 "max_dist_to_elite": 0.2},
        "rot": {"mean_rotations": 800, "unique_pairs": 300,
                "mean_unique_channels_touched": 400, "mean_pruned_channels_touched": 7,
                "mean_rot_distance_to_elite": 0.85, "angle_std": 1.3},
        "elite_overlap_hist": [0, 0, 2, 4, 26],
        "lineage_frac_elite": 0.94,
    }

    # Force an ANSI-capable console to confirm key numbers get color codes:
    # GA+rot (cyan), arch-improvement (bold green), MASK (blue), ROT (magenta),
    # collapsed mask dist (red), high elite overlap (red).
    buf = io.StringIO()
    c = Console(file=buf, force_terminal=True, color_system="truecolor",
                no_color=False, highlight=False, soft_wrap=True)
    old_console = P._console
    try:
        P._console = lambda: c
        P._emit_generation(512, 4, 50, 0.00263, 0.00404, 512, div_collapsed,
                           tag="GA+rot", prev_arch_kl=0.00435, pop_size=32, debug_every=10)
    finally:
        P._console = old_console
    out = buf.getvalue()
    # important numbers carry ANSI color codes
    assert "\x1b[" in out                      # at least one ANSI escape
    assert "\x1b[36m" in out                   # cyan tag (GA+rot)
    assert "\x1b[1;32m" in out                 # bold green (arch improvement)
    assert "\x1b[34m" in out                   # blue MASK
    assert "\x1b[35m" in out                   # magenta ROT
    assert "\x1b[31m" in out                   # red (collapsed mask dist/overlap)


def test_emit_legend_fenced_and_explains_terms():
    """The legend (printed once before gen 0) must be fenced by blank lines and
    explain the line-1 / MASK / ROT terms in plain text."""
    import contextlib
    from io import StringIO

    from trimreaper.pipeline import _emit_legend

    buf = StringIO()
    with contextlib.redirect_stdout(buf):
        _emit_legend(tag="GA+rot")
    raw = buf.getvalue()
    lines = raw.splitlines()
    # fenced: starts with a blank line, and the content ends before a blank
    assert lines[0] == ""
    assert "legend" in lines[1]
    assert "MASK" in raw
    assert "ROT" in raw
    assert "fit" in raw and "arch" in raw and "lineage" in raw
    assert "Jaccard" in raw
    # ends with a blank line before the next block
    assert raw.endswith("\n\n")
