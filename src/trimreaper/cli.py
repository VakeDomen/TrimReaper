"""Command-line entry point for TrimReaper.

Subcommands:
  run         Run the first proof-of-concept: GA-with-rotations vs GA-without
              on one layer, producing Pareto points + a frontier plot.
  baselines   Compute the one-shot baseline curves (random / weight-norm /
              activation-magnitude) for a layer.
  extract     Four-way 'rotation extraction' comparison on a FIXED mask
              (A no-rot / B random-rot / C local-PCA / D GA-from-PCA).
  compact     Build the physically-compacted MLP for an archived genome and
              report metrics.
  sweep-eps   Print guidance + defaults for the epsilon sweep (0.001/0.01/0.05).

Every ``run`` writes all of its outputs to a NEW date-prefixed folder under
``search.archive_dir`` (e.g. ``runs/archive/20261005_192021_.../<tag>``), so
consecutive runs never overwrite each other.  Each run folder holds the
per-variant archive JSONs (rot_*/norot_*), the frontier plot
(``poa_frontier.png``), and a consolidated machine-readable
``run_results.json`` report.

All subcommands accept `--key=value` overrides (dotted, e.g. --ga.population=8).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import fields

from .config import Config, validate
from .model import load_pruned_model


def _add_override_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "overrides",
        nargs="*",
        help="key=value or --key=value overrides, e.g. ga.population=8",
    )


def _parse_overrides(items: list[str]) -> dict:
    out = {}
    for item in items:
        # accept both `--key=value` and `key=value`
        item = item[2:] if item.startswith("--") else item
        if "=" not in item:
            raise SystemExit(f"expected key=value, got: {item!r}")
        key, _, value = item.partition("=")
        out[key] = value
    return out


def _load_config(args) -> Config:
    cfg = Config.defaults()
    if getattr(args, "config", None) and os.path.exists(args.config):
        cfg = Config.from_yaml(args.config)
    overrides = {}
    for key, value in _parse_overrides(args.overrides).items():
        group, _, name = key.partition(".")
        if not name:
            continue
        grp = getattr(cfg, group, None)
        if grp is None or not hasattr(grp, name):
            raise SystemExit(f"unknown config key: {key}")
        # coerce to the field's declared type
        f = fields(grp)
        target_type = next((fld.type for fld in f if fld.name == name), None)
        cur = getattr(grp, name)
        if isinstance(cur, bool):
            setattr(grp, name, value.lower() in {"1", "true", "yes", "on"})
        elif isinstance(cur, int):
            setattr(grp, name, int(value))
        elif isinstance(cur, float):
            setattr(grp, name, float(value))
        elif isinstance(cur, list):
            import ast

            setattr(grp, name, ast.literal_eval(value))
        else:
            setattr(grp, name, value)
    # model.layers is a list; support comma-separated
    if getattr(args, "layers", None):
        cfg.model.layers = [int(x) for x in args.layers.split(",")]
    validate(cfg)
    return cfg


def _config_to_dict(cfg) -> dict:
    """Serialize the active config to a plain JSON-safe dict."""
    from dataclasses import asdict

    d = asdict(cfg)
    return d


def _make_run_dir(cfg, tag: str = "") -> str:
    """Create and return a date-prefixed run directory so consecutive runs
    never overwrite each other.

    Layout: ``<archive_dir>/<YYYYMMDD_HHMMSS_micro>[_tag]/``.  The microsecond
    component keeps runs started within the same second from colliding.
    """
    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    name = stamp if not tag else f"{stamp}_{tag}"
    run_dir = os.path.join(cfg.search.archive_dir, name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def cmd_run(args) -> int:
    import time

    def log(msg: str) -> None:
        # stable phase marker so you always know which stage you're in
        print(f"[run] {msg}", flush=True)

    cfg = _load_config(args)
    run_dir = _make_run_dir(cfg, getattr(args, "tag", "") or "")
    log(f"run outputs -> {run_dir}")
    t0_all = time.time()
    log(f"model_id={cfg.model.model_id} device={cfg.resolve_device()} layers={cfg.model.layers}")
    log(f"settings: rounds={cfg.search.rounds} target={cfg.search.start_target}->{cfg.search.max_target or 'width'}"
        f" step={cfg.search.ratchet_step} epsilon={cfg.search.epsilon} population={cfg.ga.population}")

    from .data import WikiTextStreamer
    from .evaluate import baseline_logits, evaluate_genome_on_set
    from .pipeline import ratchet_search
    from .sweep import plot_frontiers

    log("loading model (watching GPU util rise here)...")
    t0 = time.time()
    pm = load_pruned_model(cfg)
    log(f"model loaded in {time.time()-t0:.1f}s. layer={cfg.model.layers[0]}")

    layer = cfg.model.layers[0]
    streamer = WikiTextStreamer(cfg)
    cap = cfg.data.max_docs or "unlimited"
    log(f"tokenizing data (max_docs={cap}); CPU-bound, GPU stays idle until this finishes...")
    t0 = time.time()
    val_batches = streamer.holdout_batches()   # validation: used by search to accept records
    test_batches = streamer.test_batches()     # test: untouched, only for the final report
    log(f"data ready in {time.time()-t0:.1f}s ({len(val_batches)} val + {len(test_batches)} test batches)")

    def _progress(tag: str, eps: float):
        # The pipeline now emits the full compact per-generation block itself
        # (fit/arch + MASK + ROT + periodic DBG), including the "GA+rot" /
        # "GA-norot" tag derived from the variant label. This callback is kept
        # for API compatibility but prints nothing, avoiding a duplicate line.
        def cb(target, gen, best_kl, removed, arch_kl=float("nan"), diversity=None):
            return
        return cb

    log(f"phase 1/3: GA-with-rotations on layer {layer}...")
    t0 = time.time()
    res_rot = ratchet_search(
        pm, cfg, layer, streamer, val_batches, use_rotations=True,
        progress=_progress("GA+rot", cfg.search.epsilon),
        archive_dir=run_dir, variant="rot",
    )
    log(f"GA+rotations done in {time.time()-t0:.1f}s — {res_rot.n_generations} generations total")

    log(f"phase 2/3: GA-without-rotations (baseline) on layer {layer}...")
    t0 = time.time()
    res_norot = ratchet_search(
        pm, cfg, layer, streamer, val_batches, use_rotations=False,
        progress=_progress("GA-norot", cfg.search.epsilon),
        archive_dir=run_dir, variant="norot",
    )
    log(f"GA-no-rotations done in {time.time()-t0:.1f}s — {res_norot.n_generations} generations total")

    # Evaluate every recorded genome on the untouched TEST set (final report only).
    log(f"evaluating recorded genomes on the untouched test set ({len(test_batches)} batches)...")
    for res in (res_rot, res_norot):
        for p in res.points:
            if p.genome and p.genome.rotations is not None:
                p.test_kl = evaluate_genome_on_set(pm, test_batches, layer, p.genome, cfg.data.seq_len)
            else:
                p.test_kl = float("nan")

    ga_points = []
    for p in res_rot.points:
        ga_points.append((p.removed, p.validated_kl, "GA+rotations"))
    for p in res_norot.points:
        ga_points.append((p.removed, p.validated_kl, "GA-no-rotations"))

    log("phase 3/3: computing one-shot baselines + plotting frontier...")
    from .sweep import baseline_curves

    baseline_widths = [cfg.search.start_target, cfg.search.start_target + cfg.search.ratchet_step]
    baselines = baseline_curves(
        pm, cfg, layer, streamer,
        widths=baseline_widths,
        val_batches=val_batches,
        test_batches=test_batches,
    )
    out = os.path.join(run_dir, "poa_frontier.png")
    plot_frontiers(ga_points, baselines, out)

    log(f"frontier plot written to {out}")
    print("\n=== GA+rotations archive (val_kl = validation KL, test_kl = untouched test KL) ===", flush=True)
    for p in res_rot.points:
        print(f"  removed={p.removed:5d}  kl={p.kl:.5f}  val_kl={p.validated_kl:.5f}  test_kl={p.test_kl:.5f}")
    print("=== GA-no-rotations archive ===")
    for p in res_norot.points:
        print(f"  removed={p.removed:5d}  kl={p.kl:.5f}  val_kl={p.validated_kl:.5f}  test_kl={p.test_kl:.5f}")
    print("=== one-shot baselines (SAME validation set as GA) ===")
    for name, sets in baselines.items():
        for vname, pts in sets.items():
            if pts:
                line = ", ".join(f"{w}->{kl:.5f}" for w, kl in pts)
                print(f"  {name} [{vname}] {line}")

    # ---- persist a full machine-readable run report alongside the plot ----
    report_path = _write_run_report(run_dir, cfg, res_rot, res_norot, baselines)
    log(f"run results saved to {report_path}")
    print(f"\n[run] DONE in {time.time()-t0_all:.1f}s. run_dir={run_dir}")
    return 0


def _write_run_report(run_dir, cfg, res_rot, res_norot, baselines) -> str:
    """Serialize the full run results (both GA archives + baselines + config)
    to ``run_dir/run_results.json`` and return the path."""
    import time as _time

    def _pt(p):
        from .ga import coverage_stats

        return {
            "removed": p.removed,
            "kl": p.kl,
            "validated_kl": p.validated_kl,
            "test_kl": p.test_kl,
            "rotations": [r.to_tuple() for r in p.genome.rotations],
            "pruned": sorted(p.genome.pruned),
            **coverage_stats(p.genome),
        }

    report = {
        "run_dir": run_dir,
        "created": _time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": _config_to_dict(cfg),
        "ga_rotations": [_pt(p) for p in res_rot.points],
        "ga_no_rotations": [_pt(p) for p in res_norot.points],
        # Per-assigned-mutation-rate win statistics per variant (which explorer
        # rates produced elite/archive wins or got re-based).
        "rate_stats": {
            "rotations": getattr(res_rot, "rate_stats", {}),
            "no_rotations": getattr(res_norot, "rate_stats", {}),
        },
        "baselines": {
            name: {vname: [list(pt) for pt in pts] for vname, pts in sets.items()}
            for name, sets in baselines.items()
        },
    }
    report_path = os.path.join(run_dir, "run_results.json")
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)
    return report_path


def cmd_rotsearch(args) -> int:
    """Rotation-isolation experiment.

    Freeze the activation-magnitude pruning mask and optimize ONLY rotations
    around it, sweeping the rotation budget (rotations per removed channel).
    For each budget this asks: can basis rotation push a good fixed mask's KL
    below its no-rotation baseline?  Logs val/test KL plus rotation coverage
    (n_rot / unique channels touched / unique PRUNED channels touched).
    """
    import time

    def log(msg: str) -> None:
        print(f"[rotsearch] {msg}", flush=True)

    cfg = _load_config(args)
    run_dir = _make_run_dir(cfg, getattr(args, "tag", "") or "")
    log(f"run outputs -> {run_dir}")
    t0 = time.time()

    from .baselines import activation_magnitude_ranking
    from .data import WikiTextStreamer
    from .evaluate import evaluate_genome_on_set, evaluate_pruned_on_set
    from .ga import coverage_stats
    from .pipeline import ratchet_search

    pm = load_pruned_model(cfg)
    layer = cfg.model.layers[0]
    streamer = WikiTextStreamer(cfg)
    val_batches = streamer.holdout_batches()
    test_batches = streamer.test_batches()
    seq_len = cfg.data.seq_len
    target = cfg.search.start_target
    width = pm.pristine[layer]["up_proj"].shape[0]

    # 1) activation-magnitude mask (freeze): delete the target smallest |act| channels
    probe = [streamer.batch(cfg.data.fit_batch, seq_len) for _ in range(4)]
    amag = activation_magnitude_ranking(pm, layer, probe, seq_len)
    if target > len(amag):
        target = len(amag)
    frozen = sorted(amag[:target])
    log(f"activation-magnitude mask frozen: {target}/{width} channels removed")

    # 2) no-rotation baseline on the frozen mask (the number to beat)
    base_val = evaluate_pruned_on_set(pm, val_batches, layer, frozen, seq_len)
    base_test = evaluate_pruned_on_set(pm, test_batches, layer, frozen, seq_len)
    log(f"frozen-mask no-rotation baseline: val_kl={base_val:.5f} test_kl={base_test:.5f}")

    # 3) rotation-budget sweep around the frozen mask
    sweep = cfg.search.rotation_sweep or [cfg.genome.rotations_per_removed]
    budget_into_rot = cfg.genome.rotations_per_removed
    max_rot = cfg.genome.max_rotations
    min_floor = cfg.genome.min_rotations
    results = {"baseline": {"val": base_val, "test": base_test}}
    for rpr in sweep:
        cfg.search.freeze_mask = True
        cfg.search.frozen_pruned = frozen
        cfg.search.start_target = target
        cfg.search.max_target = target  # single fixed level
        cfg.genome.rotations_per_removed = rpr
        cfg.genome.max_rotations = max(1, int(rpr * target))
        # keep the evolvable-length floor (min_rotations default = 256), capped
        # so it never exceeds this budget's ceiling.
        cfg.genome.min_rotations = min(min_floor, cfg.genome.max_rotations)
        log(f"--- budget {rpr} rot/removed (max_rot={cfg.genome.max_rotations}, "
            f"rounds={cfg.search.rounds}) ---")

        res = ratchet_search(
            pm, cfg, layer, streamer, val_batches, use_rotations=True,
            progress=None, archive_dir=None, variant=f"rot{rpr}",
        )
        if res.best_genome is None:
            log(f"budget {rpr}: no archived genome")
            continue
        bg = res.best_genome
        test_kl = evaluate_genome_on_set(pm, test_batches, layer, bg, seq_len)
        cov = coverage_stats(bg)
        out = {
            "ratio": rpr,
            "n_rot": cov["n_rot"],
            "unique_channels_touched": cov["unique_channels_touched"],
            "unique_pruned_touched": cov["unique_pruned_touched"],
            "effective_rotations": cov["effective_rotations"],
            "cross_boundary_rotations": cov["cross_boundary_rotations"],
            "deleted_channel_coverage": round(cov["deleted_channel_coverage"], 4),
            "pruned_touch_frac": round(cov["unique_pruned_touched"] / max(1, target), 4),
            "val_kl": res.points[0].validated_kl if res.points else float("nan"),
            "test_kl": test_kl,
        }
        results[f"rot{rpr}"] = out
        log(f"budget {rpr}: val={out['val_kl']:.5f} test={out['test_kl']:.5f} "
            f"| n_rot={out['n_rot']} eff={out['effective_rotations']} "
            f"pruned-touched={out['unique_pruned_touched']}"
            f" cov={out['deleted_channel_coverage']*100:.0f}%")

    # restore config fields for the report
    cfg.genome.rotations_per_removed = budget_into_rot
    cfg.genome.max_rotations = max_rot
    cfg.genome.min_rotations = min_floor
    cfg.search.freeze_mask = False

    report = {
        "run_dir": run_dir,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target": target,
        "frozen_pruned": frozen,
        "results": results,
        "config": _config_to_dict(cfg),
    }
    import json

    report_path = os.path.join(run_dir, "rotsearch_results.json")
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)
    log(f"results saved -> {report_path}")
    log(f"DONE in {time.time()-t0:.1f}s")
    return 0


def cmd_extract(args) -> int:
    """Four-way 'rotation extraction' comparison on a FIXED mask.

    Uses the analytical pre-analysis (Wanda-style importance + local-PCA Givens)
    and compares, on the SAME fixed pruned mask:
      A. no rotations          (plain deleted channels)
      B. random rotations      (rotation_budget random Givens around the mask)
      C. local-PCA rotations   (the analytical greedy builder)
      D. GA-from-PCA           (GA seeded from C, full mutation)
    Reports validation + test KL for each, plus the analytical energy report
    (how much deleted-coordinate energy the PCA basis pushed into kept partners)
    and the GA's rotation coverage. All four share the identical mask.

    The fixed mask = the ``target`` weakest channels by importance (dead
    channels are INCLUDED automatically; ``analysis.skip_bottom`` may skip a
    few of the very weakest). ``data.seq_len`` and ``data.fit_batch`` size the
    probe calibration pass.
    """
    import time

    import torch

    def log(msg: str) -> None:
        print(f"[extract] {msg}", flush=True)

    cfg = _load_config(args)
    run_dir = _make_run_dir(cfg, getattr(args, "tag", "") or "")
    log(f"run outputs -> {run_dir}")
    t0 = time.time()

    from .analysis import channel_importance, local_pca_rotations, select_fixed_mask
    from .data import WikiTextStreamer
    from .evaluate import evaluate_genome_on_set, evaluate_pruned_on_set
    from .ga import GARandom, Genome, make_random_genome
    from .model import load_pruned_model
    from .pipeline import ratchet_search

    pm = load_pruned_model(cfg)
    layer = cfg.model.layers[0]
    streamer = WikiTextStreamer(cfg)
    val_batches = streamer.holdout_batches()
    test_batches = streamer.test_batches()
    seq_len = cfg.data.seq_len
    target = cfg.search.start_target
    mlp = pm.mlp_modules[layer]
    width = pm.pristine[layer]["up_proj"].shape[0]
    target = min(target, width)

    # ---- calibration pass: collect post-SwiGLU hidden over probe batches ----
    probe_batches = [streamer.batch(cfg.data.fit_batch, seq_len) for _ in range(4)]
    pm.restore_all()
    hidden: list[torch.Tensor] = []

    def pre_down(mod, args):
        hidden.append(args[0].detach().cpu().float().flatten(0, 1))

    handle = mlp.down_proj.register_forward_pre_hook(pre_down)
    try:
        with torch.inference_mode():
            for batch in probe_batches:
                _ = pm.model(input_ids=batch[:, :seq_len].to(pm.model.device), use_cache=False)
    finally:
        handle.remove()
    H = torch.cat(hidden, dim=0) if hidden else torch.empty(0, width)
    log(f"calibration hidden collected: {H.shape[0]} tokens")

    # ---- 1) importance + fixed mask ----
    score = channel_importance([H], pm.pristine[layer]["down_proj"])
    mask = select_fixed_mask(score, target, skip_bottom=cfg.analysis.skip_bottom)
    log(f"fixed mask: {target}/{width} weakest channels removed (skip_bottom={cfg.analysis.skip_bottom})")

    # ---- 2) analytical local-PCA rotations + energy report ----
    rots, report = local_pca_rotations(H, mask, max_rows=cfg.analysis.max_rows)
    log(f"local-PCA: {report['n_rot']} rotations, deleted-energy "
        f"before={report['deleted_before_energy']:.4f} after={report['deleted_after_energy']:.4f} "
        f"ratio={report['energy_ratio']:.4f}")

    # ---- 3) evaluate the four variants on the SAME mask ----
    rngwrap = GARandom(cfg)
    rng = rngwrap.python

    # A. no rotations
    klA_val = evaluate_pruned_on_set(pm, val_batches, layer, mask, seq_len)
    klA_test = evaluate_pruned_on_set(pm, test_batches, layer, mask, seq_len)
    log(f"A no-rotation: val={klA_val:.5f} test={klA_test:.5f}")

    # B. random rotations (same budget as the PCA gives)
    from .ga import rotation_budget
    n_rot = rotation_budget(cfg, target) if report["n_rot"] == 0 else report["n_rot"]
    rand_genome = make_random_genome(cfg, width, target, rng, fixed_pruned=set(mask))
    rand_genome.rotations = rand_genome.rotations[:n_rot]
    klB_val = evaluate_genome_on_set(pm, val_batches, layer, rand_genome, seq_len)
    klB_test = evaluate_genome_on_set(pm, test_batches, layer, rand_genome, seq_len)
    log(f"B random-rot: val={klB_val:.5f} test={klB_test:.5f} n_rot={len(rand_genome.rotations)}")

    # C. local-PCA rotations
    pca_genome = Genome(rotations=rots, pruned=set(mask))
    pca_genome.id = -1
    klC_val = evaluate_genome_on_set(pm, val_batches, layer, pca_genome, seq_len)
    klC_test = evaluate_genome_on_set(pm, test_batches, layer, pca_genome, seq_len)
    log(f"C local-PCA: val={klC_val:.5f} test={klC_test:.5f} n_rot={len(rots)}")

    # D. GA from PCA seed (freeze the mask, seed from the analytical solution)
    cfg.search.freeze_mask = True
    cfg.search.frozen_pruned = mask
    cfg.search.start_target = target
    cfg.search.max_target = target
    res = ratchet_search(
        pm, cfg, layer, streamer, val_batches, use_rotations=True,
        progress=None, archive_dir=run_dir, variant="ga_pca",
        seed_genome=pca_genome,
    )
    klD_val = res.points[0].validated_kl if res.points else float("nan")
    bg = res.best_genome
    klD_test = float("nan")
    if bg is not None:
        from .ga import coverage_stats
        cov = coverage_stats(bg)
        klD_test = evaluate_genome_on_set(pm, test_batches, layer, bg, seq_len)
        log(f"D GA-from-PCA: val={klD_val:.5f} test={klD_test:.5f} gens={res.n_generations} "
            f"n_rot={cov['n_rot']} pruned-touched={cov['unique_pruned_touched']}")
    else:
        log("D GA-from-PCA: no archived genome")

    # ---- 4) report ----
    cfg.search.freeze_mask = False
    results = {
        "target": target,
        "width": width,
        "mask": mask,
        "skip_bottom": cfg.analysis.skip_bottom,
        "mask_importance_rank": [int(i) for i in mask],
        "analytical": {
            "n_rot": report["n_rot"],
            "deleted_before_energy": report["deleted_before_energy"],
            "deleted_after_energy": report["deleted_after_energy"],
            "energy_ratio": report["energy_ratio"],
        },
        "no_rotation": {"val_kl": klA_val, "test_kl": klA_test},
        "random_rotation": {"val_kl": klB_val, "test_kl": klB_test, "n_rot": len(rand_genome.rotations)},
        "local_pca": {"val_kl": klC_val, "test_kl": klC_test, "n_rot": len(rots)},
        "ga_from_pca": {
            "val_kl": klD_val, "test_kl": klD_test, "generations": res.n_generations,
            "rate_stats": getattr(res, "rate_stats", {}),
        },
        "run_dir": run_dir,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": _config_to_dict(cfg),
    }
    report_path = os.path.join(run_dir, "extract_results.json")
    with open(report_path, "w") as fh:
        json.dump(results, fh, indent=2)
    log(f"results saved -> {report_path}")
    log(f"DONE in {time.time()-t0:.1f}s")
    return 0


def cmd_baselines(args) -> int:
    cfg = _load_config(args)
    from .data import WikiTextStreamer
    from .sweep import baseline_curves

    pm = load_pruned_model(cfg)
    layer = cfg.model.layers[0]
    streamer = WikiTextStreamer(cfg)
    val_batches = streamer.holdout_batches()
    test_batches = streamer.test_batches()
    curves = baseline_curves(pm, cfg, layer, streamer,
                             val_batches=val_batches, test_batches=test_batches)
    for name, sets in curves.items():
        for vname, pts in sets.items():
            print(name, vname, pts)
    return 0


def cmd_compact(args) -> int:
    cfg = _load_config(args)
    from .compaction import compact_model_metrics
    from .model import load_pruned_model

    pm = load_pruned_model(cfg)
    layer = cfg.model.layers[0]

    # load best archived genome if available
    import glob

    archive_files = sorted(glob.glob(os.path.join(cfg.search.archive_dir, f"layer{layer}_removed*.json")))
    if args.genome_json:
        with open(args.genome_json) as fh:
            data = json.load(fh)
        pruned = data.get("pruned", [])
        rotations = data.get("rotations", [])
    elif archive_files:
        with open(archive_files[-1]) as fh:
            data = json.load(fh)
        pruned = data.get("pruned", [])
        rotations = data.get("rotations", [])
    else:
        print("No archived genome found; run `run` first or pass --genome-json=FILE")
        return 1

    metrics = compact_model_metrics(pm, layer, pruned, rotations)
    print(json.dumps(metrics, indent=2))
    return 0


def cmd_sweep_eps(args) -> int:
    cfg = _load_config(args)
    print("Epsilon sweep guidance (PLAN.md section 10):")
    print("  epsilon is the max allowed held-out KL divergence.")
    print("  Suggested values from the plan: 0.001, 0.01, 0.05")
    print("Recommended: start with 0.01; plot the Pareto curve, then narrow.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="trimreaper", description="TrimReaper CLI")
    p.add_argument("--config", default=None, help="path to a YAML config")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run the proof-of-concept")
    r.add_argument("--layers", default=None, help="comma-separated layer indices")
    r.add_argument("--tag", default="", help="optional suffix for the dated run folder")
    _add_override_args(r)
    r.set_defaults(func=cmd_run)

    rs = sub.add_parser("rotsearch", help="rotation-isolation search: freeze the "
                                          "activation-magnitude mask, optimize rotations only")
    rs.add_argument("--layers", default=None, help="comma-separated layer indices")
    rs.add_argument("--tag", default="", help="optional suffix for the dated run folder")
    _add_override_args(rs)
    rs.set_defaults(func=cmd_rotsearch)

    b = sub.add_parser("baselines", help="compute one-shot baseline curves")
    b.add_argument("--layers", default=None, help="comma-separated layer indices")
    _add_override_args(b)
    b.set_defaults(func=cmd_baselines)

    ex = sub.add_parser("extract", help="four-way rotation-extraction comparison "
                                        "(A no-rot / B random / C local-PCA / D GA-from-PCA) on a fixed mask")
    ex.add_argument("--layers", default=None, help="comma-separated layer indices")
    ex.add_argument("--tag", default="", help="optional suffix for the dated run folder")
    _add_override_args(ex)
    ex.set_defaults(func=cmd_extract)

    c = sub.add_parser("compact", help="build compact MLP for an archived genome")
    c.add_argument("--layers", default=None, help="comma-separated layer indices")
    c.add_argument("--genome-json", default=None, help="path to a saved genome JSON")
    _add_override_args(c)
    c.set_defaults(func=cmd_compact)

    s = sub.add_parser("sweep-eps", help="epsilon sweep guidance")
    _add_override_args(s)
    s.set_defaults(func=cmd_sweep_eps)
    return p


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    # Convert override-style `--group.key=value` tokens into positional
    # `group.key=value` so argparse captures them as overrides.
    argv = [
        (t[2:] if (t.startswith("--") and "." in t.split("=", 1)[0]) else t)
        for t in argv
    ]
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
