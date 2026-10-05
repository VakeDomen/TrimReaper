"""Command-line entry point for TrimReaper.

Subcommands:
  run         Run the first proof-of-concept: GA-with-rotations vs GA-without
              on one layer, producing Pareto points + a frontier plot.
  baselines   Compute the one-shot baseline curves (random / weight-norm /
              activation-magnitude) for a layer.
  compact     Build the physically-compacted MLP for an archived genome and
              report metrics.
  sweep-eps   Print guidance + defaults for the epsilon sweep (0.001/0.01/0.05).

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


def cmd_run(args) -> int:
    import time

    def log(msg: str) -> None:
        # stable phase marker so you always know which stage you're in
        print(f"[run] {msg}", flush=True)

    cfg = _load_config(args)
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
        def cb(target, gen, best_kl, removed):
            print(f"[{tag}] target={target:5d} gen={gen:3d} best_kl={best_kl:.5f} removed={removed}  (eps={eps})", flush=True)
        return cb

    log(f"phase 1/3: GA-with-rotations on layer {layer}...")
    t0 = time.time()
    res_rot = ratchet_search(
        pm, cfg, layer, streamer, val_batches, use_rotations=True,
        progress=_progress("GA+rot", cfg.search.epsilon),
    )
    log(f"GA+rotations done in {time.time()-t0:.1f}s — {res_rot.n_generations} generations total")

    log(f"phase 2/3: GA-without-rotations (baseline) on layer {layer}...")
    t0 = time.time()
    res_norot = ratchet_search(
        pm, cfg, layer, streamer, val_batches, use_rotations=False,
        progress=_progress("GA-norot", cfg.search.epsilon),
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

    baselines = baseline_curves(
        pm, cfg, layer, streamer,
        widths=[cfg.search.start_target, cfg.search.start_target + cfg.search.ratchet_step],
    )
    out = os.path.join(cfg.search.archive_dir, "poa_frontier.png")
    plot_frontiers(ga_points, baselines, out)

    log(f"frontier plot written to {out}")
    print("\n=== GA+rotations archive (val_kl = validation KL, test_kl = untouched test KL) ===", flush=True)
    for p in res_rot.points:
        print(f"  removed={p.removed:5d}  kl={p.kl:.5f}  val_kl={p.validated_kl:.5f}  test_kl={p.test_kl:.5f}")
    print("=== GA-no-rotations archive ===")
    for p in res_norot.points:
        print(f"  removed={p.removed:5d}  kl={p.kl:.5f}  val_kl={p.validated_kl:.5f}  test_kl={p.test_kl:.5f}")
    for name, pts in baselines.items():
        print(f"=== baseline {name} ===")
        for w, kl in pts:
            print(f"  removed={w:5d}  kl={kl:.5f}")
    print(f"\n[run] DONE in {time.time()-t0_all:.1f}s. archive_dir={cfg.search.archive_dir}")
    return 0


def cmd_baselines(args) -> int:
    cfg = _load_config(args)
    from .data import WikiTextStreamer
    from .sweep import baseline_curves

    pm = load_pruned_model(cfg)
    layer = cfg.model.layers[0]
    streamer = WikiTextStreamer(cfg)
    curves = baseline_curves(pm, cfg, layer, streamer)
    for name, pts in curves.items():
        print(name, pts)
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
    _add_override_args(r)
    r.set_defaults(func=cmd_run)

    b = sub.add_parser("baselines", help="compute one-shot baseline curves")
    b.add_argument("--layers", default=None, help="comma-separated layer indices")
    _add_override_args(b)
    b.set_defaults(func=cmd_baselines)

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
