# TrimReaper

Evolutionary MLP width-pruning for Qwen models — a proof-of-concept from
`docs/PLAN.md`.

**Hypothesis:** A genetic algorithm can discover sequences of pairwise Givens
rotations on Qwen MLP neurons that let *more complete hidden channels* be
removed at the same held-out output divergence (KL from the original model)
than an otherwise-identical GA that only chooses which channels to delete.

No fine-tuning and no gradients during search.

---

## Status / what exists now

This is the **project scaffold + a verified smoke test**. The full Qwen3-4B
proof-of-concept is implemented but has **not yet been run on the real model**
(see the GPU note below). The full 14-test suite and an end-to-end smoke demo
on a tiny gated-MLP model both pass.

### Repo layout

```
docs/PLAN.md                  the experiment plan this project implements
pyproject.toml                package metadata + `trimreaper` CLI
requirements.txt              pinned runtime deps
configs/poc.yaml              example config (optional)
src/trimreaper/
  config.py                   typed config + YAML/CLI overrides
  data.py                     WikiText streaming -> random batches + holdout
  model.py                    HF model load, MLP mask hooks, rotation apply/restore
  rotation.py                 pairwise Givens rotations (rows by R, cols by R^T)
  ga.py                       genome / mutation / crossover / elitism / fitness
  evaluate.py                 KL vs baseline, masked forward, holdout eval
  pipeline.py                 ratcheting anytime search + archive
  baselines.py                random / weight-norm / activation-magnitude
  sweep.py                    baseline curves + Pareto plotting
  compaction.py               physical deletion of channels for final genomes
  cli.py                      `trimreaper` subcommands
scripts/run_smoke.py          end-to-end smoke demo (tiny model, no GPU)
tests/                        unit + integration tests
```

---

## Install

```bash
python3 -m venv .venv
./.venv/bin/pip install -U pip
./.venv/bin/pip install -e . -r requirements.txt
```

For a CUDA build of torch on a machine with a GPU:
```bash
./.venv/bin/pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.14.1
```

---

## **Important: GPU availability in the dev harness**

The target run needs a GPU:

- Driver present here, but **no `/dev/nvidia*` device nodes** and
  `torch.cuda.is_available()` returns `False`. The GPU is not reachable from
  this session (a container/root-level device-node setup is required).
- Everything is written GPU-agnostic: if CUDA is available it is used,
  otherwise it falls back to CPU.

So: the **smoke test** runs on CPU with a tiny model; the **real Qwen3-4B run**
must be launched in an environment where the GPU is actually visible.

---

## Smoke test (fast, no GPU, no big download)

```bash
./.venv/bin/python scripts/run_smoke.py --out runs/smoke
```

Builds a tiny randomly-initialized gated-MLP Qwen3 model (real `Qwen3MLP`
classes, tiny dimensions), runs the ratcheting GA **with and without**
rotations, computes the one-shot baselines, and writes:
`runs/smoke/smoke_frontier.png`.

Run the test suite:

```bash
./.venv/bin/python -m pytest tests/ -q
```

---

## Real proof-of-concept run (needs the GPU)

```bash
# default: Qwen/Qwen3-4B-Base, layer 18, BF16, epsilon 0.01
./.venv/bin/python -m trimreaper.cli run

# smaller population / fewer generations for a quick check
./.venv/bin/python -m trimreaper.cli run --ga.population=16 --search.rounds=10

# override the epsilon constraint (PLAN section 10)
./.venv/bin/python -m trimreaper.cli run --search.epsilon=0.05

# just the baselines on a layer
./.venv/bin/python -m trimreaper.cli baselines --layers=18

# physically compact an archived genome and report metrics
./.venv/bin/python -m trimreaper.cli compact --layers=18

# (planned) epsilon sweep guidance
./.venv/bin/python -m trimreaper.cli sweep-eps
```

Any dotted `--group.key=value` override is supported.

---

## How it works (summary of the plan)

1. **Search-time pruning (mask simulation).** Channels are deleted by zeroing
   the hidden activation feeding `down_proj` via a forward pre-hook — no tensor
   resizing during search. Physical compaction happens only for final genomes
   (`compaction.py`).
2. **Genome.** A sparse list of pairwise Givens rotations `(a, b, angle)` plus
   the set of pruned channels. Rotations act on `gate_proj`/`up_proj` rows (by
   `R`) and `down_proj` columns (by `Rᵀ`).
3. **Fair fitness comparison.** Each generation draws ONE random batch; every
   candidate and the original-model baseline are evaluated on that same batch.
4. **Objective.** Maximize channels removed subject to `KL <= epsilon`; within a
   fixed removal count, lower KL wins; over-epsilon gets a strong penalty.
5. **Ratcheting/anytime.** Start at a small removal target, ratchet up on
   success, keep an archive of the best validated model per level — you can stop
   anytime and get the smallest validated model so far.
6. **Holdout validation.** New records are validated on a fixed holdout that
   never participates in evolution, guarding against lucky fitness batches.

---

## Notes for a fresh run on the real GPU

- Population `32–64`, sequence length `256`, fit batch `4–8` are good starting
  points (PLAN section 7–8). The 24GB-BF16 memory strategy holds one model +
  one MLP copy, not two full models.
- Run a small epsilon sweep (`0.001 / 0.01 / 0.05`) to see what each constraint
  looks like behaviorally before fixing a real epsilon.
- The decisive comparison is **GA-with-rotations vs GA-without-rotations** on
  the channels-removed vs KL Pareto frontier (PLAN section 13).
