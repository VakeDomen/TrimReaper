"""Configuration handling for TrimReaper.

Loads a default config, overrides from a YAML file, then from explicit
kwargs (CLI). Everything is a plain dataclass so fields are typed and
discoverable. Booleans from CLI use simple truthy parsing.
"""

from __future__ import annotations

import ast
import dataclasses
import os
from dataclasses import dataclass, field, fields
from typing import Any

import yaml


def _parse_scalar(value: str) -> Any:
    """Parse a CLI string into an int/float/bool/list when possible."""
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


@dataclass
class DataConfig:
    """Fitness / holdout data configuration."""

    dataset: str = "Salesforce/wikitext"
    dataset_config: str = "wikitext-103-raw-v1"
    split: str = "train"
    seq_len: int = 256
    fit_batch: int = 4          # sequences per forward for fitness
    holdout_seq: int = 128      # sequences for the fixed VALIDATION set (search-time)
    test_seq: int = 64          # sequences for the untouched TEST set (final report)
    num_workers: int = 2
    seed: int = 0
    max_docs: int = 0           # cap on tokenized docs to load (0 = unlimited)


@dataclass
class ModelConfig:
    """Model loading configuration."""

    model_id: str = "Qwen/Qwen3-4B-Base"
    dtype: str = "bfloat16"
    device: str = ""            # "" -> auto pick cuda if available else cpu
    low_cpu_mem_usage: bool = True
    layers: list[int] = field(default_factory=lambda: [18])


@dataclass
class GenomeConfig:
    """Genome representation settings."""

    min_rotations: int = 256
    max_rotations: int = 4096
    # Rotation budget scales with the deletion target so larger prunes get more
    # rotational freedom instead of a fixed cap: budget =
    # clamp(int(rotations_per_removed * target), min_rotations, max_rotations).
    # Budget ladder for 512 deleted channels (2^x * deleted): 1x=512, 2x=1024,
    # 4x=2048, 8x=4096 (see the rotsearch rotation_sweep).
    rotations_per_removed: float = 1.0
    # Initial genomes start with a substantial rotation count rather than the
    # bare floor: n_rot ~ uniform in [max(min_rot, ceil(initial_frac*budget)),
    # budget], so early evolution isn't artificially sparse.
    initial_rotation_frac: float = 0.75
    min_angle: float = -3.141592653589793 * 2
    max_angle: float = 3.141592653589793 * 2
    # Cross-boundary preference: with this probability, rotation pairs are drawn
    # deleted<->surviving (one pruned endpoint + one kept endpoint), so rotations
    # actually perturb the removed subspace rather than mostly kept<->kept
    # (per PLAN.md section 6). Strongly biased (0.95) per the rotation-coverage
    # finding that only ~13/512 deleted channels were being touched.
    delete_survive_bias: float = 0.95


@dataclass
class GaConfig:
    """Genetic algorithm hyperparameters."""

    population: int = 32
    elitism: int = 2
    tournament_size: int = 4       # tournament selection: best of k parents
    # Independent-explorer mode (no crossover): a candidate that goes this many
    # consecutive generations without beating the global elite (same-batch KL)
    # is re-based onto a tournament-selected base and starts a fresh path.
    fail_limit: int = 5
    # LEGACY-ONLY (used only by the retired crossover make_child_population, still
    # exercised by unit tests). The independent-explorer path ignores it: mutation
    # there happens 100% of the time via angle_mutate_frac.
    mutation_rate: float = 0.3
    add_rotation_p: float = 0.15
    remove_rotation_p: float = 0.10
    # Independent-explorer mode: the fraction of rotation angles mutated EVERY
    # generation (guaranteed, 100% of the time). Used as the fallback when no
    # per-explorer mutation_frac is supplied (e.g. unit tests calling
    # mutate_from_self directly). Active search overrides this with the
    # per-explorer assigned rate (1%..32%).
    angle_mutate_frac: float = 0.05
    # Per-explorer assigned mutation rate. Each candidate slot gets a FIXED,
    # ascending rate that does NOT change over the run:
    #   candidate i -> (i + 1) / population * mutation_rate_max
    # With population=32 and rate_max=0.32 this gives 1%, 2%, ..., 32% — a
    # spread from cautious to aggressive. The rate controls how many rotation
    # angles that explorer mutates per generation, and it survives re-basing
    # (the rate belongs to the slot, not the genome it copies).
    mutation_rate_max: float = 0.32
    angle_mutate_p: float = 0.5         # legacy-only (single-angle gate for `mutate`)
    angle_mutate_std: float = 0.15      # small Gaussian angle mutations
    large_angle_p: float = 0.05         # occasional large angle mutation
    large_angle_std: float = 1.0
    replace_a_p: float = 0.2
    replace_b_p: float = 0.2
    flip_prune_p: float = 0.2
    seed: int = 0


@dataclass
class SearchConfig:
    """Ratcheting search / objective settings."""

    epsilon: float = 0.01           # max allowed held-out KL divergence
    rounds: int = 60                # generations run per ratchet target (0=anytime/run-forever)
    valid_best_k: int = 3           # top-K candidates (by fitness) validated per generation
    start_target: int = 32          # channels to remove to start ratchet
    ratchet_step: int = 32          # increment on success
    max_target: int = 0             # 0 = unlimited
    penalty_scale: float = 10.0     # strong penalty for exceeding epsilon
    archive_dir: str = "runs/archive"
    use_fast_eval: bool = True      # cache the pre-MLP prefix + run one batched tail
    fast_eval_chunk: int = 0        # tails per tail-forward; 0 = AUTO-size from free VRAM (safe)
    # Rotation-isolation experiment: freeze a fixed pruning mask and let the GA
    # optimize ONLY rotations around it.
    freeze_mask: bool = False       # when True, the pruned mask is fixed
    frozen_pruned: list[int] = field(default_factory=list)  # the fixed mask (channel indices)
    # rotation-budget sweep (rotations per removed channel) for rotsearch, e.g.
    # [0.5, 1.0, 2.0, 4.0]. Empty = run a single budget from rotations_per_removed.
    rotation_sweep: list = field(default_factory=list)
    # Diversity instrumentation: how many prior generations of parent links to
    # walk when computing the "fraction descended from current elite" metric.
    lineage_lookback: int = 5
    # Emit the verbose per-generation DBG line (overlap histogram / angle_std
    # / elite dist range) every N generations (0 = never).
    diversity_debug_every: int = 10


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    genome: GenomeConfig = field(default_factory=GenomeConfig)
    ga: GaConfig = field(default_factory=GaConfig)
    search: SearchConfig = field(default_factory=SearchConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        cfg = cls()
        for group in fields(cls):
            sub = d.get(group.name, {})
            if isinstance(sub, dict):
                current = getattr(cfg, group.name)
                for f in fields(current):
                    if f.name in sub:
                        setattr(current, f.name, sub[f.name])
        return cfg

    @classmethod
    def defaults(cls) -> "Config":
        return cls()

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r") as fh:
            d = yaml.safe_load(fh) or {}
        return cls.from_dict(d)

    def override(self, overrides: dict[str, Any]) -> "Config":
        """Apply flat dotted overrides like {'data.seq_len': 128, 'ga.population': 64}."""
        for key, value in overrides.items():
            group, _, name = key.partition(".")
            if not name:
                continue
            grp = getattr(self, group, None)
            if grp is None:
                continue
            if not hasattr(grp, name):
                raise KeyError(f"Unknown config key: {key}")
            setattr(grp, name, value)
        return self

    def resolve_device(self) -> str:
        if self.model.device:
            return self.model.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def resolve_dtype(self):
        import torch

        m = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
        return m[self.model.dtype]

    def to_dict(self) -> dict:
        out = {}
        for group in fields(self):
            out[group.name] = dataclasses.asdict(getattr(self, group.name))
        return out


def validate(cfg: Config) -> None:
    if cfg.search.epsilon <= 0:
        raise ValueError("search.epsilon must be > 0")
    if cfg.data.seq_len <= 0 or cfg.data.fit_batch <= 0:
        raise ValueError("data.seq_len / fit_batch must be > 0")
    if cfg.ga.population <= 0:
        raise ValueError("ga.population must be > 0")
    if not cfg.model.layers:
        raise ValueError("model.layers must be non-empty")
