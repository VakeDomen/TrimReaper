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

    min_rotations: int = 16
    max_rotations: int = 64
    # Rotation budget scales with the deletion target so larger prunes get more
    # rotational freedom instead of a fixed cap: budget =
    # clamp(int(rotations_per_removed * target), min_rotations, max_rotations).
    rotations_per_removed: float = 0.5
    min_angle: float = -3.141592653589793 * 2
    max_angle: float = 3.141592653589793 * 2
    # Deletion-bias window: with some probability, rotation pairs are drawn
    # such that one endpoint is a candidate-for-deletion and the other a
    # surviving channel (per PLAN.md section 6).
    delete_survive_bias: float = 0.7


@dataclass
class GaConfig:
    """Genetic algorithm hyperparameters."""

    population: int = 32
    elitism: int = 2
    tournament_size: int = 4       # tournament selection: best of k parents
    mutation_rate: float = 0.3
    add_rotation_p: float = 0.15
    remove_rotation_p: float = 0.10
    angle_mutate_p: float = 0.5
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
