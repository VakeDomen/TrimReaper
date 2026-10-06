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
    # Rotation budget scales with the deletion target: budget =
    # clamp(int(rotations_per_removed * target), min_rotations, max_rotations).
    # Every genome uses EXACTLY this many rotations (a fixed count) so genomes
    # are directly comparable and we don't evolve complexity at the same time
    # as the solution. Budget ladder for 512 deleted (2^x * deleted): 1x=512,
    # 2x=1024, 4x=2048, 8x=4096.
    rotations_per_removed: float = 1.0
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
    """Genetic algorithm / explorer-population hyperparameters."""

    population: int = 32
    elitism: int = 2
    tournament_size: int = 2       # best-of-k bases when re-basing a stuck explorer
    # Independent-explorer mode (no crossover): a candidate that goes this many
    # consecutive generations without beating the global elite (same-batch KL)
    # is re-based onto a tournament-selected base and starts a fresh path.
    fail_limit: int = 5
    seed: int = 0


@dataclass
class MutationConfig:
    """The complete mutation schema (what each explorer does per generation).

    Three mutation types, deliberately minimal and non-overlapping:
        ANGLE  — try a different rotation in the same plane (each explorer got a
                 fixed assigned rate in [angle_fraction_min, angle_fraction_max]).
        PAIR   — try a different plane: rewire the endpoints of a fixed fraction
                 of rotations (always a valid pruned<->kept pair).
        MASK   — try deleting a different dimension: swap some pruned channels
                 for kept ones (exact swap, keeps the removal count).

    ANGLE happens 100% of the time (the assigned fraction controls HOW MANY
    angles). PAIR and MASK are the reviewer's fixed, explorer-independent rates
    so they don't contaminate the per-explorer angle-mutation comparison.
    """

    # Per-explorer assigned angle-mutation fraction, linearly spaced across the
    # population from min to max (e.g. min 0.003 -> max 0.10 across 32 slots:
    # ~0.3%..10%, i.e. ~1..44 angles on a 440-rotation genome). Fixed per slot,
    # survives re-basing.
    angle_fraction_min: float = 0.003
    angle_fraction_max: float = 0.10
    # Endpoint rewiring: fraction of rotations whose (a, b) pair is re-drawn as
    # a fresh pruned<->kept pair each generation (angle preserved).
    pair_rewire_fraction: float = 0.02
    # Mask mutation: with mask_swap_probability, swap mask_swap_count pruned
    # channels for kept ones per generation (exact swap preserves the target).
    mask_swap_count: int = 1
    mask_swap_probability: float = 0.20
    # Angle change distribution: small Gaussian bump (angle_std), with an
    # occasional large jump (large_angle_probability / large_angle_std).
    angle_std: float = 0.15
    large_angle_probability: float = 0.05
    large_angle_std: float = 1.0


@dataclass
class AnalysisConfig:
    """Analytical (non-evolutionary) pre-analysis for the fixed-mask experiment.

    Channel importance uses the Wanda-style score E[h_i^2] * ||W_down[:,i]||^2
    (activation energy x outgoing weight norm); channels are ranked ascending
    (least important first). The fixed pruned mask takes the ``target`` weakest
    channels; the truly-dead ones are INCLUDED automatically (deleting them is
    free — ``skip_bottom`` defaults to 0 to skip a band if ever wanted).
    ``local_pca_rotations`` greedily pairs each deleted channel with a kept
    partner that zeroes its 2x2 variance (on ``max_rows`` cached hidden rows)
    and composes the exact Givens rotations, then the GA is seeded from that
    analytical solution.
    """

    # Subsample cap of cached hidden rows used for the covariance/energy math in
    # local_pca_rotations (0 = use all rows). Reduces the offline sweep cost.
    max_rows: int = 2048
    # Number of the very weakest channels to skip when building the mask (0 =
    # include them, the intended default).
    skip_bottom: int = 0
    # GA seeding around the analytical solution: n_exact exact clones, n_small
    # mild angle perturbations, n_large stronger angle + pair-rewire mutations,
    # remainder random.
    n_exact: int = 1
    n_small: int = 8
    n_large: int = 8
    # Number of calibration probe batches used to collect the post-SwiGLU hidden
    # activations (for the importance score AND the PCA sweep). With
    # fit_batch*seq_len tokens per batch this sizes the calibration: e.g.
    # fit_batch=4, seq_len=256 -> 4 batches = 4096 tokens. Tune by watching when
    # the selected weakest-N mask stops changing much (1/2/4/8/16 batches).
    probe_batches: int = 4


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
    mutation: MutationConfig = field(default_factory=MutationConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
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
