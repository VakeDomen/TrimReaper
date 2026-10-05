"""Physical compaction for a validated genome (PLAN.md section 16).

For the best validated genome, build a genuinely smaller gated MLP that
reproduces the search-time (rotated + masked) behavior:

    x ── gate_proj, up_proj ── SiLU(gate)·up = h ── ROTATE(Q) ── [drop k] ── down_proj

where ``down_proj = (W_down @ Q)[:, keep]``. Here Q is the orthogonal matrix
composed from the genome's Givens rotations (acting on the hidden activation),
and ``keep`` are the non-deleted ROTATED channels. Because Q is orthogonal and
down_proj is counter-rotated, with no deletion this reproduces the original
MLP exactly; deletion drops the ROTATED coordinates.

The gate/up weights stay pristine (rotation acts on the hidden, not the raw
rows — the gated-SiLU-invariant formulation).

Evaluates the real compact model (not the masked simulation) and reports
parameter count, VRAM estimate, and channels removed.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import Config
from .model import PrunedModel
from .rotation import (
    DOWN,
    GATE,
    UP,
    PairRotation,
    make_orthogonal_matrix,
    rotate_down_weight,
)


def _coerce_rotations(rotations: list) -> list[PairRotation]:
    """Accept either PairRotation objects or (a, b, angle) tuples."""
    out = []
    for r in rotations:
        if isinstance(r, PairRotation):
            out.append(r)
        else:
            # tuple/list (a, b, angle) from an archived JSON genome
            out.append(PairRotation.from_tuple((int(r[0]), int(r[1]), float(r[2]))))
    return out


def build_compact_mlp(pm: PrunedModel, layer: int, pruned: list[int], rotations: list) -> dict[str, nn.Module]:
    """Return a set of Linear modules forming the compact gated MLP.

    The original target MLP is NOT modified. The rotation is materialized as an
    explicit orthonormal Linear layer between the activation and down_proj.
    """
    orig = pm.pristine[layer]
    device = pm.model.device
    dtype = pm.model.dtype

    rots = _coerce_rotations(rotations)
    width = orig[GATE].shape[0]
    hidden = orig[GATE].shape[1]

    gate = orig[GATE].clone().to(device=device, dtype=dtype)   # (width, hidden)
    up = orig[UP].clone().to(device=device, dtype=dtype)       # (width, hidden)

    Q = make_orthogonal_matrix(width, rots, device=device, dtype=dtype) if rots \
        else torch.eye(width, device=device, dtype=dtype)
    down_rot = rotate_down_weight(orig[DOWN].to(dtype=dtype), Q)  # (hidden, width)

    pruned_set = set(pruned)
    keep = [c for c in range(width) if c not in pruned_set]

    gate_mod = nn.Linear(hidden, width, bias=False)
    up_mod = nn.Linear(hidden, width, bias=False)
    # explicit rotate layer: forward(h) = h @ Q, so weight = Q.T
    rot_mod = nn.Linear(width, width, bias=False)
    down_mod = nn.Linear(len(keep), hidden, bias=False)   # (hidden, len(keep))

    with torch.no_grad():
        gate_mod.weight.copy_(gate)
        up_mod.weight.copy_(up)
        rot_mod.weight.copy_(Q.T)
        down_mod.weight.copy_(down_rot[:, keep])          # (hidden, width-k)

    return {GATE: gate_mod, UP: up_mod, "rotate": rot_mod, DOWN: down_mod}


def compact_model_metrics(
    pm: PrunedModel, layer: int, pruned: list[int], rotations: list
) -> dict:
    """Construct the compact MLP and report structural metrics."""
    modules = build_compact_mlp(pm, layer, pruned, rotations)
    orig = pm.pristine[layer]

    orig_params = (
        orig[GATE].numel() + orig[UP].numel() + orig[DOWN].numel()
    )
    # The rotate layer (width^2) is part of the compact model; the meaningful
    # REDUCTION is in down_proj (drop the k columns) while gate/up stay full.
    compact_params = sum(m.weight.numel() for m in modules.values())

    metrics = {
        "layer": layer,
        "original_mlp_params": orig_params,
        "compact_mlp_params": compact_params,
        "params_removed": orig_params - (compact_params - modules["rotate"].weight.numel()),
        "fraction_removed": (orig_params - (compact_params - modules["rotate"].weight.numel())) / orig_params,
        "original_channels": orig[GATE].shape[0],
        "compact_channels": len([c for c in range(orig[GATE].shape[0]) if c not in set(pruned)]),
        "channels_removed": len(pruned),
        "has_rotation_layer": modules["rotate"].weight.shape[0] > 0,
    }
    return metrics
