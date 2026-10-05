"""Pairwise Givens rotations on Qwen MLP channels.

Implements PLAN.md section 3. For two MLP hidden channels ``i`` and ``j``:

    R(theta) = [ cos(theta)  -sin(theta) ]
               [ sin(theta)   cos(theta) ]

Applied consistently to the three matrices of a gated MLP:

    gate_proj rows [i, j]  <- R @ gate_proj[i, j]        (2560 -> 9728)
    up_proj   rows [i, j]  <- R @ up_proj[i, j]          (2560 -> 9728)
    down_proj cols [i, j]  <- down_proj[:, i, j] @ R^T   (9728 -> 2560)

If the MLP were linear this would preserve the function exactly; Qwen's Multi
Layer Perceptron is gated SiLU, so the transform is NOT exactly invariant --
that is intentional (the GA must find rotations that change the nonlinear
model very little yet concentrate information so channels become deletable).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

# Qwen gated-MLP matrix names. These are forwarded into the wrapper's
# parameter map so a Rotator works against {gate,up,down} projection tensors.
GATE, UP, DOWN = "gate_proj", "up_proj", "down_proj"
MLP_MATRICES = (GATE, UP, DOWN)


@dataclass
class PairRotation:
    """A single pairwise Givens rotation on channels (a, b) by ``angle``."""

    a: int
    b: int
    angle: float

    def to_tuple(self) -> tuple[int, int, float]:
        return (self.a, self.b, self.angle)

    @classmethod
    def from_tuple(cls, t: tuple[int, int, float]) -> "PairRotation":
        return cls(a=int(t[0]), b=int(t[1]), angle=float(t[2]))

    def validate(self, width: int) -> None:
        if not (0 <= self.a < width and 0 <= self.b < width):
            raise ValueError(f"rotation channel out of range: ({self.a},{self.b}) width={width}")
        if self.a == self.b:
            raise ValueError("rotation channels must differ")


def givens_matrix(angle: float, device=None, dtype=None) -> torch.Tensor:
    """Return the 2x2 Givens rotation matrix R(angle)."""
    c = math.cos(angle)
    s = math.sin(angle)
    return torch.tensor([[c, -s], [s, c]], device=device, dtype=dtype)


def apply_pair_rotation(params: dict[str, torch.Tensor], rot: PairRotation) -> None:
    """Apply one rotation IN-PLACE to the MLP weight tensors in ``params``.

    ``params`` maps matrix name -> weight, each with channel dim as follows:
        gate_proj / up_proj: shape (out_channels, in_features) -> rows i,j
        down_proj:           shape (in_features, out_channels) -> cols i,j
    """
    a, b, angle = rot.a, rot.b, rot.angle
    dev = next(iter(params.values())).device
    dt = next(iter(params.values())).dtype
    R = givens_matrix(angle, device=dev, dtype=dt)
    Rt = R.transpose(0, 1)

    g = params[GATE]
    u = params[UP]
    d = params[DOWN]

    # Rows [i,j] of a (N, K) matrix. New rows = R @ old_rows (2,K).
    #   new_row_i = c*row_i - s*row_j
    #   new_row_j = s*row_i + c*row_j
    idx = torch.tensor([a, b], device=dev)
    for mat in (g, u):
        rows = mat[idx]                     # (2, K)
        mat[idx] = R @ rows

    # Columns [i,j] of a (K, N) matrix. New cols = old_cols @ R^T.
    #   new_col_i = c*col_i - s*col_j
    #   new_col_j = s*col_i + c*col_j
    cols = d[:, idx]                        # (K, 2)
    d[:, idx] = cols @ Rt


def apply_rotation_sequence(params: dict[str, torch.Tensor], rots: list[PairRotation]) -> None:
    """Apply a sequence of rotations in order, IN-PLACE on ``params``."""
    for rot in rots:
        apply_pair_rotation(params, rot)


def rotation_distance(angle_a: float, angle_b: float) -> float:
    """Circular distance between two angles in radians."""
    d = (angle_a - angle_b) % (2 * math.pi)
    if d > math.pi:
        d -= 2 * math.pi
    return abs(d)
