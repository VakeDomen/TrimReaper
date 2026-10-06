"""Pairwise Givens rotations on Qwen MLP hidden channels.

Implements PLAN.md section 3, corrected for gated (SwiGLU) MLPs.

For two MLP hidden channels ``i`` and ``j``, the geometric idea is to rotate
the *post-SwiGLU hidden activation* and counter-rotate the projection that
reads it, so the function is EXACTLY preserved when no channel is masked:

    gate_proj ─┐
               ├─ SiLU(gate) * up ── h ── ROT(Q) ── MASK ── down_proj'
    up_proj ───┘                    down_proj' = down_proj . Q   (Q orthogonal)

    h' = h @ Q            (rotate the hidden activation)
    down_proj' = down_proj @ Q

Because Q is orthogonal, with the mask disabled:

    down_proj'(h') = (h @ Q) @ (down_proj @ Q)^T
                   = h @ Q @ Q^T @ down_proj^T = h @ down_proj^T = down_proj(h)

so the transform is function-preserving up to floating point roundoff. This is
a genuine basis rotation of the representation (unlike rotating the raw
gate/up rows, which is NOT invariant once SiLU is applied). Deleting a channel
afterwards deletes a ROTATED coordinate, which is where the redundancy-
concentration payoff comes from.

The rotations are composed into a single orthogonal matrix ``Q`` endowed with
``h' = h @ Q``; ``down_proj`` is then transformed as ``W_down @ Q``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

# Qwen gated-MLP matrix names.
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
    """Return the 2x2 Givens rotation matrix for an h' = h @ R convention."""
    c = math.cos(angle)
    s = math.sin(angle)
    # For x' = x @ M with M = [[c, s], [-s, c]], we get
    #   x'_a = c x_a - s x_b,  x'_b = s x_a + c x_b.   (standard +theta rotation)
    return torch.tensor([[c, s], [-s, c]], device=device, dtype=dtype)


def make_orthogonal_matrix(width: int, rots: list[PairRotation], device=None, dtype=None) -> torch.Tensor:
    """Compose a sequence of Givens rotations into one orthogonal matrix Q.

    Returns ``Q`` of shape (width, width) such that applying the rotations to a
    hidden vector is ``x' = x @ Q``. Rotations are applied **left-to-right**
    (in the listed order): the hidden vector is transformed by each rotation in
    turn, ``x' = (...((x @ M_1) @ M_2) ... @ M_n)``, equivalently
    ``Q = M_1 @ M_2 @ ... @ M_n`` where M_i is the block-diagonal embedding of
    the i-th Givens on rows/cols (a_i, b_i).

    Sequential application of a Givens on (a, b) updates exactly the two hidden
    coordinates:
        h'_a = c h_a - s h_b
        h'_b = s h_a + c h_b
    For ``h' = h @ Q`` this is achieved by RIGHT-multiplying Q's identity by
    each M_i, which updates the two **(a, b) COLUMNS**: because ``h'[:,j] =
    sum_i h[:,i] Q[i,j]``, rotating coordinates (a, b) rewrites columns a and b
    as
        Q[:, a] = c old_a - s old_b
        Q[:, b] = s old_a + c old_b

    IMPORTANT (perf): clone ONLY the two columns being changed, never the whole
    matrix — O(width) work per rotation instead of O(width^2) — critical at
    1000+ rotations on a 9728-wide MLP. (A row-update implementation would
    silently REVERSE the composition order; columns are what make the listed
    order equal the applied order.)
    """
    Q = torch.eye(width, device=device, dtype=dtype)
    for rot in rots:
        a, b, ang = rot.a, rot.b, rot.angle
        c = math.cos(ang)
        s = math.sin(ang)
        old_a = Q[:, a].clone()
        old_b = Q[:, b].clone()
        Q[:, a] = c * old_a - s * old_b
        Q[:, b] = s * old_a + c * old_b
    return Q


def rotate_down_weight(down: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """Counter-rotate down_proj to stay consistent with an h' = h @ Q rotation.

    ``down`` has shape (hidden, width); returns ``down @ Q``. Together with
    ``x' = x @ Q`` this preserves ``down_proj(h)`` exactly (Q orthogonal).
    """
    return down @ Q


def apply_pair_rotation(params: dict[str, torch.Tensor], rot: PairRotation) -> None:
    """[LEGACY, only for tests/linear-MLP paths] Rotate gate/up rows + down cols.

    Kept for backward compatibility with the old (linear-only) semantics. New
    code should use make_orthogonal_matrix + rotate_down_weight + a hidden hook.
    """
    a, b, angle = rot.a, rot.b, rot.angle
    dev = next(iter(params.values())).device
    dt = next(iter(params.values())).dtype
    R = givens_matrix(angle, device=dev, dtype=dt)
    Rt = R.transpose(0, 1)

    g = params[GATE]
    u = params[UP]
    d = params[DOWN]

    idx = torch.tensor([a, b], device=dev)
    for mat in (g, u):
        rows = mat[idx]
        mat[idx] = R @ rows

    cols = d[:, idx]
    d[:, idx] = cols @ Rt


def apply_rotation_sequence(params: dict[str, torch.Tensor], rots: list[PairRotation]) -> None:
    """[LEGACY] Apply a sequence of rotations in order, IN-PLACE on ``params``."""
    for rot in rots:
        apply_pair_rotation(params, rot)


def rotation_distance(angle_a: float, angle_b: float) -> float:
    """Circular distance between two angles in radians."""
    d = (angle_a - angle_b) % (2 * math.pi)
    if d > math.pi:
        d -= 2 * math.pi
    return abs(d)
