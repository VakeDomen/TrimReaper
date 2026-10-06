"""Analytical (non-evolutionary) pre-analysis for the fixed-mask rotation
experiment.

Three pieces, all operating on the POST-SwiGLU hidden activations ``h`` (the
input to ``down_proj``) where the Givens rotations act:

1. ``channel_importance`` — a Wanda-style per-channel importance score
   ``s_i = E[h_i^2] * ||W_down[:,i]||^2`` (activation energy * outgoing weight
   norm). Channels with a large activation AND strong outgoing weights are
   important; small activation * weak weights => probably removable. Returns
   channels sorted ASCENDING (least important first).

2. ``select_fixed_mask`` — turn the importance ordering into a fixed pruned
   set. The default just takes the ``target`` weakest channels (the truly-dead
   channels are *automatically included* in the deleted set — deleting a dead
   channel is free, so the boundary channel at rank ``target`` is where
   rotations matter). An optional ``skip_bottom`` can be set to skip a number
   of the very weakest, but the intended behavior is skip_bottom=0.

3. ``local_pca_rotations`` — the greedy analytical basis: for each deleted
   channel, pair it with the kept channel whose 2x2 covariance lets a Givens
   rotation push the *deleted* coordinate's variance (L2 energy) closest to
   zero, apply the exact 2D rotation to the shared activation statistics, and
   move on. One sweep; reports the residual energy left in the deleted
   coordinates. This is local PCA composed of pairwise Givens rotations and
   needs NO LLM inference during the sweep (it manipulates cached hidden stats).
"""

from __future__ import annotations

import math

import torch

from .rotation import DOWN, PairRotation, make_orthogonal_matrix


def channel_importance(
    h_batches: list[torch.Tensor],
    down_weight: torch.Tensor,
    max_rows: int | None = None,
) -> torch.Tensor:
    """Per-channel importance score ``E[h_i^2] * ||W_down[:,i]||^2``.

    ``h_batches``: list of post-SwiGLU hidden tensors, each ``(B, S, width)``,
    from a PRISTINE forward (so the scores describe the untouched model).
    ``down_weight``: the layer's ``down_proj`` weight, shape ``(width_out,
    width)``; we use column ``i`` = ``down_weight[:, i]``.

    Returns a ``(width,)`` float tensor ``score`` where ``score[i]`` is the
    importance of channel ``i``; LOWER = more removable. Already on CPU.
    """
    if not h_batches:
        raise ValueError("channel_importance needs at least one activation batch")
    widths = {int(h.shape[-1]) for h in h_batches}
    if len(widths) != 1:
        raise ValueError(f"mismatched hidden widths in batches: {widths}")
    width = widths.pop()
    dw = down_weight.float().detach().cpu()
    if dw.shape[1] != width:
        raise ValueError(f"down_proj width {dw.shape[1]} != hidden width {width}")

    acc = torch.zeros(width)
    count = 0
    for h in h_batches:
        hf = h.float().detach().cpu()
        if max_rows is not None and max_rows > 0:
            rows = min(max_rows, hf.shape[0] * hf.shape[1])
            hf = hf.reshape(-1, width)[:rows]
        else:
            hf = hf.reshape(-1, width)
        acc += (hf * hf).sum(dim=0)
        count += hf.shape[0]
    e_h2 = acc / max(1, count)                       # E[h_i^2]
    w_norm2 = (dw * dw).sum(dim=0)                    # ||W_down[:,i]||^2
    return e_h2 * w_norm2


def importance_order(score: torch.Tensor) -> list[int]:
    """Channels sorted ascending by importance (least important first)."""
    return torch.argsort(score).tolist()


def select_fixed_mask(
    score: torch.Tensor,
    target: int,
    skip_bottom: int = 0,
) -> list[int]:
    """Choose the fixed pruned set from an importance score.

    Takes the ``target`` weakest channels as the deleted set (dead channels are
    automatically included — skipping them is OFF by default). ``skip_bottom``
    may skip that many of the very weakest if you want to exclude trivially-dead
    dimensions. Returns the sorted list of deleted channel ids.
    """
    if skip_bottom is None or skip_bottom < 0:
        skip_bottom = 0
    order = importance_order(score)
    head = order[skip_bottom:]
    selected = head[:target]
    return sorted(selected)


def _givens_angle(d: torch.Tensor, k: torch.Tensor, cd: float,
                  cdk: float, ck: float) -> float:
    """Angle of the 2D rotation that minimizes the energy of the ``d`` output.

    For ``x' = x @ R`` with ``R = [[c, s], [-s, c]]``:
        x'_d = c x_d - s x_k ,  x'_k = s x_d + c x_k.
    We assign the ``d`` output to the MINOR (least-energy) direction of the
    2x2 moment matrix ``[[E[d^2], E[dk]], [E[dk], E[k^2]]]``. The minor
    eigenvector ``(v_d, v_k)`` satisfies ``(c, -s) = (v_d, v_k)``, so
    ``c = v_d`` and ``s = -v_k`` -> ``angle = atan2(-v_k, v_d)``.

    ``d, k, cd, cdk, ck`` are broadcast-compatible (floats or same-shape
    tensors): ``cd``=E[d^2], ``cdk``=E[dk], ``ck``=E[k^2]. Returns the angle in
    radians broadcast to the same shape.
    """
    cd = torch.as_tensor(cd, dtype=torch.float64)
    cdk = torch.as_tensor(cdk, dtype=torch.float64)
    ck = torch.as_tensor(ck, dtype=torch.float64)
    # eigenvalue direction: tan(2a) = 2*cdk / (cd - ck)  =>  a = 0.5*atan2(2*cdk, cd - ck)
    a = 0.5 * torch.atan2(2.0 * cdk, cd - ck)
    # eigenvector at angle `a`: (cos a, sin a) is the MAJOR direction.
    # minor direction is perpendicular: (-sin a, cos a) = (v_d, v_k).
    v_d = -torch.sin(a)
    v_k = torch.cos(a)
    # angle such that (c, -s) = (v_d, v_k): c = v_d, -s = v_k => s = -v_k
    angle = torch.atan2(-v_k, v_d)
    return angle


def local_pca_rotations(
    h: torch.Tensor,
    pruned: list[int],
    max_rows: int | None = None,
) -> tuple[list[PairRotation], dict]:
    """Greedy one-sweep analytical basis: push deleted-channel energy to zero.

    ``h``: ``(T, width)`` post-SwiGLU hidden activations (single flattened
    tensor) from a PRISTINE forward. ``pruned``: the fixed deleted channel ids.

    For each deleted channel (in given order), scan all KEPT channels and pick
    the partner whose 2x2 Givens rotation minimizes the deleted coordinate's
    residual energy; apply that exact rotation to the shared activation columns
    (so later partner choices see updated statistics), record the PairRotation,
    and continue. ``max_rows`` subsamples ``h`` for the covariance so the sweep
    is cheap (it only needs statistics).

    Returns ``(rotations, report)`` where ``report`` carries the residual L2
    energy in the deleted coordinates and per-deleted residual energies.
    """
    width = h.shape[-1]
    pruned_set = set(int(c) for c in pruned)
    if len(pruned_set) != len(pruned):
        raise ValueError("pruned contains duplicates")
    if not pruned:
        raise ValueError("local_pca_rotations needs a non-empty pruned set")
    for c in pruned:
        if not (0 <= int(c) < width):
            raise ValueError(f"pruned channel {c} out of range [0,{width})")

    # NOTE: .float()/.cpu() can return the SAME tensor (alias) for an already
    # float/CPU input, so we explicitly .clone() to avoid mutating the caller's
    # `h` in place during the greedy sweep.
    H = h.detach().float().cpu().clone()
    if max_rows is not None and max_rows > 0 and max_rows < H.shape[0]:
        H = H[:max_rows]
    if H.dim() == 3:
        H = H.reshape(-1, width)
    n = float(H.shape[0])
    if n <= 0:
        raise ValueError("local_pca_rotations needs at least one observation row")
    H0 = H.clone()

    # We work EXACTLY with the running hidden activations (the same space the
    # model rotates: h' = h @ Q), so every energy/cross-moment is computed from
    # real columns — no error-prone full-moment bookkeeping. A 2x2 Givens on
    # (d, k) updates just those two columns of H.
    kept = [c for c in range(width) if c not in pruned_set]
    kept_idx = {c: i for i, c in enumerate(kept)}

    e2 = (H * H).mean(dim=0)                 # (width,) current E[h_i^2]
    rotations: list[PairRotation] = []
    for d in pruned:
        d = int(d)
        e_d = e2[d].item()
        hd = H[:, d]
        # vectorized cross-moments + kept-variances for partner selection
        cdk_arr = (hd.unsqueeze(1) * H[:, kept]).mean(dim=0)   # E[d k] over kept
        ck_arr = e2[kept]                                       # E[k^2] over kept
        tr2 = 0.5 * (e_d + ck_arr)
        det = e_d * ck_arr - cdk_arr * cdk_arr
        disc = torch.clamp(tr2 * tr2 - det, min=0.0)
        min_energy = tr2 - torch.sqrt(disc)
        k_idx = int(torch.argmin(min_energy).item())
        k = kept[k_idx]
        cdk = float(cdk_arr[k_idx].item())
        ck = float(ck_arr[k_idx].item())
        angle = float(_givens_angle(d, k, e_d, cdk, ck).item())
        # apply the exact 2x2 Givens to columns (d, k): h'_d = c h_d - s h_k,
        # h'_k = s h_d + c h_k (matches the model's h @ Q convention).
        c = math.cos(angle)
        s = math.sin(angle)
        old_d = H[:, d].clone()
        old_k = H[:, k].clone()
        H[:, d] = c * old_d - s * old_k
        H[:, k] = s * old_d + c * old_k
        e2[d] = (H[:, d] * H[:, d]).mean()
        e2[k] = (H[:, k] * H[:, k]).mean()
        # Emit with the DELETED channel as endpoint `a` and the KEPT channel as
        # endpoint `b`: make_orthogonal_matrix gives h'_a = c h_a - s h_b, so
        # with a=d, b=k the deleted coordinate absorbs the minor axis
        # (h'_d = c h_d - s h_k), exactly matching the transform applied above.
        rotations.append(PairRotation(a=d, b=k, angle=angle))

    # TRUE residual after the composed Q applied to the ORIGINAL data
    Q = make_orthogonal_matrix(width, rotations)
    Y = H0 @ Q
    residual = {d: float((Y[:, int(d)] ** 2).mean()) for d in pruned}
    total_before = float((H0[:, list(pruned_set)] ** 2).sum(dim=1).mean().item()) if len(pruned) else 0.0
    total_after = float(sum(residual.values()))
    report = {
        "n_rot": len(rotations),
        "deleted_before_energy": total_before,
        "deleted_after_energy": total_after,
        "energy_ratio": (total_after / total_before) if total_before > 0 else float("nan"),
        "per_deleted_residual": {str(d): v for d, v in residual.items()},
    }
    return rotations, report
