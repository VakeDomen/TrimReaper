"""Non-evolutionary baselines (PLAN.md section 13).

One-shot pruning curves used to compare against the evolutionary methods:
  - random channel deletion
  - weight-norm channel ranking  (L2 norm of the channel's weights)
  - activation-magnitude ranking (mean abs activation of each channel)

Each returns an ordered list of channels to delete (highest priority first).
The pipeline then evaluates the KL curve by deleting progressively more.
"""

from __future__ import annotations

import random

import torch

from .model import PrunedModel
from .rotation import DOWN, GATE, UP


def random_ranking(width: int, rng: random.Random | None = None) -> list[int]:
    rng = rng or random.Random(0)
    order = list(range(width))
    rng.shuffle(order)
    return order


def weight_norm_ranking(pm: PrunedModel, layer: int) -> list[int]:
    """Rank channels by total L2 weight magnitude across gate/up/down.

    For gate/up rows the channel index is a row of the (out=width, in) matrix;
    for down_proj it is a column. Compute each channel's contribution and sort
    ascending so the SMALLEST-magnitude channels are deleted first.
    """
    m = pm.pristine[layer]
    gate = m[GATE].float()      # (width, hidden)
    up = m[UP].float()          # (width, hidden)
    down = m[DOWN].float()      # (hidden, width)

    gate_norm = gate.norm(dim=1)          # (width,)
    up_norm = up.norm(dim=1)              # (width,)
    down_norm = down.norm(dim=0)          # (width,)
    total = gate_norm + up_norm + down_norm
    # delete smallest magnitude first
    order = torch.argsort(total).tolist()
    return order


def activation_magnitude_ranking(
    pm: PrunedModel, layer: int, probe_batches: list[torch.Tensor], seq_len: int
) -> list[int]:
    """Rank channels by mean |activation| under a probe set on the PRISTINE model.

    The probe measures the hidden-channel activations act(gate(x)) * up(x)
    (the input to down_proj) for each channel, averaged over the probe batches.
    Channels with the smallest mean absolute activation are deleted first.
    """
    mlp = pm.mlp_modules[layer]
    pm.restore_all()
    # temporarily hook the pre-activation of down_proj to capture hidden states
    captured = []

    def hook(mod, args):
        captured.append(args[0].detach())

    handle = mlp.down_proj.register_forward_pre_hook(hook)
    try:
        with torch.inference_mode():
            for batch in probe_batches:
                _ = pm.model(input_ids=batch[:, :seq_len].to(pm.model.device), use_cache=False)
    finally:
        handle.remove()

    if not captured:
        return list(range(mlp.down_proj.in_features))
    hiddens = torch.cat(captured, dim=0)          # (B*N, seq, width)
    mags = hiddens.abs().mean(dim=(0, 1))         # (width,) mean |act| per channel
    order = torch.argsort(mags).tolist()          # smallest first
    return order
