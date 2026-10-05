"""Unit tests for Givens rotation math and MLP rotation application."""

import math

import torch


def _make_mlp_tensors(width=8, hidden=5, dtype=torch.float64):
    """Build fake MLP weight tensors matching Qwen's layout."""
    gt = torch.randn(width, hidden, dtype=dtype)
    up = torch.randn(width, hidden, dtype=dtype)
    down = torch.randn(hidden, width, dtype=dtype)
    return {"gate_proj": gt, "up_proj": up, "down_proj": down}


def test_givens_matrix_orthogonal():
    from trimreaper.rotation import givens_matrix

    th = 0.7
    R = givens_matrix(th)
    # R^T R = I
    prod = R @ R.t()
    assert torch.allclose(prod, torch.eye(2, dtype=R.dtype), atol=1e-9)


def test_rotation_preserves_pair_gram_structure():
    """An orthogonal rotation preserves the Gram matrix of the rotated pair.

    Rows [i,j] are rotated by an orthogonal matrix R, so the Gram structure
    (and hence pair norm) is preserved. Individual row norms DO change because
    the rows mix — that is expected and correct.
    """
    from trimreaper.rotation import PairRotation, apply_pair_rotation

    params = _make_mlp_tensors()
    orig = {k: v.clone() for k, v in params.items()}

    rot = PairRotation(a=2, b=5, angle=0.9)
    apply_pair_rotation(params, rot)

    for key in ("gate_proj", "up_proj"):
        pair_rot = params[key][[2, 5]]     # (2,K), rotated as R @ old
        pair_orig = orig[key][[2, 5]]
        # rows rotate with R on the LEFT -> preserved inner Gram is T @ A
        assert torch.allclose(pair_rot.t() @ pair_rot, pair_orig.t() @ pair_orig, atol=1e-9)

    # columns [2,5] of down_proj rotate by R^T (also orthogonal)
    pair_rot = params["down_proj"][:, [2, 5]]   # (K,2), rotated as old @ R^T
    pair_orig = orig["down_proj"][:, [2, 5]]
    # columns rotate with R^T on the RIGHT -> preserved outer Gram is A @ A^T
    assert torch.allclose(pair_rot @ pair_rot.t(), pair_orig @ pair_orig.t(), atol=1e-9)


def test_rotation_inverse_recovers_original():
    """Applying +angle then -angle should restore the original weights."""
    from trimreaper.rotation import PairRotation, apply_pair_rotation

    params = _make_mlp_tensors()
    orig = {k: v.clone() for k, v in params.items()}
    th = 1.1
    apply_pair_rotation(params, PairRotation(1, 6, th))
    apply_pair_rotation(params, PairRotation(1, 6, -th))
    for k in params:
        assert torch.allclose(params[k], orig[k], atol=1e-9)


def test_rows_and_cols_transform_as_documented():
    """Verify rows rotate by R and columns by R^T, matching PLAN.md section 3."""
    from trimreaper.rotation import PairRotation, apply_pair_rotation, givens_matrix

    W = torch.randn(8, 8, dtype=torch.float64)
    D = torch.randn(8, 8, dtype=torch.float64)
    params = {
        "gate_proj": W.clone(),
        "up_proj": torch.zeros(8, 8, dtype=torch.float64),
        "down_proj": D.clone(),
    }
    th = 0.5
    apply_pair_rotation(params, PairRotation(3, 4, th))
    R = givens_matrix(th, dtype=W.dtype)

    g = params["gate_proj"]
    # new rows [[g3],[g4]] = R @ old rows
    expected = R @ W[[3, 4]]
    assert torch.allclose(g[[3, 4]], expected, atol=1e-9)
    # untouched rows unchanged
    assert torch.allclose(g[5], W[5], atol=1e-9)

    # down columns: new cols [3,4] = old cols @ R^T
    d = params["down_proj"]
    expected_cols = D[:, [3, 4]] @ R.t()
    assert torch.allclose(d[:, [3, 4]], expected_cols, atol=1e-9)
    from trimreaper.rotation import PairRotation, apply_rotation_sequence

    params = _make_mlp_tensors()
    seq = [PairRotation(0, 1, 0.3), PairRotation(4, 7, 1.2)]
    apply_rotation_sequence(params, seq)
    # no crash; sequence lengths preserved
    for k, v in params.items():
        assert v.shape == _make_mlp_tensors()[k].shape
