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


def test_make_orthogonal_matrix_is_orthogonal_by_rows():
    """The Q builder must compose many Givens rotations into a single
    ORTHOGONAL matrix — even after switching from a full-matrix clone to the
    row-only clone (fix #6: clone just the two changed rows each rotation). Q^T Q
    must equal I for a long rotation sequence."""
    from trimreaper.rotation import PairRotation, make_orthogonal_matrix

    width = 32
    rots = [PairRotation(a=i, b=(i + 7) % width, angle=0.1 * i) for i in range(20)]
    Q = make_orthogonal_matrix(width, rots, dtype=torch.float64)
    assert Q.shape == (width, width)
    gram = Q.t() @ Q
    assert torch.allclose(gram, torch.eye(width, dtype=torch.float64), atol=1e-9)
    # applying to a vector preserves its norm (rotation)
    v = torch.randn(width, dtype=torch.float64)
    assert torch.allclose(((v @ Q) ** 2).sum(), (v ** 2).sum(), atol=1e-9)


def test_make_orthogonal_matches_reference_columns():
    """Regression: the column-only clone optimization must reproduce the exact
    matrix produced by a reference naive implementation (clone whole Q each
    step), so identical rotations yield identical Q. The reference uses COLUMN
    updates (right-multiplication) which is what makes the listed rotation order
    equal the applied order."""
    import math

    from trimreaper.rotation import PairRotation, make_orthogonal_matrix

    width = 16
    rots = [PairRotation(3, 10, 0.7), PairRotation(1, 15, -1.1), PairRotation(0, 7, 2.0)]

    def reference(width, rots):
        Q = torch.eye(width, dtype=torch.float64)
        for r in rots:
            c, s = math.cos(r.angle), math.sin(r.angle)
            old = Q.clone()
            Q[:, r.a] = c * old[:, r.a] - s * old[:, r.b]
            Q[:, r.b] = s * old[:, r.a] + c * old[:, r.b]
        return Q

    q_new = make_orthogonal_matrix(width, rots, dtype=torch.float64)
    q_ref = reference(width, rots)
    assert torch.allclose(q_new, q_ref, atol=1e-12)


def test_make_orthogonal_applies_in_listed_order_overlapping_pairs():
    """Regression for the Q composition ORDER bug.

    The builder must apply rotations left-to-right (in the listed order). A
    row-update implementation silently left-multiplies and therefore REVERSES
    the sequence. Overlapping pairs (0,1),(1,2),(0,2) do NOT commute, so they
    expose the ordering error where disjoint pairs would not: applying the same
    rotations as a sequential column transform (matching the evaluator) and as
    one composed x@Q must give the same hidden vector."""
    import math

    from trimreaper.rotation import PairRotation, make_orthogonal_matrix

    width = 4
    rots = [PairRotation(0, 1, 0.7), PairRotation(1, 2, -0.5), PairRotation(0, 2, 1.2)]
    X = torch.randn(50, width, dtype=torch.float64)

    def apply_seq(x, seq):
        y = x.clone()
        for r in seq:
            a, b, ang = r.a, r.b, r.angle
            c, s = math.cos(ang), math.sin(ang)
            oa = y[:, a].clone()
            ob = y[:, b].clone()
            y[:, a] = c * oa - s * ob
            y[:, b] = s * oa + c * ob
        return y

    # sequential forward application = the order the evaluator actually applies
    expected = apply_seq(X, rots)
    Q = make_orthogonal_matrix(width, rots, dtype=torch.float64)
    torch.testing.assert_close(X @ Q, expected, atol=1e-9, rtol=1e-9)
    # and NOT the reversed order
    reversed_ = apply_seq(X, list(reversed(rots)))
    assert not torch.allclose(X @ Q, reversed_, atol=1e-6)


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
