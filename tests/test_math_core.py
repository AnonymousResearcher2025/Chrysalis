import itertools
import math
import numpy as np
import pytest
from chrysalis import _core
from chrysalis.math import fit_bridge, normalize, quantile


def test_conformal_independent_rank():
    for m in range(0, 60):
        for alpha in (.01, .05, .1, .5):
            s = list(range(m))
            q, rank = quantile(s, alpha)
            expected = math.ceil((1 - alpha) * (m + 1))
            assert rank == expected
            assert q == (expected - 1 if expected <= m else math.inf)
    with pytest.raises(ValueError):
        quantile([float('nan')], .05)


def test_bridges_and_distances():
    rng = np.random.default_rng(11)
    x = normalize(rng.normal(size=(100, 6)))
    w, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    y = x @ w
    b = fit_bridge(x, y, rank=64, ridge=1e-6)
    assert np.max(np.abs(b(x) - y)) < 1e-6
    assert np.allclose(b.W.T @ b.W, np.eye(6), atol=1e-6)
    target = rng.normal(size=(6, 9))
    b = fit_bridge(x, x @ target + 2, rank=3, ridge=1e-6)
    assert np.linalg.matrix_rank(b.W, tol=1e-5) <= 3
    assert not np.allclose(np.linalg.norm(b(x), axis=1), 1)
    # Compare regularized objective with independently optimized rank-3 factor.
    import scipy.optimize
    def loss(z):
        A, B = z[:18].reshape(6, 3), z[18:].reshape(3, 9)
        W = A @ B
        xc = x - x.mean(0)
        yc = x @ target - (x @ target).mean(0)
        return np.sum((xc @ W - yc) ** 2) + 1e-6 * np.sum(W ** 2)
    u, s, vt = np.linalg.svd(b.W, full_matrices=False)
    z = np.r_[(u[:, :3] * np.sqrt(s[:3])).ravel(), (np.sqrt(s[:3])[:, None] * vt[:3]).ravel()].astype('float64')
    opt = scipy.optimize.minimize(loss, z, method='BFGS')
    assert loss(z) <= opt.fun + 1e-4
    assert _core.distance([1, 0], [0, 1]) == pytest.approx(np.sqrt(2))
    with pytest.raises(ValueError):
        normalize([[0, 0]])


def test_certificate_against_exhaustive_rankings():
    rng = np.random.default_rng(9)
    for n in range(1, 8):
        for k in range(1, n + 3):
            for _ in range(40):
                actual = rng.uniform(0, 3, size=n)
                radius = .4
                estimates = np.maximum(0, actual + rng.uniform(-radius, radius, n))
                order = sorted(range(n), key=lambda i: (estimates[i], i))
                c = [_core.Candidate(i, float(estimates[i]), 0, 1) for i in order]
                result = _core.certificate(c, k, radius)
                true = set(sorted(range(n), key=lambda i: (actual[i], i))[:k])
                measured = len(true & set(order[:k])) / k
                assert measured + 1e-12 >= result['bound']
    tied = [_core.Candidate(i, 1., 0, 3) for i in range(4)]
    assert _core.certificate(tied, 2, 0)['bound'] == 0
    assert _core.certificate([], 10, math.inf)['bound'] == 0
    assert _core.certificate(tied[:2], 4, math.inf)['bound'] == .5


def test_graph_is_real_traversal_and_reverse_consistent():
    rng = np.random.default_rng(4)
    x = normalize(rng.normal(size=(500, 16)))
    g = _core.Graph(12, 60, 1.2, 42)
    g.build(x.tolist(), [0] * len(x))
    topo = g.topology()
    assert topo['top'] > 0
    assert all(len(layer) <= 12 for n in topo['edges'] for layer in n)
    hits = []
    for q in normalize(rng.normal(size=(30, 16))):
        c = g.search(q.tolist(), 64)
        truth = np.argsort(np.linalg.norm(x - q, axis=1))[:10]
        hits.append(len(set(truth) & {a.id for a in c[:10]}) / 10)
        assert len(c) < 500
    assert np.mean(hits) > .9
    for i in range(40):
        g.repair(i)
        assert g.check_reverse()


def test_ambiguity_closed_overlap_and_ties():
    c = [_core.Candidate(i, d, 0, 1) for i, d in enumerate([1., 1.2, 1.4, 4.])]
    assert _core.ambiguity(c, 2, [.2]) == [1, 0, 2]
