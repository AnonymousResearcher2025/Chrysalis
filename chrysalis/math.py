"""Numerical calibration. Arrays use row-vector convention x @ W + b."""
from dataclasses import dataclass
import math
import numpy as np


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError('nonfinite embedding')
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError('zero embedding cannot be unit normalized')
    return x / norms


def quantile(scores, alpha):
    if not 0 < alpha < 1:
        raise ValueError('alpha must be in (0,1)')
    s = np.sort(np.asarray(scores, dtype=np.float64))
    if s.ndim != 1 or np.any(s < 0) or np.any(~np.isfinite(s)):
        raise ValueError('nonconformity scores must be finite, nonnegative')
    rank = math.ceil((1 - alpha) * (len(s) + 1))
    return (float(s[rank - 1]) if rank <= len(s) else math.inf), rank


@dataclass
class Bridge:
    W: np.ndarray
    b: np.ndarray
    kind: str

    def __call__(self, x):
        return np.asarray(x) @ self.W + self.b


def fit_bridge(x, y, *, rank, ridge):
    """Procrustes or reduced-rank ridge: min ||Xc W-Yc||²+λ||W||², rank(W)≤r.

    Whiten X by A=(Xc'Xc+λI); truncate A^(-1/2)Xc'Yc, then unwhiten.
    This solves the regularized rank-constrained objective (not truncated OLS).
    Native inputs are normalized before this function; no whitening preprocessing
    other than the objective's solver; affine outputs remain unnormalized.
    """
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) < 2 or len(x) != len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('at least two finite paired fitting samples required')
    if x.shape[1] == y.shape[1]:
        u, _, vt = np.linalg.svd(x.T @ y, full_matrices=False)
        return Bridge((u @ vt).astype('float32'), np.zeros(y.shape[1], 'float32'), 'orthogonal')
    if ridge <= 0 or rank < 1:
        raise ValueError('positive ridge and rank required')
    xm, ym = x.mean(0), y.mean(0)
    xc, yc = x - xm, y - ym
    # Thin SVD avoids a d_old³ solve with only 200 fit samples/region.
    ux, sx, vx = np.linalg.svd(xc, full_matrices=False)
    z = (sx / np.sqrt(sx * sx + ridge))[:, None] * (ux.T @ yc)
    uz, sz, vz = np.linalg.svd(z, full_matrices=False)
    r = min(rank, len(sz), int(np.linalg.matrix_rank(xc)))
    w = (vx.T / np.sqrt(sx * sx + ridge)) @ (uz[:, :r] * sz[:r]) @ vz[:r]
    return Bridge(w.astype('float32'), (ym - xm @ w).astype('float32'), 'reduced-rank-ridge')


def calibrate_region(bridge, old, exact, queries, alpha, rng):
    estimated = bridge(old)
    queries = np.asarray(queries)
    if len(old) == 0:
        return dict(epsilon=math.inf, gamma=math.inf, distance_scores=[], vector_scores=[],
                    dbar=None, support=0, status='insufficient-support')
    if len(queries) == 0:
        raise ValueError('residual query pool is empty')
    q = queries[rng.integers(len(queries), size=len(old))]
    dstar = np.linalg.norm(q - exact, axis=1)
    s = np.abs(np.linalg.norm(q - estimated, axis=1) - dstar)
    v = np.linalg.norm(estimated - exact, axis=1)
    eps, order = quantile(s, alpha)
    gamma, _ = quantile(v, alpha)
    return dict(epsilon=eps, gamma=gamma, distance_scores=sorted(s.tolist()),
                vector_scores=sorted(v.tolist()), dbar=float(dstar.mean()),
                support=len(s), order=order,
                status='supported' if np.isfinite(eps) else 'insufficient-support')
