import math

import numpy as np


def quantile(scores, alpha):
    """The ceil((1 - alpha)(m + 1))-th smallest score, or +inf if that rank exceeds m."""
    s = np.sort(np.asarray(scores, dtype=np.float64))
    rank = math.ceil((1 - alpha) * (len(s) + 1))
    return float(s[rank - 1]) if rank <= len(s) else math.inf


def region_radii(bridge, old, exact, queries, alpha, rng):
    """Distance radius eps_r, vector radius gamma_r and mean exact distance dbar_r
    of one region; each calibration item is scored against one recent query."""
    if len(old) == 0:
        return dict(eps=math.inf, gamma=math.inf, dbar=math.nan, m=0)
    mapped = bridge(old)
    q = queries[rng.integers(len(queries), size=len(old))]
    d_star = np.linalg.norm(q - exact, axis=1)
    scores = np.abs(np.linalg.norm(q - mapped, axis=1) - d_star)
    return dict(eps=quantile(scores, alpha), gamma=quantile(np.linalg.norm(mapped - exact, axis=1), alpha),
                dbar=float(d_star.mean()), m=len(old))


def diagnostic(radii, tau):
    widths = np.array([r["eps"] / r["dbar"] if r["m"] else math.inf for r in radii])
    median = float(np.median(widths))
    return dict(median_relative_width=median, recommend_full_reembed=bool(median > tau))


def normalized_error(d_hat, d_star, eps):
    """|d_hat - d*| / eps with an infinite radius contributing zero; a zero radius
    contributes zero only for an exact match."""
    err = np.abs(np.asarray(d_hat, np.float64) - np.asarray(d_star, np.float64))
    eps = np.asarray(eps, np.float64)
    out = np.zeros_like(err)
    finite = np.isfinite(eps) & (eps > 0)
    out[finite] = err[finite] / eps[finite]
    out[(eps == 0) & (err > 0)] = math.inf
    return out


def query_score(d_hat, d_star, eps):
    """s_j: largest region-normalized error over a query's candidates bridged in the view (0 if none)."""
    if len(d_hat) == 0:
        return 0.0
    return float(normalized_error(d_hat, d_star, eps).max())


def widths(eps, scale):
    """Interval half-widths scale * eps, unbounded when either factor is infinite."""
    eps = np.asarray(eps, np.float32)
    if math.isinf(scale):
        return np.full_like(eps, np.inf)
    with np.errstate(invalid="ignore"):
        return np.where(np.isinf(eps), np.float32(np.inf), eps * np.float32(scale)).astype(np.float32)
