import dataclasses

import numpy as np

from . import _core


def coarse_quantizer(old, regions, seed, train_per_region=256):
    import faiss

    old = np.ascontiguousarray(old, dtype=np.float32)
    rng = np.random.default_rng(seed)
    sample = old[rng.choice(len(old), min(len(old), regions * train_per_region), replace=False)]
    km = faiss.Kmeans(old.shape[1], regions, niter=25, seed=seed, max_points_per_centroid=train_per_region)
    km.train(sample)
    _, assign = km.index.search(old, 1)
    return km.centroids.copy(), assign[:, 0].astype(np.uint16)


def stratified_sample(assign, regions, per_region, fit_per_region, seed):
    """Per region: (fitting ids, calibration ids), drawn without replacement."""
    rng = np.random.default_rng(seed)
    order = np.argsort(assign, kind="stable")
    bounds = np.searchsorted(assign[order], np.arange(regions + 1))
    out = []
    for r in range(regions):
        members = order[bounds[r]:bounds[r + 1]]
        take = rng.permutation(members)[:per_region]
        cut = int(round(len(take) * fit_per_region / per_region))
        out.append((np.sort(take[:cut]), np.sort(take[cut:])))
    return out


@dataclasses.dataclass
class Bridge:
    kind: str
    V: np.ndarray = None
    U: np.ndarray = None
    W: np.ndarray = None
    b: np.ndarray = None
    ridge: float = None

    def core(self):
        if self.kind == "procrustes":
            return _core.Map.dense(self.W, self.b)
        return _core.Map.lowrank(self.V, self.U, self.b)

    def __call__(self, x):
        return self.core().apply(np.ascontiguousarray(x, dtype=np.float32))


def _reduced_rank_ridge(x, y, rank, ridge):
    xm, ym = x.mean(0), y.mean(0)
    xc, yc = x - xm, y - ym
    ux, sx, vx = np.linalg.svd(xc, full_matrices=False)
    shrink = 1.0 / np.sqrt(sx * sx + ridge)
    z = (sx * shrink)[:, None] * (ux.T @ yc)
    uz, sz, vz = np.linalg.svd(z, full_matrices=False)
    r = min(rank, len(sz))
    V = (vx.T * shrink) @ (uz[:, :r] * sz[:r])
    U = vz[:r].T
    b = ym - (xm @ V) @ U.T
    return V, U, b


def fit_bridge(x, y, rank, ridge_grid, folds, seed):
    """Orthogonal Procrustes when widths match, else a rank-limited ridge affine map
    whose penalty is chosen by k-fold cross-validation on the fitting pairs."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape[1] == y.shape[1]:
        u, _, vt = np.linalg.svd(x.T @ y)
        return Bridge("procrustes", W=(u @ vt).astype(np.float32), b=np.zeros(y.shape[1], np.float32))
    folds = min(folds, len(x))
    split = np.array_split(np.random.default_rng(seed).permutation(len(x)), folds)
    best = None
    for lam in ridge_grid:
        err = 0.0
        for f in range(folds):
            test = split[f]
            train = np.concatenate([split[g] for g in range(folds) if g != f])
            V, U, b = _reduced_rank_ridge(x[train], y[train], rank, lam)
            err += float(((x[test] @ V @ U.T + b - y[test]) ** 2).sum())
        if best is None or err < best[0]:
            best = (err, lam)
    V, U, b = _reduced_rank_ridge(x, y, rank, best[1])
    return Bridge("lowrank", V=V.astype(np.float32), U=U.astype(np.float32), b=b.astype(np.float32), ridge=best[1])


def save(path, bridges):
    arrays = {}
    for r, br in enumerate(bridges):
        arrays[f"{r}/kind"] = np.array(br.kind)
        for name in ("V", "U", "W", "b"):
            if getattr(br, name) is not None:
                arrays[f"{r}/{name}"] = getattr(br, name)
        if br.ridge is not None:
            arrays[f"{r}/ridge"] = np.array(br.ridge)
    np.savez(path, **arrays)


def load(path, regions):
    z = np.load(path)
    out = []
    for r in range(regions):
        kw = {k: z[f"{r}/{k}"] for k in ("V", "U", "W", "b") if f"{r}/{k}" in z}
        ridge = float(z[f"{r}/ridge"]) if f"{r}/ridge" in z else None
        out.append(Bridge(str(z[f"{r}/kind"]), ridge=ridge, **kw))
    return out
