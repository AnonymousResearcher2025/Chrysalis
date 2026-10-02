"""Baselines train only on explicitly supplied operational paired samples."""
import numpy as np
from . import _core
from .math import fit_bridge, normalize


def noadapt_maps(new_dim, old_dim, seed):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(new_dim, old_dim)).astype('float32') / np.sqrt(new_dim)
    trunc = np.zeros((new_dim, old_dim), dtype='float32')
    np.fill_diagonal(trunc, 1)
    return {'projection': lambda q: normalize(np.asarray(q) @ w),
            'truncation': lambda q: normalize(np.asarray(q) @ trunc)}


def global_adapter(old_pairs, new_pairs, *, rank, ridge, seed, mlp_steps=200):
    """Choose query-side adapter by disjoint paired validation MSE.

    Candidate families are paper-specified; optimizer and selection criterion are
    reconstruction choices. All families share the same paired-sample budget.
    No evaluation query, full-corpus oracle, or regional native seed is used.
    """
    import torch
    rng = np.random.default_rng(seed)
    ids = rng.permutation(len(old_pairs))
    cut = max(2, int(.8 * len(ids)))
    train, val = ids[:cut], ids[cut:]
    if len(val) == 0:
        raise ValueError('adapter validation pairs required')
    x, y = new_pairs, old_pairs
    width = max(x.shape[1], y.shape[1])
    pad = lambda a: np.pad(a, ((0, 0), (0, width - a.shape[1])))
    u, _, vt = np.linalg.svd(pad(x[train]).T @ pad(y[train]), full_matrices=False)
    wp = (u @ vt)[:x.shape[1], :y.shape[1]].astype('float32')
    affine = fit_bridge(x[train], y[train], rank=rank, ridge=ridge)
    families = {'padded_procrustes': lambda q: np.asarray(q) @ wp,
                'low_rank_affine': affine}
    torch.manual_seed(seed)
    # Fixed affine residual path + 2-layer nonlinear residual MLP.
    model = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 128), torch.nn.GELU(),
                                torch.nn.Linear(128, y.shape[1]))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    tx, target = torch.from_numpy(x[train].astype('float32')), torch.from_numpy(y[train].astype('float32'))
    base = torch.from_numpy(affine(x[train]).astype('float32'))
    model.train()
    for _ in range(mlp_steps):
        opt.zero_grad()
        loss = torch.mean((base + model(tx) - target) ** 2)
        loss.backward()
        opt.step()
    model.eval()
    def residual(q):
        with torch.inference_mode():
            return affine(q) + model(torch.from_numpy(np.asarray(q, dtype='float32'))).numpy()
    families['residual_mlp'] = residual
    scores = {name: float(np.mean((normalize(fn(x[val])) - y[val]) ** 2)) for name, fn in families.items()}
    selected = min(scores, key=lambda name: (scores[name], name))
    return lambda q: normalize(families[selected](q)), dict(selected=selected, paired_validation_mse=scores,
                                                           sample_count=len(ids), training_steps=mlp_steps)


class DualIndex:
    def __init__(self, old_graph, corpus_count, graph_parameters):
        self.old = old_graph
        self.new = _core.Graph(**graph_parameters)
        self.count, self.done, self.cutover = corpus_count, 0, False
        self.native_vectors = []

    def backfill(self, raw, encoder, batch_size=32, persist=None):
        for start in range(0, len(raw), batch_size):
            vectors = encoder.encode(raw[start:start + batch_size], category='dualindex')
            if persist:
                persist(vectors, start)
            self.new.append(vectors.tolist(), [0] * len(vectors))
            self.native_vectors.extend(vectors.tolist())
            self.done += len(vectors)
        if self.done != self.count:
            raise RuntimeError('incomplete backfill')
        self.cutover = True # caller durably publishes alias, only after graph sync
        return self.new


def full_reembed(raw, encoder, parameters, persist=None):
    vectors = encoder.encode(raw, category='fullreembed')
    if persist:
        persist(vectors)
    graph = _core.Graph(**parameters)
    graph.build(vectors.tolist(), [0] * len(vectors))
    return graph, vectors
