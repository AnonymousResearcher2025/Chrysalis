"""Baselines: Global (Drift-Adapter-style query map), graph builds for FullReembed and DualIndex,
B1 (rank merge of Seo et al.), B2 and B2+ (HMoE), and B3 (label-free FastFill)."""
import numpy as np

from . import _core


def _pad(x, d):
    x = np.asarray(x, np.float32)
    return x if x.shape[1] == d else np.concatenate([x, np.zeros((len(x), d - x.shape[1]), np.float32)], 1)


def _train(model, x, y, loss_fn, seed, epochs=50, lr=3e-4, weight_decay=0.01, batch=256, patience=5, device="cpu"):
    """AdamW with early stopping on a 10 percent validation split."""
    import torch

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(x), generator=g)
    n_val = max(1, len(x) // 10)
    x = torch.as_tensor(x, dtype=torch.float32, device=device)
    y = torch.as_tensor(y, dtype=torch.float32, device=device)
    val, tr = perm[:n_val], perm[n_val:]
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        order = tr[torch.randperm(len(tr), generator=g)]
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            opt.zero_grad()
            loss_fn(model, x[idx], y[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(loss_fn(model, x[val], y[val]))
        if v < best - 1e-7:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model.eval()


def _mse(model, x, y):
    return ((model(x) - y) ** 2).sum(1).mean()


def _modules():
    import torch
    from torch import nn

    class LowRank(nn.Module):
        def __init__(self, d, rank):
            super().__init__()
            self.V = nn.Linear(d, rank, bias=False)
            self.U = nn.Linear(rank, d)
            self.s = nn.Parameter(torch.ones(d))

        def forward(self, x):
            return self.s * self.U(self.V(x))

    class ResidualMLP(nn.Module):
        def __init__(self, d, hidden=256, dropout=0.1):
            super().__init__()
            self.f = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, d))
            self.s = nn.Parameter(torch.ones(d))

        def forward(self, x):
            return self.s * (x + self.f(x))

    class Uncertain(nn.Module):
        def __init__(self, d, hidden=1024):
            super().__init__()
            self.f = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, d))
            self.var = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, 1), nn.Softplus())

        def forward(self, x):
            return x + self.f(x)

    return LowRank, ResidualMLP, Uncertain


class QueryAdapter:
    """Drift-Adapter families mapping E_{v+1} queries into the E_v space, in a zero-padded
    common width: orthogonal Procrustes, low-rank affine and residual MLP (both with a
    diagonal scaling), trained on item pairs."""

    def __init__(self, family, new_items, old_items, rank=64, seed=0, device="cpu"):
        import torch

        self.family, self.d_old = family, old_items.shape[1]
        self.d = max(new_items.shape[1], old_items.shape[1])
        b, a = _pad(new_items, self.d), _pad(old_items, self.d)
        if family == "procrustes":
            u, _, vt = np.linalg.svd(b.astype(np.float64).T @ a.astype(np.float64))
            self.R = (u @ vt).astype(np.float32)
            return
        LowRank, ResidualMLP, _ = _modules()
        torch.manual_seed(seed)
        model = LowRank(self.d, rank) if family == "lowrank" else ResidualMLP(self.d)
        self.model = _train(model, b, a, _mse, seed, device=device)
        self.device = device

    def __call__(self, q):
        import torch

        q = _pad(np.atleast_2d(q), self.d)
        if self.family == "procrustes":
            out = q @ self.R
        else:
            with torch.no_grad():
                out = self.model(torch.as_tensor(q, device=self.device)).cpu().numpy()
        return out[:, : self.d_old].astype(np.float32)


def build_graph(vectors, cfg, seed=0):
    g = _core.Graph(vectors.shape[1], cfg.max_degree, cfg.ef_construction, cfg.prune_alpha, seed)
    g.build(np.ascontiguousarray(vectors, np.float32), cfg.build_threads)
    return g


class RankMerge:
    """B1: the old index serves through the global map while a second index backfills;
    both distance-ranked lists are merged, preferring the backfilled copy of an item."""

    def __init__(self, cfg, old_graph, adapter, d_new, seed=0):
        self.cfg, self.old, self.adapter = cfg, old_graph, adapter
        self.new = _core.Graph(d_new, cfg.max_degree, cfg.ef_construction, cfg.prune_alpha, seed)
        self.ids = []

    def add(self, ids, vectors):
        for i, v in zip(ids, vectors):
            self.new.insert(np.ascontiguousarray(v, np.float32))
            self.ids.append(int(i))

    def search(self, q):
        k = self.cfg.k
        a_ids, a_d = self.old.search(self.adapter(q)[0], self.cfg.ef_search)
        best = {int(i): float(d) for i, d in zip(a_ids[:k], a_d[:k])}
        if self.ids:
            b_ids, b_d = self.new.search(q, self.cfg.ef_search)
            for j, d in zip(b_ids[:k], b_d[:k]):
                best[self.ids[j]] = float(d)
        merged = sorted(best.items(), key=lambda t: (t[1], t[0]))[:k]
        return dict(ids=np.array([i for i, _ in merged]))


class TranslatedIndex:
    """B2 and B3: every stored vector is translated into the new space, re-embedded items
    replace their translation through the graph's insert path, as in FreshDiskANN."""

    def __init__(self, cfg, old_graph, translated):
        self.cfg = cfg
        self.graph = old_graph.with_vectors(np.ascontiguousarray(translated, np.float32))
        self.native = np.zeros(len(translated), bool)

    def replace(self, ids, vectors):
        for i, v in zip(ids, vectors):
            self.graph.reinsert(int(i), np.ascontiguousarray(v, np.float32))
            self.native[int(i)] = True

    def search(self, q):
        ids, _ = self.graph.search(q, self.cfg.ef_search)
        return dict(ids=ids[: self.cfg.k])


class HMoE:
    """Hierarchical mixture of LoRA experts (Yang and Cao, 2026) through its reference
    implementation, with the paper's canonical settings, plus its translation confidence."""

    def __init__(self, old_items, new_items, seed=0, device="cuda"):
        from embedding_translation.config import GatingMoEConfig, MappingConfig, SimpleLinearMapperConfig
        from embedding_translation.evaluation import TranslationConfidence
        from embedding_translation.mapper.hmoe import HMoEMapper

        cfg = MappingConfig(device=device, hmoe_config=GatingMoEConfig(
            moe_type="hierarchical_lora", num_levels=3, branch_factor=2, lora_rank=8, alpha=0.5, beta=0.7, tau=0.8,
            random_state=seed, mapper_config=SimpleLinearMapperConfig(activation="selu", layer_num=4)))
        self.mapper = HMoEMapper(cfg)
        self.mapper.fit(np.asarray(old_items, np.float32), np.asarray(new_items, np.float32),
                        reference_indices=np.arange(len(old_items)))
        self.tc = TranslationConfidence.fit(np.asarray(old_items, np.float32))

    def map(self, x, batch=65536):
        x = np.asarray(x, np.float32)
        return np.concatenate([np.asarray(self.mapper.transform(x[s:s + batch]), np.float32)
                               for s in range(0, len(x), batch)]) if len(x) else np.zeros((0, 0), np.float32)

    def confidence(self, x):
        return np.asarray(self.tc.score(np.asarray(x, np.float32)), np.float64)


class FastFillL2:
    """Label-free FastFill: an old-to-new MLP trained with the uncertainty-weighted L2 loss
    ||h(x) - y||^2 / sigma^2 + d log sigma^2 (Jaeckle et al., 2023, Eq. 4 with 1/lambda = d);
    items are backfilled in decreasing predicted uncertainty."""

    def __init__(self, old_items, new_items, seed=0, device="cuda"):
        import torch

        _, _, Uncertain = _modules()
        self.d = max(old_items.shape[1], new_items.shape[1])
        self.d_new, self.device = new_items.shape[1], device
        torch.manual_seed(seed)
        dim = self.d

        def loss(model, x, y):
            var = model.var(x)[:, 0] + 1e-6
            return (((model(x) - y) ** 2).sum(1) / var + dim * torch.log(var)).mean()

        self.model = _train(Uncertain(self.d), _pad(old_items, self.d), _pad(new_items, self.d), loss, seed, device=device)

    def _run(self, x, head, batch=65536):
        import torch

        out = []
        with torch.no_grad():
            for s in range(0, len(x), batch):
                t = torch.as_tensor(_pad(x[s:s + batch], self.d), device=self.device)
                out.append((self.model(t)[:, : self.d_new] if head == "map" else self.model.var(t)[:, 0]).cpu().numpy())
        return np.concatenate(out)

    def map(self, x):
        return self._run(np.asarray(x, np.float32), "map").astype(np.float32)

    def uncertainty(self, x):
        return self._run(np.asarray(x, np.float32), "var")


def backfill_order(n, seed_items, priority=None, seed=0):
    """Remaining items in random order, or in decreasing `priority`."""
    rest = np.setdiff1d(np.arange(n), seed_items)
    if priority is None:
        return np.random.default_rng(seed).permutation(rest)
    return rest[np.lexsort((rest, -priority[rest]))]


def mean_confidence(conf_of, returned, native):
    """Per-query HMoE confidence: mean translation confidence over the returned items,
    with native items counted as fully confident."""
    vals = np.ones(len(returned))
    mask = ~native[returned]
    if mask.any():
        vals[mask] = conf_of(returned[mask])
    return float(vals.mean())

