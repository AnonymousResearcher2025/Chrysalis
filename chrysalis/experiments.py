"""Experiments of Section 4. Every function writes per-request logs (JSON lines) and a
summary under its output directory; the summaries are computed only from those logs."""
import json
import math
import os
import shutil
import threading
import time
from pathlib import Path

import numpy as np

from . import _core, bridges, config, conformal, encoders, evaluation
from .baselines import (FastFillL2, HMoE, QueryAdapter, RankMerge, TranslatedIndex, backfill_order, build_graph,
                        mean_confidence)
from .data import read_queries
from .index import NATIVE, MigratingIndex
from .scheduler import Scheduler, TokenBucket, maintain
from .storage import RawStore
from .workers import LocalPool

FAMILIES = ("procrustes", "lowrank", "mlp")


class Pair:
    """Corpus, queries and cached embeddings of one model pair. The oracle (exact E_{v+1}
    vectors of the whole corpus) is used only for ground truth and metrics."""

    def __init__(self, name, data, work, cfg, devices=("cuda:0",), glove=None):
        self.models = config.models()
        self.name, self.spec, self.cfg = name, self.models["pairs"][name], cfg
        self.data, self.cache = Path(data), Path(work) / "cache" / name
        self.cache.mkdir(parents=True, exist_ok=True)
        self.devices, self.glove = list(devices), glove
        self.raw = RawStore(self.data / self.spec["corpus"])
        self.n = len(self.raw)
        self.query_sets = {"eval": read_queries(self.data / self.spec["queries"])}
        if name == "P2":
            self.query_sets["trec"] = read_queries(self.data / "trec-dl")
        new = self.models[self.spec["new"]]
        self.d_new, self.model_id = new["dim"], f'{new["id"]}@{new.get("revision", "")}'

    def encoder(self, which, ledger=None, device=None):
        return encoders.load(self.spec[which], self.models, device or self.devices[0], self.cfg.encode_batch,
                             ledger=ledger, glove_path=self.glove)

    def _cached(self, name, make):
        path = self.cache / f"{name}.npy"
        if not path.exists():
            np.save(path.with_suffix(".tmp.npy"), make())
            path.with_suffix(".tmp.npy").rename(path)
        return np.load(path, mmap_mode="r")

    def _corpus(self, which):
        enc = self.encoder(which)
        step = 1 << 16
        return np.concatenate([enc.encode(self.raw.get(range(s, min(self.n, s + step))))
                               for s in range(0, self.n, step)])

    def old_vectors(self):
        return self._cached("old", lambda: self._corpus("old"))

    def oracle(self):
        return self._cached("oracle", lambda: self._corpus("new"))

    def queries(self, which, qset="eval"):
        texts = [q["text"] for q in self.query_sets[qset]]
        return self._cached(f"queries-{qset}-{which}", lambda: self.encoder(which).encode(texts, role="query"))

    def truth(self, qset="eval"):
        return self._cached(f"truth-{qset}", lambda: evaluation.ground_truth(self.queries("new", qset), self.oracle(), self.cfg.k))

    def old_graph(self):
        path = self.cache / "old-graph.bin"
        if not path.exists():
            build_graph(np.asarray(self.old_vectors()), self.cfg).save(str(path))
        return _core.Graph.load(str(path))

    def fresh_graph(self):
        path = self.cache / "fresh-graph.bin"
        if not path.exists():
            build_graph(np.asarray(self.oracle()), self.cfg).save(str(path))
        return _core.Graph.load(str(path))

    def training_pairs(self, seed):
        """The region-stratified R x n_r sample every method pays for (Chrysalis fits on half of it)."""
        _, region = bridges.coarse_quantizer(np.asarray(self.old_vectors()), self.cfg.regions, seed)
        splits = bridges.stratified_sample(region, self.cfg.regions, self.cfg.items_per_region, self.cfg.fit_per_region, seed)
        return np.sort(np.concatenate([np.concatenate([f, c]) for f, c in splits])), region


class Run:
    """One Chrysalis migration: warm-up, calibration and rotation, ready to serve."""

    def __init__(self, pair, out, cal_seed, replay_seed, cfg=None, translator=None, force=False):
        self.pair, self.cfg = pair, cfg or pair.cfg
        self.out = Path(out)
        if self.out.exists():
            shutil.rmtree(self.out)
        self.out.mkdir(parents=True)
        MigratingIndex.create(self.out / "index", self.cfg, pair.old_graph(), np.asarray(pair.old_vectors()))
        self.ledger = encoders.Ledger(self.out / "ledger.tsv")
        encs = [pair.encoder("new", self.ledger, d) for d in pair.devices]
        self.pool = LocalPool(pair.raw, encs[0], encs[1:] or encs, self.cfg.encode_batch)
        self.ix = MigratingIndex(self.out / "index", self.cfg)
        self.ix.attach(self.pool)
        q_new, q_old, old = pair.queries("new"), pair.queries("old"), pair.old_graph()
        stream = evaluation.zipf_stream(len(q_new), self.cfg.stream_requests, self.cfg.zipf_exponent, replay_seed)
        warm, self.requests = stream[: self.cfg.warmup_requests], stream[self.cfg.warmup_requests:]
        self.ix.observe(warm.tolist(), q_new[warm])
        for r in warm:
            self.ix.add_heat(old.search(np.asarray(q_old[r]), self.cfg.ef_search)[0])
        self.ix.heat_on = False
        t0 = time.perf_counter()
        self.diagnosis = self.ix.upgrade(pair.d_new, pair.model_id, seed=cal_seed, translator=translator, force=force)
        self.calibration_seconds = time.perf_counter() - t0
        if hasattr(self.ix, "lam"):
            self.ix.rotate()
        self.serve_after = time.perf_counter() - t0
        self.calibration_gpu = self.ledger.seconds()

    def serve(self, requests, label, qset="eval", mode=None, drain=True, confidence=None, live_routing=False):
        q, truth, oracle = self.pair.queries("new", qset), self.pair.truth(qset), self.pair.oracle()
        rec = evaluation.Recorder(self.out / f"{label}.jsonl")
        for t, r in enumerate(requests):
            c0 = time.thread_time()
            res = self.ix.search(np.asarray(q[r]), key=(qset, int(r)), mode=mode)
            res["cpu"] = time.thread_time() - c0
            if confidence is not None:
                res["confidence"] = confidence(res["ids"])
            row = evaluation.request_row(np.asarray(q[r]), res, oracle, truth[r], self.cfg.k)
            rec.add(request=t, query=int(r), gpu=self.ledger.seconds(), native=self.ix.native_fraction(), **row)
            if drain and self.pool.drain() and live_routing:
                while self.ix.audit():
                    pass
                self.ix.close_epoch(calibrate=False)
            if (t + 1) % 100 == 0:
                maintain(self.ix)
        rec.close()
        return evaluation.summarize(rec.rows)


def _runs(pair, out, cfg=None):
    cfg = cfg or pair.cfg
    for c in cfg.calibration_seeds:
        for s in cfg.replay_seeds:
            yield c, s, Path(out) / f"cal{c}-replay{s}"


def _mean(summaries):
    summaries = [s for s in summaries if s.get("requests", 1)] or summaries
    keys = set.intersection(*(set(s) for s in summaries))
    return {k: float(np.nanmean([s[k] for s in summaries])) for k in sorted(keys)
            if isinstance(summaries[0][k], (int, float))}


def _save(out, name, obj):
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / name).write_text(json.dumps(obj, indent=2))
    return obj


def global_baseline(pair, seed, requests=None, qset="eval"):
    """Best of the three Drift-Adapter families, fitted on all R x n_r pairs, serving the old index."""
    ids, _ = pair.training_pairs(seed)
    old, new = np.asarray(pair.old_vectors())[ids], np.asarray(pair.oracle())[ids]
    q, truth, graph = pair.queries("new", qset), pair.truth(qset), pair.old_graph()
    requests = np.arange(len(q)) if requests is None else requests
    out = {}
    for fam in FAMILIES:
        ad = QueryAdapter(fam, new, old, pair.cfg.bridge_rank, seed, device=pair.devices[0])
        mapped = ad(np.asarray(q))
        rec = [len(set(graph.search(mapped[r], pair.cfg.ef_search)[0][: pair.cfg.k]) & set(truth[r])) / pair.cfg.k
               for r in requests]
        out[fam] = float(np.mean(rec))
    out["best"] = max(FAMILIES, key=lambda f: out[f])
    out["recall"] = out[out["best"]]
    return out


# ---------------------------------------------------------------- Table 3
def quality(pair, out):
    """Recall@10 and mean bound from the calibration seed to a fully native index (Table 3)."""
    rows = {k: [] for k in ("seed_off", "seed_res", "f25", "native")}
    glob = []
    for c, s, d in _runs(pair, out):
        run = Run(pair, d, c, s)
        rows["seed_off"].append(run.serve(run.requests, "seed_off", mode="off"))
        rows["seed_res"].append(run.serve(run.requests, "seed_res", mode="async"))
        Scheduler(run.ix, "utility").run(int(0.25 * pair.n))
        rows["f25"].append(run.serve(run.requests, "f25", mode="async"))
        Scheduler(run.ix, "utility").run(None)
        run.ix.retire()
        rows["native"].append(run.serve(run.requests, "native"))
        glob.append(global_baseline(pair, c, run.requests))
    return _save(out, "table3.json", dict(pair=pair.name, global_=_mean(glob), **{k: _mean(v) for k, v in rows.items()}))


# ---------------------------------------------------------------- Table 4 and Figure 2
class Method:
    """Interface of the matched-budget comparison."""

    def search(self, q):
        raise NotImplementedError

    def background(self, items):
        raise NotImplementedError


class ChrysalisMethod(Method):
    def __init__(self, run, mode):
        self.run, self.mode, self.sched = run, mode, Scheduler(run.ix, "utility")

    def search(self, q, key):
        res = self.run.ix.search(q, key=key, mode=self.mode)
        self.run.pool.drain()
        return res

    def background(self, items):
        n = self.sched.run(items)
        maintain(self.run.ix)
        return n


class ReembedMethod(Method):
    """Shared backfill driver for B1, B2 and B3."""

    def __init__(self, pair, ledger, order, add, search):
        self.pair, self.order, self.pos, self.add, self._search = pair, order, 0, add, search
        self.enc = pair.encoder("new", ledger)

    def search(self, q, key):
        return self._search(q)

    def background(self, items):
        ids = self.order[self.pos:self.pos + items]
        self.pos += len(ids)
        if len(ids):
            self.add(ids, self.enc.encode(self.pair.raw.get(ids), category="scheduled"))
        return len(ids)


def full_reembed_seconds(pair, out):
    """Busy GPU seconds of one full re-embedding (the budget unit)."""
    path = Path(out) / "fullreembed.json"
    if path.exists():
        return json.loads(path.read_text())["gpu_seconds"]
    ledger = encoders.Ledger()
    enc = pair.encoder("new", ledger)
    step = 1 << 16
    for s in range(0, pair.n, step):
        enc.encode(pair.raw.get(range(s, min(pair.n, s + step))), category="fullreembed")
    return _save(out, "fullreembed.json", dict(gpu_seconds=ledger.seconds()))["gpu_seconds"]


def _baseline(name, pair, out, seed):
    """Build B1, B2 or B3 after it has paid for the native seed and its training pairs."""
    ledger = encoders.Ledger(Path(out) / "ledger.tsv")
    ids, _ = pair.training_pairs(seed)
    enc = pair.encoder("new", ledger)
    new = enc.encode(pair.raw.get(ids), category="calibration")
    old = np.asarray(pair.old_vectors())
    t0 = time.perf_counter()
    if name == "B1":
        ad = max((QueryAdapter(f, new, old[ids], pair.cfg.bridge_rank, seed, pair.devices[0]) for f in FAMILIES),
                 key=lambda a: _adapter_fit(a, new, old[ids]))
        m = RankMerge(pair.cfg, pair.old_graph(), ad, pair.d_new, seed)
        m.add(ids, new)
        method = ReembedMethod(pair, ledger, backfill_order(pair.n, ids, seed=seed), m.add, m.search)
    else:
        model = HMoE(old[ids], new, seed, pair.devices[0]) if name == "B2" else FastFillL2(old[ids], new, seed, pair.devices[0])
        t = TranslatedIndex(pair.cfg, pair.old_graph(), model.map(old))
        t.replace(ids, new)
        prio = None if name == "B2" else model.uncertainty(old)
        method = ReembedMethod(pair, ledger, backfill_order(pair.n, ids, prio, seed), t.replace, t.search)
    method.training_seconds = time.perf_counter() - t0
    return method, ledger


def _adapter_fit(adapter, new, old):
    return -float(((adapter(new) - old) ** 2).sum(1).mean())


def budget_curve(pair, method, ledger, requests, full_seconds, out, label, step=None, probe=1000, stop=1.0):
    """Recall over a probe window of the stream after each increment of background work."""
    step = step or max(1, pair.n // 200)
    q, truth = pair.queries("new"), pair.truth()
    rec = evaluation.Recorder(Path(out) / f"{label}.jsonl")
    pos, done = 0, False
    while not done:
        window = [requests[(pos + j) % len(requests)] for j in range(probe)]
        pos += probe
        hits, bounds = [], []
        for r in window:
            res = method.search(np.asarray(q[r]), (("eval", int(r))))
            hits.append(len(set(np.asarray(res["ids"])[: pair.cfg.k].tolist()) & set(truth[r].tolist())) / pair.cfg.k)
            if res.get("bound") is not None:
                bounds.append(res["bound"])
        budget = ledger.seconds() / full_seconds
        rec.add(budget=budget, recall=float(np.mean(hits)), mean_bound=float(np.mean(bounds)) if bounds else None)
        done = budget >= stop or method.background(step) == 0
    rec.close()
    return rec.rows


def _at(curve, budget):
    b = np.array([c["budget"] for c in curve])
    r = np.array([c["recall"] for c in curve])
    return float(np.interp(budget, b, r)) if len(b) else math.nan


def _reach(curve, target, key="recall"):
    for a, c in zip(curve, curve[1:]):
        if a[key] is not None and a[key] >= target:
            return a["budget"]
        if c[key] is not None and c[key] >= target:
            w = (target - a[key]) / (c[key] - a[key]) if a[key] is not None and c[key] != a[key] else 1.0
            return float(a["budget"] + w * (c["budget"] - a["budget"]))
    return math.nan


def matched_budget(pair, out, targets=(0.98, 0.99)):
    """Recall at a 25 percent budget and budget to reach each target (Table 4, Figure 2)."""
    full = full_reembed_seconds(pair, out)
    res = {}
    for c, s, d in _runs(pair, out):
        for name in ("Chrysalis", "Chrysalis-resolution-off", "B1", "B2", "B2+", "B3"):
            path = d / name
            if name.startswith("Chrysalis"):
                run = Run(pair, path, c, s)
                method, ledger = ChrysalisMethod(run, "off" if name.endswith("off") else "async"), run.ledger
                requests = run.requests
            elif name == "B2+":
                ids, _ = pair.training_pairs(c)
                old = np.asarray(pair.old_vectors())
                tr = HMoE(old[ids], np.asarray(pair.oracle())[ids], c, pair.devices[0])
                run = Run(pair, path, c, s, cfg=pair.cfg.replace(resolution_mode="off"), translator=tr)
                method, ledger, requests = ChrysalisMethod(run, "off"), run.ledger, run.requests
            else:
                path.mkdir(parents=True, exist_ok=True)
                method, ledger = _baseline(name, pair, path, c)
                requests = evaluation.zipf_stream(len(pair.queries("new")), pair.cfg.stream_requests,
                                                  pair.cfg.zipf_exponent, s)[pair.cfg.warmup_requests:]
            curve = budget_curve(pair, method, ledger, requests, full, path, "curve")
            res.setdefault(name, []).append(dict(at25=_at(curve, 0.25), **{f"to{t}": _reach(curve, t) for t in targets},
                                                 bound095=_reach(curve, 0.95, "mean_bound")))
    return _save(out, "table4.json", dict(pair=pair.name, full_gpu_seconds=full, **{k: _mean(v) for k, v in res.items()}))


def plot_budget(out):
    """Figure 2 from the first run's curves of a matched-budget experiment."""
    first = sorted(Path(out).glob("cal*-replay*"))[0]
    curves = {}
    for name in ("Chrysalis", "Chrysalis-resolution-off", "B1", "B2", "B2+", "B3"):
        path = first / name / "curve.jsonl"
        if path.exists():
            curves[name] = [json.loads(line) for line in open(path)]
    figure2(curves, Path(out) / "figure2.pdf")
    return dict(figure=str(Path(out) / "figure2.pdf"))


def figure2(curves, path):
    """Recall against GPU work for each method; dashed: Chrysalis mean bound, circle at 0.95."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for name, rows in curves.items():
        ax.plot([r["budget"] for r in rows], [r["recall"] for r in rows], label=name, lw=1.2)
        if name == "Chrysalis":
            pts = [(r["budget"], r["mean_bound"]) for r in rows if r["mean_bound"] is not None]
            ax.plot(*zip(*pts), "--", color=ax.lines[-1].get_color(), lw=1)
            hit = next((p for p in pts if p[1] >= 0.95), None)
            if hit:
                ax.plot(*hit, "o", mfc="none", color=ax.lines[-1].get_color())
    ax.set_xlabel("GPU work (fraction of FullReembed)")
    ax.set_ylabel("Recall@10")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path)


# ---------------------------------------------------------------- Table 5
def coverage(pair, out, alphas=(0.10, 0.05, 0.01), region_counts=(1, 64, 256, 1024), held_out=200):
    """Per-comparison coverage of eps_r on held-out pairs and the region sweep (Table 5)."""
    cfg, old, oracle = pair.cfg, np.asarray(pair.old_vectors()), np.asarray(pair.oracle())
    q_new = pair.queries("new")
    res = dict(coverage={}, sweep={})
    for c in cfg.calibration_seeds:
        stream = evaluation.zipf_stream(len(q_new), cfg.stream_requests, cfg.zipf_exponent, c)[: cfg.warmup_requests]
        recent = np.asarray(q_new)[np.unique(stream)]
        rng = np.random.default_rng(c)
        for R in region_counts + ("1-all",):
            regions = 1 if R == "1-all" else R
            per = (cfg.regions * cfg.items_per_region) if R == "1-all" else cfg.items_per_region
            fit_n = per // 2
            _, region = bridges.coarse_quantizer(old, regions, c)
            splits = bridges.stratified_sample(region, regions, per + held_out, fit_n, c)
            widths, cov = [], {a: [] for a in alphas}
            for fit, rest in splits:
                cal, hold = rest[: per - fit_n], rest[per - fit_n:]
                br = bridges.fit_bridge(old[fit], oracle[fit], cfg.bridge_rank, cfg.ridge_grid, cfg.ridge_folds, c)
                qc = recent[rng.integers(len(recent), size=len(cal))]
                d_star = np.linalg.norm(qc - oracle[cal], axis=1)
                scores = np.abs(np.linalg.norm(qc - br(old[cal]), axis=1) - d_star)
                widths.append(conformal.quantile(scores, cfg.alpha) / d_star.mean())
                if R == cfg.regions:
                    qh = recent[rng.integers(len(recent), size=len(hold))]
                    err = np.abs(np.linalg.norm(qh - br(old[hold]), axis=1) - np.linalg.norm(qh - oracle[hold], axis=1))
                    for a in alphas:
                        cov[a].extend((err <= conformal.quantile(scores, a)).tolist())
            res["sweep"].setdefault(str(R), {}).setdefault("relative_width", []).append(float(np.mean(widths)))
            if R == cfg.regions:
                for a in alphas:
                    res["coverage"].setdefault(str(1 - a), []).append(float(np.mean(cov[a])))
    for R in region_counts:
        rates = []
        for c, s, d in _runs(pair, out, cfg):
            if s != cfg.replay_seeds[0]:
                continue
            run = Run(pair, d / f"R{R}", c, s, cfg=cfg.replace(regions=R))
            rates.append(run.serve(run.requests, "seed_res", mode="async")["resolution_rate"])
        res["sweep"][str(R)]["resolution_rate"] = rates
    res["coverage"] = {k: float(np.mean(v)) for k, v in res["coverage"].items()}
    res["sweep"] = {R: {k: float(np.mean(v)) for k, v in m.items()} for R, m in res["sweep"].items()}
    return _save(out, "table5.json", dict(pair=pair.name, **res))


# ---------------------------------------------------------------- Tables 6 and 7
def cold_queries(run, qset="eval"):
    """Queries whose exact top-10 lies in the least-visited 10 percent of regions by warm-up heat."""
    ix = run.ix
    heat = np.bincount(ix.region[: ix.n], weights=ix.heat_at()[: ix.n], minlength=ix.cfg.regions)
    cold = set(np.argsort(heat, kind="stable")[: max(1, ix.cfg.regions // 10)].tolist())
    truth = run.pair.truth(qset)
    return np.array([j for j in range(len(truth)) if all(int(ix.region[i]) in cold for i in truth[j])], np.int64)


def workload(run, name, seed):
    rng = np.random.default_rng(seed)
    if name == "zipf":
        return run.requests, "eval"
    if name == "uniform":
        return rng.permutation(len(run.pair.queries("new"))), "eval"
    if name == "cold":
        return rng.permutation(cold_queries(run)), "eval"
    if name == "trec":
        return rng.permutation(len(run.pair.queries("new", "trec"))), "trec"
    raise ValueError(name)


def hmoe_confidence(pair, run, seed):
    """HMoE's translation confidence over the returned items (native items count as 1)."""
    ids, _ = pair.training_pairs(seed)
    old = np.asarray(pair.old_vectors())
    from embedding_translation.evaluation import TranslationConfidence

    tc = TranslationConfidence.fit(old[ids])
    return lambda returned: mean_confidence(lambda x: tc.score(old[x]), np.asarray(returned), run.ix.state == NATIVE)


def reliability(pair, out, workloads, recalibrate=False):
    """Certificate reliability on served queries at f = 25 percent (Table 6)."""
    res = {}
    for c, s, d in _runs(pair, out):
        run = Run(pair, d, c, s)
        Scheduler(run.ix, "utility").run(int(0.25 * pair.n))
        maintain(run.ix)
        conf = hmoe_confidence(pair, run, c)
        for w in workloads:
            reqs, qset = workload(run, w, s)
            res.setdefault(w, []).append(run.serve(reqs, w, qset=qset, mode="async", confidence=conf))
        if recalibrate and "cold" in workloads:
            reqs, _ = workload(run, "cold", s + 1)
            head, tail = reqs[: pair.cfg.calibration_queries], reqs[pair.cfg.calibration_queries:]
            run.serve(head, "cold-head", mode="async")
            g0 = run.ledger.seconds()
            run.ix.close_epoch()
            gpu = run.ledger.seconds() - g0
            summ = run.serve(tail, "cold-recalibrated", mode="async")
            res.setdefault("cold-recalibrated", []).append(dict(gpu_minutes=gpu / 60, **summ))
    return _save(out, "table6.json", dict(pair=pair.name, **{k: _mean(v) for k, v in res.items()}))


VARIANTS = {
    "Chrysalis": {},
    "Live routing": {"live_routing": True},
    "First calibration": {"freeze_lambda": True},
    "No region scaling": {"region_scaling": False},
    "Count m only": {"count_rule": "m"},
}


def ablation(pair, out, workload_name):
    """Certificate ablation at f = 25 percent (Table 7)."""
    res = {}
    for c, s, d in _runs(pair, out):
        for name, flags in VARIANTS.items():
            run = Run(pair, d / name.replace(" ", "-"), c, s)
            live = flags.get("live_routing", False)
            for key, value in flags.items():
                if key != "live_routing":
                    setattr(run.ix, key, value)
            Scheduler(run.ix, "utility").run(int(0.25 * pair.n))
            maintain(run.ix)
            reqs, qset = workload(run, workload_name, s)
            res.setdefault(name, []).append(run.serve(reqs, workload_name, qset=qset, mode="async", live_routing=live))
    return _save(out, "table7.json", dict(pair=pair.name, workload=workload_name, **{k: _mean(v) for k, v in res.items()}))


# ---------------------------------------------------------------- Table 8
def _disk(path):
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.stat(os.path.join(root, name)).st_size
            except FileNotFoundError:
                pass
    return total


class DiskSampler(threading.Thread):
    def __init__(self, paths, every=1.0):
        super().__init__(daemon=True)
        self.paths, self.every, self.peak, self.stop = paths, every, 0, threading.Event()

    def sample(self):
        self.peak = max(self.peak, sum(_disk(p) for p in self.paths if Path(p).exists()))

    def run(self):
        while not self.stop.is_set():
            self.sample()
            self.stop.wait(self.every)


def cost(pair, out, qps=100.0, window=1000):
    """GPU time, peak storage and wall time of a live migration (Table 8). Background
    workers, the scheduler and a paced client run concurrently."""
    full = full_reembed_seconds(pair, out)
    pair.fresh_graph()
    native_bytes = os.path.getsize(pair.cache / "fresh-graph.bin")
    res = []
    for c, s, d in _runs(pair, out):
        if s != pair.cfg.replay_seeds[0]:
            continue
        t_start = time.perf_counter()
        sampler = DiskSampler([d / "index"])
        run = Run(pair, d, c, s)
        sampler.start()
        run.pool.start()
        sched, stop = Scheduler(run.ix, "utility"), threading.Event()

        def background():
            while not stop.is_set() and not run.ix.all_native():
                if sched.step() == 0:
                    time.sleep(0.05)
                maintain(run.ix)

        bg = threading.Thread(target=background, daemon=True)
        bg.start()
        q, bounds, hit = pair.queries("new"), [], None
        for t, r in enumerate(np.resize(run.requests, 10 * len(run.requests))):
            due = t_start + run.serve_after + t / qps
            time.sleep(max(0.0, due - time.perf_counter()))
            res_q = run.ix.search(np.asarray(q[r]), key=("eval", int(r)))
            bounds.append(res_q["bound"] if res_q["bound"] is not None else 1.0)
            if hit is None and len(bounds) >= window and np.mean(bounds[-window:]) >= 0.95:
                hit = dict(gpu=run.ledger.seconds(), minutes=(time.perf_counter() - t_start) / 60)
            if run.ix.all_native():
                break
        stop.set()
        bg.join()
        run.pool.close()
        Scheduler(run.ix, "utility").run(None)
        run.ix.retire()
        sampler.stop.set()
        sampler.join()
        res.append(dict(to_bound_gpu_hours=hit["gpu"] / 3600 if hit else math.nan,
                        to_bound_fraction=hit["gpu"] / full if hit else math.nan,
                        to_bound_minutes=hit["minutes"] if hit else math.nan,
                        to_native_gpu_hours=run.ledger.seconds() / 3600, to_native_ratio=run.ledger.seconds() / full,
                        peak_storage=sampler.peak / native_bytes, serve_after_minutes=run.serve_after / 60))
    dual = dual_index(pair, out)
    return _save(out, "table8.json", dict(pair=pair.name, fullreembed_gpu_hours=full / 3600,
                                          dual_peak_storage=dual["peak"] / native_bytes,
                                          dual_minutes=dual["minutes"], **_mean(res)))


def dual_index(pair, out):
    """DualIndex: re-embed everything with every worker beside the old index, build the
    second graph, then switch the alias."""
    d = Path(out) / "dualindex"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    MigratingIndex.create(d / "old", pair.cfg, pair.old_graph(), np.asarray(pair.old_vectors()))
    sampler = DiskSampler([d])
    sampler.start()
    t0 = time.perf_counter()
    encs = [pair.encoder("new", None, dev) for dev in pair.devices]
    parts = np.array_split(np.arange(pair.n), len(encs))
    vecs = [None] * len(encs)

    def work(j):
        out_j = np.lib.format.open_memmap(d / f"new-{j}.npy", "w+", np.float32, (len(parts[j]), pair.d_new))
        for s in range(0, len(parts[j]), 1 << 16):
            ids = parts[j][s:s + (1 << 16)]
            out_j[s:s + len(ids)] = encs[j].encode(pair.raw.get(ids))
        out_j.flush()
        vecs[j] = out_j

    ts = [threading.Thread(target=work, args=(j,)) for j in range(len(encs))]
    [t.start() for t in ts]
    [t.join() for t in ts]
    build_graph(np.concatenate(vecs), pair.cfg).save(str(d / "new-graph.bin"), with_vectors=False)
    minutes = (time.perf_counter() - t0) / 60
    sampler.sample()
    sampler.stop.set()
    sampler.join()
    shutil.rmtree(d)
    return dict(peak=sampler.peak, minutes=minutes)


# ---------------------------------------------------------------- Table 9
def scheduler_ablation(pair, out, orders=("random", "heat", "radius", "utility")):
    """GPU fraction of FullReembed until the mean bound reaches 0.95, resolution disabled (Table 9)."""
    full = full_reembed_seconds(pair, out)
    res = {}
    for c, s, d in _runs(pair, out):
        for order in orders:
            run = Run(pair, d / order, c, s, cfg=pair.cfg.replace(resolution_mode="off"))
            m = ChrysalisMethod(run, "off")
            m.sched = Scheduler(run.ix, order, seed=s)
            curve = budget_curve(pair, m, run.ledger, run.requests, full, d / order, "curve")
            res.setdefault(order, []).append(dict(bound095=_reach(curve, 0.95, "mean_bound")))
    return _save(out, "table9.json", dict(pair=pair.name, **{k: _mean(v) for k, v in res.items()}))


# ---------------------------------------------------------------- Table 10
def latency(pair, out, requests=20000, night_dollars=1.0):
    """Index-only search latency on one core at saturation, after rotation (Table 10)."""
    os.sched_setaffinity(threading.get_native_id(), {0})
    q, res = pair.queries("new"), {}
    g = pair.fresh_graph()
    reqs = evaluation.zipf_stream(len(q), requests, pair.cfg.zipf_exponent, pair.cfg.replay_seeds[0])
    lat, cpu = [], []
    for r in reqs:
        t, c = time.perf_counter(), time.thread_time()
        g.search(np.asarray(q[r]), pair.cfg.ef_search)
        lat.append(time.perf_counter() - t)
        cpu.append(time.thread_time() - c)
    res["native"] = dict(p50_ms=1000 * float(np.percentile(lat, 50)), p99_ms=1000 * float(np.percentile(lat, 99)),
                         qps_per_core=1 / float(np.mean(cpu)))
    c, s = pair.cfg.calibration_seeds[0], pair.cfg.replay_seeds[0]
    for mode in ("async", "sync"):
        run = Run(pair, Path(out) / f"latency-{mode}", c, s, cfg=pair.cfg.replace(resolution_mode=mode))
        run.pool.start()
        res[f"{mode}_hour0"] = run.serve(run.requests[:requests], f"{mode}-hour0", mode=mode, drain=False)
        if mode == "sync":
            res["sync_hour0"]["distinct_wait_share"] = _distinct_wait(Path(out) / f"latency-{mode}" / "sync-hour0.jsonl")
            spot = pair.cfg.gpu_price_per_hour["spot"]
            sched = Scheduler(run.ix, "utility", TokenBucket(night_dollars, 0.0, spot), run.ledger)
            while sched.step() or len(run.pool.queues):
                maintain(run.ix)
                time.sleep(0.01)
            res["sync_night"] = run.serve(run.requests[:requests], "sync-night", mode=mode, drain=False)
            res["sync_night"]["night_gpu_hours"] = run.ledger.seconds() / 3600
        run.pool.close()
    return _save(out, "table10.json", dict(pair=pair.name, **res))


def _distinct_wait(path):
    waited = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            waited[r["query"]] = waited.get(r["query"], False) or r["resolved"] > 0 or r["waited"] > 0
    return float(np.mean(list(waited.values())))


# ---------------------------------------------------------------- Section 4.6
def fallback(pair, out):
    """Calibration diagnostic on a cross-family pair (P4)."""
    run = Run(pair, Path(out) / "fallback", pair.cfg.calibration_seeds[0], pair.cfg.replay_seeds[0])
    return _save(out, "fallback.json", dict(pair=pair.name, gpu_hours=run.ledger.seconds() / 3600, **run.diagnosis))


def rare_entities(pair, out):
    """Least frequent tenth of entities (passage titles) in Wiki 1M; queries whose exact
    nearest passage belongs to one of them."""
    titles = [json.loads(line)["title"] for line in open(pair.raw.dir / "corpus.jsonl", encoding="utf-8")]
    _, inv, counts = np.unique(titles, return_inverse=True, return_counts=True)
    rare = set(np.nonzero(counts <= np.quantile(counts, 0.1, method="lower"))[0].tolist())
    truth = pair.truth()
    queries = np.array([j for j in range(len(truth)) if inv[truth[j][0]] in rare], np.int64)
    res = dict(queries=len(queries), global_=[], seed=[], resolution=[])
    for c, s, d in _runs(pair, out):
        if s != pair.cfg.replay_seeds[0]:
            continue
        res["global_"].append(global_baseline(pair, c, queries)["recall"])
        run = Run(pair, d, c, s)
        res["seed"].append(run.serve(queries, "rare-off", mode="off")["recall"])
        res["resolution"].append(run.serve(queries, "rare-async", mode="async")["recall"])
    return _save(out, "rare.json", {k: (float(np.mean(v)) if isinstance(v, list) else v) for k, v in res.items()})


def structure_repair(pair, out):
    """Recall with exact distances: stale graph, auditor-repaired graph, fresh graph."""
    q, truth, k, ef = pair.queries("new"), pair.truth(), pair.cfg.k, pair.cfg.ef_search
    oracle = np.asarray(pair.oracle())

    def recall(search):
        return float(np.mean([len(set(search(np.asarray(q[r]))[: k].tolist()) & set(truth[r].tolist())) / k
                              for r in range(len(q))]))

    stale = pair.old_graph().with_vectors(oracle)
    fresh = pair.fresh_graph()
    c, s = pair.cfg.calibration_seeds[0], pair.cfg.replay_seeds[0]
    run = Run(pair, Path(out) / "repair", c, s)
    t0 = time.process_time()
    audit_cpu = [0.0]
    original = run.ix.audit

    def timed(limit=64):
        a = time.process_time()
        n = original(limit)
        audit_cpu[0] += time.process_time() - a
        return n

    run.ix.audit = timed
    Scheduler(run.ix, "utility").run(None)
    p99 = {}
    for busy in (False, True):
        stop = threading.Event()

        def keep_auditing():
            while not stop.is_set() and run.ix.audit():
                pass

        if busy:
            run.ix.audit_queue.extend(range(pair.n))
            worker = threading.Thread(target=keep_auditing, daemon=True)
            worker.start()
        lat = [run.ix.search(np.asarray(q[r]), observe=False)["latency"] for r in run.requests[:5000]]
        stop.set()
        if busy:
            worker.join()
        p99["auditing" if busy else "idle"] = 1000 * float(np.percentile(lat, 99))
    run.ix.retire()
    view = run.ix.holder.current()
    old = pair.old_graph()
    changed = sum(len(set(view.neighbors(i)) - set(old.neighbors(i))) for i in range(pair.n))
    total = sum(len(view.neighbors(i)) for i in range(pair.n))
    return _save(out, "repair.json", dict(
        pair=pair.name, stale=recall(lambda x: stale.search(x, ef)[0]), fresh=recall(lambda x: fresh.search(x, ef)[0]),
        repaired=recall(lambda x: run.ix.search(x, observe=False)["ids"]), rewritten_edge_fraction=changed / total,
        p99_ms_idle=p99["idle"], p99_ms_auditing=p99["auditing"],
        auditor_cpu_hours=audit_cpu[0] / 3600, total_cpu_hours=(time.process_time() - t0) / 3600))


# ---------------------------------------------------------------- Table 11
def faults(pair, out, args, migrations=5, index_kills=(("rotate", 2), ("switch", 2), ("stage", 2)), spot_losses=3,
           partitions=1, gap=60.0, host_addr="127.0.0.1:7100", sync_addr="127.0.0.1:7101"):
    """Fault injection over live migrations with a separate index host and GPU worker
    processes: index kills at rotation, epoch switch and edge staging, spot workers
    terminated without notice, and workers partitioned until their leases expire."""
    import signal
    import subprocess
    import sys

    from .workers import Client

    base = [sys.executable, "-m", "chrysalis.cli"]
    shared = ["--pair", pair.name, "--data", str(pair.data), "--work", str(args.work), "--config", args.config]
    for item in args.set or []:
        shared += ["--set", item]
    spot = [d for d in args.spot.split(",") if d] or pair.devices[1:] or pair.devices
    lease = pair.cfg.lease_seconds
    pair.old_vectors(), pair.old_graph(), pair.queries("new"), pair.queries("old")
    rows = []
    for m in range(migrations):
        d = Path(out) / f"migration{m}"
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        logs = []

        def start_worker(dev, sync=False):
            log = d / f"worker{len(logs)}.log"
            logs.append(log)
            cmd = base + ["worker"] + shared + ["--host", host_addr, "--devices", dev, "--log", str(log)]
            if sync:
                cmd += ["--sync", "--address", sync_addr]
            return subprocess.Popen(cmd), log

        def start_host(recover):
            cmd = base + ["host"] + shared + ["--index", str(d / "index"), "--address", host_addr,
                                              "--on-demand", sync_addr, "--seed", str(m + 1)]
            return subprocess.Popen(cmd + (["--recover"] if recover else []))

        sync_w, _ = start_worker(pair.devices[0], sync=True)
        workers = [start_worker(dev) for dev in spot]
        hostp, client = start_host(False), Client(host_addr)
        while not (d / "index" / "ready").exists():
            time.sleep(1)
        q = pair.queries("new")
        events = sorted([("index", p) for p, c in index_kills for _ in range(c)], key=lambda e: e[1] != "rotate")
        events += [("spot", None)] * spot_losses + [("partition", None)] * partitions
        last_start, partitioned = time.time(), []
        for kind, point in events:
            if kind == "index":
                (d / "index" / "crash").write_text(point)
                try:
                    hostp.wait(timeout=1800)
                except subprocess.TimeoutExpired:
                    (d / "index" / "crash").unlink(missing_ok=True)
                    rows.append(dict(fault=f"index:{point}", migration=m, injected=False))
                    continue
                (d / "index" / "ready").unlink(missing_ok=True)
                t0, last_start = time.perf_counter(), time.time()
                hostp = start_host(True)
                while True:
                    try:
                        client.call("Search", q=np.asarray(q[0]), timeout=1)
                        break
                    except Exception:
                        time.sleep(0.05)
                rows.append(dict(fault=f"index:{point}", migration=m, recovery_seconds=time.perf_counter() - t0))
                if point == "rotate":
                    continue
            elif kind == "spot":
                proc, _ = workers.pop(0)
                proc.send_signal(signal.SIGKILL)
                proc.wait()
                time.sleep(5)
                workers.append(start_worker(spot[len(workers) % len(spot)]))
                rows.append(dict(fault="spot", migration=m))
            else:
                proc, log = workers[0]
                proc.send_signal(signal.SIGSTOP)
                t_stop = time.time()
                time.sleep(lease + 5)
                proc.send_signal(signal.SIGCONT)
                partitioned.append((log, t_stop))
                rows.append(dict(fault="partition", migration=m))
            time.sleep(gap)
        while not client.call("Status", timeout=10)["converged"]:
            time.sleep(5)
        time.sleep(lease + 1)
        status = client.call("Status", timeout=10)
        for p in [hostp, sync_w] + [w[0] for w in workers]:
            p.send_signal(signal.SIGKILL)
            p.wait()
        encoded, acked = [], set()
        for log in logs:
            if log.exists():
                for line in open(log):
                    rec = json.loads(line)
                    if "encoded" in rec and rec["t"] > last_start:
                        encoded += rec["encoded"]
                    elif "acked" in rec and rec["accepted"]:
                        acked.update(rec["acked"])
        stale = 0
        for log, t_stop in partitioned:
            lines = [json.loads(line) for line in open(log)]
            for a, b in zip(lines, lines[1:]):
                if "encoded" in a and a["t"] < t_stop and "acked" in b and b["t"] > t_stop + lease:
                    stale += int(b["accepted"] > 0)
        final = MigratingIndex.open(d / "index", pair.cfg, LocalPool(pair.raw, None))
        rows.append(dict(fault="audit", migration=m, lost=int(sum(final.state[i] != NATIVE for i in acked)),
                         stranded=int(status["resolving"]), encoded=len(encoded), unique=len(set(encoded)),
                         expired_leases=int(status["expired_leases"]), stale_accepted=stale,
                         stale_rejected=int(status["stale_rejected"])))
        _save(d, "faults.json", rows)
    return _save(out, "table11.json", _fault_table(rows))


def _fault_table(rows):
    rec = [r["recovery_seconds"] for r in rows if "recovery_seconds" in r]
    audits = [r for r in rows if r["fault"] == "audit"]
    repeats = sum(a["encoded"] - a["unique"] for a in audits)
    leases = sum(a["expired_leases"] for a in audits)
    return dict(index_killed=dict(n=len(rec), median_s=float(np.median(rec)) if rec else math.nan,
                                  max_s=float(np.max(rec)) if rec else math.nan),
                spot_lost=sum(r["fault"] == "spot" for r in rows), partitioned=sum(r["fault"] == "partition" for r in rows),
                lost=sum(a["lost"] for a in audits), stranded=sum(a["stranded"] for a in audits),
                stale_accepted=sum(a["stale_accepted"] for a in audits),
                repeats_per_expired_lease=repeats / leases if leases else math.nan, audits=audits)
