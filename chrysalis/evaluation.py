import json
import math

import numpy as np

from . import conformal


def zipf_stream(n_queries, length, exponent, seed):
    """Seeded request stream over query indices with Zipf popularity; ranks are a random permutation."""
    rng = np.random.default_rng(seed)
    rank_of = rng.permutation(n_queries)
    p = 1.0 / np.arange(1, n_queries + 1) ** exponent
    return rank_of[rng.choice(n_queries, size=length, p=p / p.sum())]


def ground_truth(queries, corpus, k):
    """Exact kNN by brute force (Faiss; GPU when available)."""
    import faiss

    queries = np.ascontiguousarray(queries, np.float32)
    corpus = np.ascontiguousarray(corpus, np.float32)
    out = np.empty((len(queries), k), np.int64)
    step = 4096
    for s in range(0, len(queries), step):
        if faiss.get_num_gpus() > 0:
            _, out[s:s + step] = faiss.knn_gpu(faiss.StandardGpuResources(), queries[s:s + step], corpus, k)
        else:
            _, out[s:s + step] = faiss.knn(queries[s:s + step], corpus, k)
    return out


def request_row(q, res, oracle, truth, k):
    """Per-request metrics. `oracle` holds exact E_{v+1} vectors and is never seen by the index."""
    ids = np.asarray(res["ids"])[:k]
    full = len(set(ids.tolist()) & set(truth[:k].tolist())) / k
    row = dict(recall=full, epoch=int(res.get("epoch", 0)), requested=int(res.get("requested", 0)),
               resolved=int(res.get("resolved", 0)), waited=int(res.get("waited", 0)),
               latency=float(res.get("latency", 0.0)), cpu=float(res.get("cpu", 0.0)))
    cand = res.get("candidates")
    if cand is not None:
        d_star = np.linalg.norm(oracle[cand] - q, axis=1)
        exact_top = cand[np.lexsort((cand, d_star))[:k]]
        row["examined_recall"] = len(set(ids.tolist()) & set(exact_top.tolist())) / k
        row["capture"] = len(set(cand.tolist()) & set(truth[:k].tolist())) / k
        if res.get("bound") is not None:
            mask = res["in_view"] == 0
            row["bound"] = float(res["bound"])
            row["score"] = conformal.query_score(res["d_hat"][mask], d_star[mask], res["eps"][mask])
            row["lam"] = float(res["lam"])
    if "confidence" in res:
        row["confidence"] = float(res["confidence"])
    return row


def auroc(positive, score):
    """Area under the ROC curve for detecting `positive` from `score` (higher = more suspicious)."""
    from sklearn.metrics import roc_auc_score

    positive = np.asarray(positive, bool)
    if positive.all() or not positive.any():
        return math.nan
    return float(roc_auc_score(positive, score))


def summarize(rows, low=0.9):
    """Recall, mean bound, M1, M2, fully certified share, gap, Miss and AUROC."""
    if not rows:
        return dict(requests=0)
    out = dict(requests=len(rows), recall=float(np.mean([r["recall"] for r in rows])),
               resolution_rate=float(np.mean([r["requested"] > 0 for r in rows])),
               wait_rate=float(np.mean([r["resolved"] > 0 or r["waited"] > 0 for r in rows])))
    lat = np.array([r["latency"] for r in rows]) * 1000
    out.update(p50_ms=float(np.percentile(lat, 50)), p99_ms=float(np.percentile(lat, 99)))
    cpu = [r["cpu"] for r in rows if r["cpu"] > 0]
    if cpu:
        out["qps_per_core"] = float(1.0 / np.mean(cpu))
    cert = [r for r in rows if "bound" in r]
    if cert:
        b = np.array([r["bound"] for r in cert])
        full = np.array([r["recall"] for r in cert])
        exam = np.array([r["examined_recall"] for r in cert])
        out.update(mean_bound=float(b.mean()), M1=float(np.mean(exam < b - 1e-12)), M2=float(np.mean(full < b - 1e-12)),
                   fully_certified=float(np.mean(b >= 1.0)), gap=float(full.mean() - b.mean()),
                   miss=float(np.mean([r["score"] > r["lam"] for r in cert])),
                   auroc_certificate=auroc(full < low, -b), capture=float(np.mean([r["capture"] for r in cert])))
        conf = [r for r in cert if "confidence" in r]
        if conf:
            out["auroc_hmoe"] = auroc([r["recall"] < low for r in conf], [-r["confidence"] for r in conf])
    return out


class Recorder:
    def __init__(self, path):
        self.f = open(path, "a")
        self.rows = []

    def add(self, **row):
        self.rows.append(row)
        self.f.write(json.dumps(row) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()
