"""Independent evaluator: ground-truth vectors never passed to serving modules."""
import json
from pathlib import Path
import time
import numpy as np


class Oracle:
    def __init__(self, vectors):
        self.vectors = np.asarray(vectors, dtype='float64')

    def ranking(self, query, ids=None, k=10):
        ids = np.arange(len(self.vectors)) if ids is None else np.asarray(ids, dtype=int)
        d = np.linalg.norm(self.vectors[ids] - np.asarray(query, dtype='float64'), axis=1)
        return ids[np.lexsort((ids, d))[:k]].tolist()

    def assess(self, query, result, k):
        returned = set(result['results'])
        full = self.ranking(query, k=k)
        examined = self.ranking(query, result['candidate_ids'], k=k)
        rf, rc = len(returned & set(full)) / k, len(returned & set(examined)) / k
        bound = result.get('bound')
        return dict(recall_full=rf, recall_examined=rc,
                    sound_full=None if bound is None else rf + 1e-12 >= bound,
                    sound_examined=None if bound is None else rc + 1e-12 >= bound)


def zipf_replay(query_count, length, seed):
    rng = np.random.default_rng(seed)
    ranks = rng.permutation(query_count)
    p = 1 / np.arange(1, query_count + 1, dtype='float64')
    return ranks[rng.choice(query_count, size=length, p=p / p.sum())]


def replay(query_raw, query_ids, encoder, serve, oracle, output, *, k, length, seed, after_query=None, label='', observer=None):
    order = zipf_replay(len(query_raw), length, seed)
    rows = []
    replay_started = time.perf_counter()
    with Path(output).open('w', encoding='utf-8') as f:
        for t, qi in enumerate(order):
            start = time.perf_counter()
            q = encoder.encode([query_raw[qi]], role='query', category='query')[0]
            encoded = time.perf_counter()
            result = serve(q)
            finish = time.perf_counter()
            row = dict(label=label, seed=seed, sequence=t, query_id=query_ids[qi], **result,
                       latency_ms=(finish - start) * 1000, index_latency_ms=(finish - encoded) * 1000,
                       query_encoding_ms=(encoded - start) * 1000, **oracle.assess(q, result, k))
            row['replay_wall_elapsed_seconds'] = finish - replay_started
            f.write(json.dumps(row) + '\n')
            f.flush()
            rows.append(row)
            if observer:
                observer(row)
            if after_query:
                after_query()
    return summary(rows)


def summary(rows):
    if not rows:
        return dict(queries=0)
    index_latencies = [r['index_latency_ms'] for r in rows]
    bounds = [r['bound'] for r in rows if r.get('bound') is not None]
    elapsed_by_seed = {}
    for row in rows:
        if 'replay_wall_elapsed_seconds' in row:
            elapsed_by_seed[row['seed']] = max(elapsed_by_seed.get(row['seed'], 0),
                                              row['replay_wall_elapsed_seconds'])
    observed_seconds = sum(elapsed_by_seed.values())
    result = dict(queries=len(rows), recall_full=float(np.mean([r['recall_full'] for r in rows])),
                  recall_examined=float(np.mean([r['recall_examined'] for r in rows])),
                  mean_bound=float(np.mean(bounds)) if bounds else None,
                  p50_index_ms=float(np.quantile(index_latencies, .5)), p99_index_ms=float(np.quantile(index_latencies, .99)),
                  qps_index=1000 / np.mean(index_latencies),
                  qps_observed=len(rows) / observed_seconds if observed_seconds else None,
                  observed_replay_seconds=observed_seconds or None,
                  p50_end_to_end_ms=float(np.quantile([r['latency_ms'] for r in rows], .5)),
                  resolution_rate=float(np.mean([r.get('requested', 0) > 0 for r in rows])),
                  native_fraction=float(np.mean([r.get('native_fraction', 0) for r in rows])),
                  sound_full=float(np.mean([r['sound_full'] for r in rows if r['sound_full'] is not None])) if bounds else None,
                  sound_examined=float(np.mean([r['sound_examined'] for r in rows if r['sound_examined'] is not None])) if bounds else None,
                  snapshot_valid_queries=sum(r.get('snapshot_applicable', False) for r in rows))
    return result


def graph_serve(graph, k, ef, adapter=None):
    def serve(q):
        query = adapter(np.asarray([q]))[0] if adapter else q
        c = graph.search(query.tolist(), ef)
        return dict(results=[x.id for x in c[:k]], candidate_ids=[x.id for x in c],
                    candidate_count=len(c), bound=None, snapshot_applicable=False, requested=0)
    return serve


def tables(run_dir):
    """Generate table analogues strictly from raw outputs, never paper constants."""
    import csv
    root = Path(run_dir)
    groups = {}
    for path in root.rglob('queries-*.jsonl'):
        for line in path.read_text().splitlines():
            r = json.loads(line)
            groups.setdefault(r['label'], []).append(r)
    rows = [dict(configuration=label, **summary(values)) for label, values in sorted(groups.items())]
    out = root / 'tables'
    out.mkdir(exist_ok=True)
    def write(name, data):
        if not data:
            (out / name).write_text('status\nnot_run\n')
            return
        with (out / name).open('w', newline='') as f:
            fields = sorted({k for r in data for k in r})
            writer = csv.DictWriter(f, fields)
            writer.writeheader()
            writer.writerows(data)
    write('table2-quality.csv', rows)
    coverage = json.loads((root / 'offline-coverage.json').read_text()) if (root / 'offline-coverage.json').exists() else []
    resolution_rows = groups.get('SeedAsync', [])
    if resolution_rows:
        for row in coverage:
            row['resolution_rate'] = float(np.mean([r.get('requested', 0) > 0 for r in resolution_rows]))
    write('table3-coverage.csv', coverage)
    economics = json.loads((root / 'economics.json').read_text()) if (root / 'economics.json').exists() else []
    write('table4-economics.csv', economics)
    write('table5-serving.csv', [dict(configuration=r['configuration'], p50_index_ms=r['p50_index_ms'],
                                     p99_index_ms=r['p99_index_ms'], index_service_rate_per_second=r['qps_index'],
                                     qps_observed=r['qps_observed'],
                                     resolution_rate=r['resolution_rate']) for r in rows])
    (out / 'README.txt').write_text('New measured outputs. Index latency excludes query encoding; end-to-end latency includes it.\n'
                                    'qps_observed divides query count by measured sequential replay elapsed time, including query encoding and intervening local queue drains.\n'
                                    'index_service_rate_per_second is reciprocal mean index latency, not observed throughput.\n'
                                    'This single-client smoke is not a saturated single-core QPS reproduction.\n')
    return rows
