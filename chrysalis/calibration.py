import hashlib
import uuid
import numpy as np
from sklearn.cluster import KMeans
from .math import fit_bridge, calibrate_region, quantile, normalize


def partition(old, R, seed):
    if R < 1 or R > 65536 or len(old) < R:
        raise ValueError('regions must fit uint16 and not exceed corpus size')
    km = KMeans(n_clusters=R, n_init=10, random_state=seed).fit(old)
    return km.labels_.astype('uint16'), km.cluster_centers_.astype('float32')


def sample_regions(regions, fit_count, residual_count, offline_count, seed, region_count=None):
    """Disjoint corpus identity splits; offline items never reach the encoder cache."""
    rng = np.random.default_rng(seed)
    output = []
    for r in range(region_count if region_count is not None else int(max(regions)) + 1):
        ids = rng.permutation(np.flatnonzero(regions == r))
        # Proportional downscaling documented; no pooling across drift regions.
        sizes = np.array([fit_count, residual_count, offline_count])
        if len(ids) < sizes.sum():
            sizes = np.floor(len(ids) * sizes / sizes.sum()).astype(int)
            sizes[0] += len(ids) - sizes.sum()
        a, b = sizes[0], sizes[0] + sizes[1]
        output.append(dict(fit=ids[:a].tolist(), residual=ids[a:b].tolist(), offline=ids[b:b + sizes[2]].tolist()))
    return output


def regional(old, regions, samples, encode_items, residual_queries, *, rank, ridge, alpha, seed):
    rng = np.random.default_rng(seed)
    maps, retained = [], {}
    for r, split in enumerate(samples):
        fi, ri = split['fit'], split['residual']
        if not fi and not ri and not split['offline']:
            maps.append(dict(W=np.zeros((old.shape[1], residual_queries.shape[1])).tolist(),
                             b=np.zeros(residual_queries.shape[1]).tolist(), kind='unsupported-empty-region',
                             epsilon=float('inf'), gamma=float('inf'), distance_scores=[], vector_scores=[],
                             dbar=None, support=0, status='empty-region'))
            continue
        if len(fi) < 2:
            raise RuntimeError(f'region {r}: insufficient fit support; merge/repartition explicitly or FullReembed')
        operational = fi + ri
        exact = encode_items(operational)
        retained.update(zip(operational, exact))
        bridge = fit_bridge(old[fi], exact[:len(fi)], rank=rank, ridge=ridge)
        stats = calibrate_region(bridge, old[ri], exact[len(fi):], residual_queries, alpha, rng)
        maps.append(dict(W=bridge.W.tolist(), b=bridge.b.tolist(), kind=bridge.kind, **stats))
    return maps, retained


def diagnostic(maps, tau):
    widths = [m['epsilon'] / m['dbar'] if m['dbar'] and m['dbar'] > 0 else float('inf') for m in maps]
    median = float(np.median(widths))
    return dict(median_relative_width=median, widths=widths, recommend_full_reembed=median > tau,
                minority_wide_regions=sum(w > tau for w in widths), threshold=tau)


def query_epoch(index, queries, query_ids, encode_items, alpha_q, ef, *, change_threshold):
    if len(queries) != len(query_ids) or len(set(query_ids)) != len(query_ids):
        raise ValueError('unique calibration query identities required')
    if not 0 < change_threshold <= 1:
        raise ValueError('epoch threshold must be (0,1]')
    # Service-wide lock holds frozen routing while scoring. Retained exact vectors
    # are durable but invisible to graph until all scores and epoch metadata sync.
    with index.lock:
        graph, fingerprint = index.frozen()
        existing = index.store.get('epoch_scoring')
        # An interrupted epoch is restarted in full; retained encoder outputs reused.
        index.store.transaction({'epoch_scoring': dict(snapshot=fingerprint, query_ids=query_ids)})
        cache = {int(i): index.store.vector(n) for i, n in index.store.get('retained', {}).items()}
        scores, visited = [], []
        for q in queries:
            q = normalize([q])[0]
            candidates = graph.search(q.tolist(), ef)
            unresolved = [c.id for c in candidates if c.state != 3 and c.id not in cache]
            if unresolved:
                exact = encode_items(unresolved)
                index.retain(dict(zip(unresolved, exact)))
                cache.update(zip(unresolved, exact))
            maximum = 0.
            for c in candidates:
                if c.state != 3:
                    exact_distance = np.linalg.norm(q.astype('float64') - cache[c.id].astype('float64'))
                    maximum = max(maximum, abs(c.d - exact_distance))
            scores.append(float(maximum))
            visited.append([c.id for c in candidates])
        if index.fingerprint() != fingerprint:
            raise RuntimeError('routing changed during calibration')
        radius, rank = quantile(scores, alpha_q)
        meta = index.store.get('meta')
        epoch = dict(id=uuid.uuid4().hex, snapshot=fingerprint, epsilon_cert=radius, alpha_q=alpha_q,
                     scores=scores, rank=rank, query_ids=query_ids, visited=visited,
                     query_hash=hashlib.sha256(np.asarray(queries).tobytes()).hexdigest(),
                     baseline_flips=meta['flips'], baseline_edge_changes=meta['edge_changes'],
                     change_threshold=change_threshold, resolution_during_replay=False)
        index.store.publish({'epoch': epoch, 'epoch_scoring': None})
        return epoch


def epoch_due(index):
    e, m = index.store.get('epoch'), index.store.get('meta')
    if m['retired']:
        return False
    if not e:
        return True
    flips = (m['flips'] - e['baseline_flips']) / max(1, m['count'])
    edges = sum(len(layer) for n in index.graph.topology()['edges'] for layer in n)
    rewrites = (m['edge_changes'] - e['baseline_edge_changes']) / max(1, edges)
    return flips >= e['change_threshold'] or rewrites >= e['change_threshold']
