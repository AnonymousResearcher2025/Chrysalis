import json
from pathlib import Path
import threading
import time
import uuid
import numpy as np
from . import _core
from .math import normalize
from .storage import Store, dumps

STATES = {'legacy': 0, 'bridged': 1, 'resolving': 2, 'native': 3}


class Index:
    """Single-host index service; durable Store is authoritative over C++ mirror."""
    def __init__(self, root, crash=None):
        self.store = Store(root, crash)
        self.lock = self.store.lock
        self.graph = None
        if self.store.get('meta'):
            self.store.recover()
            self.reload()

    def reload(self):
        meta = self.store.get('meta')
        self.graph = _core.Graph(**meta['graph_parameters'])
        ns = self.store.nodes()
        self.graph.restore([self.store.vector(n).tolist() for n in ns], [n['region'] for n in ns],
                           [STATES[n['state']] for n in ns], [n['fence'] for n in ns], self.store.get('topology'))
        maps = self.store.get('bridges')
        self._bridges = maps
        self.radii = [m['epsilon'] for m in maps] if maps else [0.] * (1 + max((n['region'] for n in ns), default=0))
        if maps:
            self.graph.bridges([m['W'] for m in maps], [m['b'] for m in maps],
                               [m['epsilon'] for m in maps], [m['gamma'] for m in maps])

    @classmethod
    def create(cls, root, vectors, regions, raw, version, segment_size=4096, graph_parameters=None):
        obj = cls(root)
        if obj.store.get('meta'):
            obj.close()
            raise FileExistsError(root)
        if segment_size < 1 or len(vectors) != len(raw) or len(regions) != len(raw):
            raise ValueError('corpus lengths/segment size')
        x = normalize(vectors)
        parameters = graph_parameters or dict(M=32, efConstruction=200, alpha=1.2, seed=42)
        g = _core.Graph(**parameters)
        g.build(x.tolist(), list(map(int, regions)))
        updates = {}
        for start in range(0, len(x), segment_size):
            name = obj.store.write_array(x[start:start + segment_size], 'legacy')
            for row, i in enumerate(range(start, min(start + segment_size, len(x)))):
                updates[f'node/{i}'] = dict(file=name, row=row, region=int(regions[i]), state='legacy',
                                            version=version, fence=0, owner=None, expires=0, origin='legacy')
        rawpath = obj.store.root / 'raw.jsonl'
        import os
        with rawpath.open('w', encoding='utf-8') as f:
            for i, item in enumerate(raw):
                f.write(dumps(dict(id=i, version=version, item=item)) + '\n')
            f.flush()
            os.fsync(f.fileno())
        updates['meta'] = dict(count=len(x), old_version=version, new_version=None, retired=False,
                               graph_parameters=parameters, segment_size=segment_size, flips=0,
                               edge_changes=0, migration_started=False, index_id=uuid.uuid4().hex, routing_revision=0,
                               origin_counts={'legacy': len(x)})
        updates['topology'] = g.topology()
        updates['heat'] = []
        obj.store.publish(updates)
        obj.graph = g
        obj._bridges = None
        obj.radii = [0.] * (1 + max(map(int, regions), default=0))
        return obj

    def configure(self, new_version, bridges):
        with self.lock:
            meta = self.store.get('meta')
            if meta['migration_started']:
                raise RuntimeError('chained upgrades unsupported; direct re-embedding required')
            if not new_version or new_version == meta['old_version']:
                raise ValueError('successor version required')
            for n in self.store.nodes():
                if n['region'] >= len(bridges):
                    raise ValueError('missing region bridge')
            meta.update(new_version=new_version, migration_started=True)
            meta['routing_revision'] = meta.get('routing_revision', 0) + 1
            self.store.publish({'meta': meta, 'bridges': bridges})
            self.reload()

    def rotate_one(self):
        """One segment frontier, with atomic metadata batch and native race exclusion."""
        with self.lock:
            if self.store.get('epoch_scoring'):
                raise RuntimeError('frozen calibration in progress')
            ns = self.store.nodes()
            legacy = [n for n in ns if n['state'] == 'legacy']
            if not legacy:
                return False
            if not self._bridges:
                raise RuntimeError('configure bridges before rotation')
            source = legacy[0]['file']
            ids = [i for i, n in enumerate(ns) if n['file'] == source]
            # graph.vector transforms only LEGACY; native/bridged are never transformed twice.
            maps = self._bridges
            x = [None] * len(ids)
            # Dense region GEMMs within the bounded segment, no corpus-width copy.
            for region in sorted({ns[i]['region'] for i in ids}):
                rows = [row for row, i in enumerate(ids) if ns[i]['region'] == region and ns[i]['state'] == 'legacy']
                if rows:
                    old = np.asarray([self.store.vector(ns[ids[row]]) for row in rows])
                    mapped = old @ np.asarray(maps[region]['W'], dtype='float32') + np.asarray(maps[region]['b'], dtype='float32')
                    for row, value in zip(rows, mapped):
                        x[row] = value.tolist()
            for row, i in enumerate(ids):
                if x[row] is None:
                    x[row] = self.store.vector(ns[i]).tolist()
            path = self.store.write_array(x, 'rotated')
            updates = {}
            for row, i in enumerate(ids):
                n = ns[i]
                n.update(file=path, row=row)
                if n['state'] == 'legacy':
                    n.update(state='bridged', version=self.store.get('meta')['new_version'])
                updates[f'node/{i}'] = n
            meta = self.store.get('meta')
            meta['routing_revision'] = meta.get('routing_revision', 0) + 1
            updates['meta'] = meta
            def durable():
                self.store.publish(updates)
                return True
            self.graph.publish(ids, x, [STATES[updates[f'node/{i}']['state']] for i in ids],
                               [updates[f'node/{i}']['fence'] for i in ids], durable, self.store.reclaim)
            return True

    def rotate(self):
        while self.rotate_one():
            pass

    def raw(self, i):
        # Corpus cache belongs to worker, immutable and versioned. Large deployments
        # should supply the S3Raw backend; this local implementation reads JSONL.
        if not hasattr(self, '_raw'):
            self._raw = [json.loads(line) for line in (self.store.root / 'raw.jsonl').read_text(encoding='utf-8').splitlines()]
        r = self._raw[i]
        if r['version'] != self.store.get('meta')['old_version']:
            raise RuntimeError('raw version mismatch')
        return r['item']

    def claim(self, i, owner, lease=120, now=None):
        with self.lock:
            if self.store.get('epoch_scoring'):
                raise RuntimeError('frozen calibration in progress')
            # Materialize this legacy candidate only, rather than blocking an
            # async query on the entire preceding streaming frontier.
            n = self.store.node(i)
            if n['state'] == 'legacy':
                if not self._bridges:
                    raise RuntimeError('configure successor before resolution')
                x = self.graph.vector(i)
                path = self.store.write_array([x], 'on-demand-bridge')
                n.update(file=path, row=0, state='bridged', version=self.store.get('meta')['new_version'])
                meta = self.store.get('meta')
                meta['routing_revision'] += 1
                def durable():
                    self.store.publish({f'node/{i}': n, 'meta': meta})
                    return True
                self.graph.publish([i], [x], [1], [n['fence']], durable, self.store.reclaim)
            claim = self.store.claim(i, owner, lease, now)
            if claim:
                n = self.store.node(i)
                self.graph.update(i, self.store.vector(n).tolist(), 2, n['fence'])
            return claim

    def publish_native(self, claim, vector, origin, now=None):
        x = normalize([vector])[0]
        with self.lock:
            if self.store.get('epoch_scoring'):
                raise RuntimeError('frozen calibration in progress')
            expected = len(self.graph.vector(claim['id']))
            if len(x) != expected:
                raise ValueError('native embedding dimension differs from successor space')
            return self.graph.publish([claim['id']], [x.tolist()], [3], [claim['fence']],
                                      lambda: self.store.native(claim, x, origin, now, reclaim=False), self.store.reclaim)

    def acquire(self, claim, worker, seconds=120, now=None):
        with self.lock:
            if self.store.get('epoch_scoring'):
                raise RuntimeError('frozen calibration in progress')
            c = self.store.acquire(claim, worker, seconds, now)
            if c:
                n = self.store.node(c['id'])
                self.graph.update(c['id'], self.store.vector(n).tolist(), 2, c['fence'])
            return c

    def retain(self, exact):
        with self.lock:
            retained = self.store.get('retained', {})
            for i, x in exact.items():
                if str(i) not in retained and self.store.node(i)['state'] != 'native':
                    file = self.store.write_array([normalize([x])[0]], 'calibration')
                    retained[str(i)] = dict(file=file, row=0)
            self.store.transaction({'retained': retained})

    def retained(self, i):
        n = self.store.get('retained', {}).get(str(i))
        return None if n is None else self.store.vector(n)

    def publish_retained(self):
        if self.store.get('epoch_scoring'):
            raise RuntimeError('premature native publication during frozen replay')
        for key, n in list(self.store.get('retained', {}).items()):
            i = int(key)
            c = self.claim(i, 'calibration', 3600)
            if c:
                self.publish_native(c, self.store.vector(n), 'seed')
            if self.store.node(i)['state'] == 'native':
                r = self.store.get('retained', {})
                r.pop(key, None)
                self.store.transaction({'retained': r})
        self.store.reclaim()
        meta = self.store.get('meta')
        if 'initial_seed_count' not in meta:
            meta['initial_seed_count'] = sum(n['origin'] == 'seed' for n in self.store.nodes())
            self.store.transaction({'meta': meta})

    def frozen(self):
        """Routing copy independent of mutable store. Durable replay marker blocks writes."""
        with self.lock:
            ns = self.store.nodes()
            g = _core.Graph(**self.store.get('meta')['graph_parameters'])
            g.restore([self.graph.vector(i) for i in range(len(ns))], [n['region'] for n in ns],
                      [3 if n['state'] == 'native' else 1 for n in ns], [n['fence'] for n in ns], self.graph.topology())
            return g, self.fingerprint()

    def fingerprint(self):
        # Durable routing revision avoids a corpus scan per query. Claim ownership
        # doesn't increment it: resolving retains the same published vector.
        m = self.store.get('meta')
        return m['index_id'] + ':' + str(m['routing_revision'])

    def search(self, query, k=10, ef=96, rho=4, mode='async', resolver=None, enqueue=None, timeout=120):
        if mode not in ('sync', 'async', 'none') or rho < 0:
            raise ValueError('resolution mode/budget')
        q = normalize([query])[0].tolist()
        with self.lock:
            if self.store.get('epoch_scoring'):
                raise RuntimeError('calibration epoch is scoring; service paused')
            meta = self.store.get('meta')
            if meta['retired']:
                candidate = self.graph.search(q, ef)
                origins = meta.get('origin_counts', {})
                denominator = meta['count']
                if self.store.get('config', {}).get('fraction_denominator') == 'non-seed':
                    denominator -= meta.get('initial_seed_count', 0)
                return dict(results=[c.id for c in candidate[:k]], candidate_ids=[c.id for c in candidate],
                            candidate_count=len(candidate), bound=None, m=None, level=None, epoch_id=None,
                            snapshot_applicable=False, applicability='native-exact-ranking', scope='native-ANN',
                            insufficient_candidates=len(candidate) < k, native_fraction=1., requested=0, resolved=0,
                            origin_counts=origins,
                            scheduled_fraction=(origins.get('scheduled', 0) + origins.get('expedited', 0)) / max(1, denominator),
                            seed_fraction=origins.get('seed', 0) / max(1, meta['count']),
                            query_resolved_fraction=origins.get('query', 0) / max(1, meta['count']))
            candidate = self.graph.search(q, ef)
            ids = [x.id for x in candidate]
            heat_updates = {}
            for i in ids:
                heat_updates[f'heat/{i}'] = self.store.get(f'heat/{i}', 0) + 1
            self.store.transaction(heat_updates)
            eps = self.radii
            ambiguous = _core.ambiguity(candidate, k, eps)[:rho] if mode != 'none' else []
            fingerprint = self.fingerprint()
        requested, published = 0, 0
        for i in ambiguous:
            if mode == 'async':
                if enqueue is None:
                    raise ValueError('async requires a durable queue')
                claim = self.claim(i, uuid.uuid4().hex)
                if claim:
                    enqueue(dict(claim=claim, origin='query'))
                    requested += 1
            else:
                if resolver is None:
                    raise ValueError('sync requires a resolver')
                deadline = time.monotonic() + timeout
                while self.store.node(i)['state'] != 'native':
                    if time.monotonic() > deadline:
                        break # explicit unresolved interval preserved on timeout
                    claim = self.claim(i, uuid.uuid4().hex)
                    if claim:
                        requested += 1
                        x = self.retained(i)
                        if x is None:
                            x = resolver(i)
                        published += int(self.publish_native(claim, x, 'query'))
                    else:
                        time.sleep(.005)
        with self.lock:
            # C is fixed from traversal. Resolution does not retraverse the graph.
            if mode == 'sync':
                candidate = self.graph.rescore(q, ids)
            epoch = self.store.get('epoch')
            radius = epoch['epsilon_cert'] if epoch else float('inf')
            cert = _core.certificate(candidate, k, radius)
            applicable = bool(epoch and fingerprint == epoch['snapshot'] and self.fingerprint() == fingerprint)
            cert.update(epoch_id=epoch['id'] if epoch else None, level=1 - epoch['alpha_q'] if epoch else None,
                        snapshot_applicable=applicable, scope='examined-set',
                        applicability='snapshot-marginal' if applicable else 'evolving-empirical',
                        resolved=published, requested=requested, candidate_ids=ids,
                        results=[x.id for x in candidate[:k]],
                        candidate_intervals=[dict(id=x.id, estimate=x.d, native=x.state == 3,
                                                 lower=_core.interval(x, radius)[0], upper=_core.interval(x, radius)[1]) for x in candidate],
                        native_fraction=self.store.get('meta')['flips'] / max(1, self.store.get('meta')['count']))
            if self.store.get('meta')['retired']:
                cert.update(bound=None, m=None, applicability='native-exact-ranking', epoch_id=None, level=None)
            meta = self.store.get('meta')
            origins = meta.get('origin_counts', {})
            denominator = meta['count']
            if self.store.get('config', {}).get('fraction_denominator') == 'non-seed':
                denominator -= meta.get('initial_seed_count', 0)
            cert.update(origin_counts=dict(origins), scheduled_fraction=(origins.get('scheduled', 0) + origins.get('expedited', 0)) / max(1, denominator),
                        seed_fraction=origins.get('seed', 0) / max(1, meta['count']),
                        query_resolved_fraction=origins.get('query', 0) / max(1, meta['count']))
            return cert

    def repair(self, i):
        with self.lock:
            if self.store.get('epoch_scoring'):
                raise RuntimeError('frozen calibration in progress')
            result = self.graph.repair(i)
            meta = self.store.get('meta')
            meta['edge_changes'] += result['rewritten_endpoints']
            if result['rewritten_endpoints']:
                meta['routing_revision'] = meta.get('routing_revision', 0) + 1
            queue = set(self.store.get('repair_queue', []))
            queue.discard(i)
            # Ambiguous pool pairs queue their pivot for future native-triggered audit.
            pending = self.store.get('ambiguous_edges', {})
            pending[str(i)] = result['ambiguous']
            self.store.publish({'topology': self.graph.topology(), 'meta': meta,
                                'repair_queue': sorted(queue), 'ambiguous_edges': pending})
            return result

    def audit(self, all_nodes=False):
        ids = range(self.store.get('meta')['count']) if all_nodes else self.store.get('repair_queue', [])
        return sum(self.repair(i)['rewritten_endpoints'] for i in list(ids))

    def retire(self):
        with self.lock:
            ns = self.store.nodes()
            if any(n['state'] != 'native' for n in ns):
                raise RuntimeError('cannot retire incomplete migration')
            # Compact native overlays segment by segment; reclaim superseded files.
            size = self.store.get('meta')['segment_size']
            for start in range(0, len(ns), size):
                ids = range(start, min(start + size, len(ns)))
                path = self.store.write_array([self.store.vector(ns[i]) for i in ids], 'final')
                updates = {}
                for row, i in enumerate(ids):
                    ns[i].update(file=path, row=row)
                    updates[f'node/{i}'] = ns[i]
                self.store.publish(updates)
                self.store.reclaim()
            self.graph.retire()
            self._bridges = None
            self.radii = [0.] * len(self.radii)
            meta = self.store.get('meta')
            meta['retired'] = True
            self.store.publish({'meta': meta, 'bridges': None, 'epoch': None, 'retained': {}, 'ambiguous_edges': {},
                                'centers': None, 'samples': None})
            self.store.reclaim()
            self.store.compact()

    def close(self):
        self.store.close()
