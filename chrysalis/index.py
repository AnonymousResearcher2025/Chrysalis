import json
import math
import threading
import time
import zlib
from collections import deque
from pathlib import Path

import numpy as np

from . import _core, bridges, conformal, storage
from .storage import crash_point

LEGACY, BRIDGED, RESOLVING, NATIVE = 0, 1, 2, 3
F_NATIVE, F_REDIRECT, F_DELETED = 1, 2, 4
CALIBRATION = ("calibration", "calibration_query")


class Leases:
    """Resolving claims with fencing tokens. Tokens embed the process incarnation,
    so writes from claims issued before a restart are rejected."""

    def __init__(self, incarnation, seconds):
        self.incarnation, self.seconds, self.counter = incarnation, seconds, 0
        self.active = {}

    def issue(self, node, now):
        self.counter += 1
        fence = (self.incarnation << 40) | self.counter
        self.active[node] = (fence, now + self.seconds)
        return fence

    def valid(self, node, fence, now):
        held = self.active.get(node)
        return held is not None and held[0] == fence and held[1] >= now

    def expired(self, now):
        return [n for n, (_, exp) in self.active.items() if exp < now]


class MigratingIndex:
    """One logical index that moves from E_v to E_{v+1} in place while serving.

    Node states: legacy (old vector), bridged (mapped vector with a regional radius),
    resolving (claimed for re-embedding under a lease) and native."""

    def __init__(self, root, cfg):
        self.root, self.cfg = Path(root), cfg
        self.meta = json.loads((self.root / "index.json").read_text())
        self.n = self.meta["n"]
        self.lock = threading.RLock()
        self.log_lock = threading.Lock()
        self.epoch_lock = threading.RLock()
        self.published = threading.Condition(self.lock)
        self.holder = _core.Holder()
        self.pool = None
        self.pending_live = None
        self.retired = False
        self.flips = 0
        self.stale_rejected = 0
        self.expired_leases = 0
        self.audit_queue = deque()
        self.uncertain = {}
        self.final_audit = set()
        self.recent = deque(maxlen=cfg.recent_window)
        self.calibrated = set()
        self.requests = 0
        self.heat_on = True
        self.heat = np.zeros(self.n)
        self.heat_clock = np.zeros(self.n)
        self.region_scaling = True
        self.count_rule = "max"
        self.freeze_lambda = False

    # ------------------------------------------------------------ creation
    @staticmethod
    def create(root, cfg, graph, old_vectors):
        """Persist an E_v index: its graph and its vectors in fixed-size segments."""
        root = Path(root)
        for sub in ("segments", "native", "staged", "snapshot"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        storage.save_graph(root, graph)
        rows = cfg.segment_rows
        for s in range(math.ceil(len(old_vectors) / rows)):
            storage.write_segment(root, s, old_vectors[s * rows:(s + 1) * rows])
        storage.write_atomic(root / "index.json", json.dumps(dict(
            n=len(old_vectors), d_old=int(old_vectors.shape[1]), segment_rows=rows)).encode())
        storage.Manifest(root / "manifest.log").append("create", n=len(old_vectors), d_old=int(old_vectors.shape[1]))

    def attach(self, pool):
        self.pool = pool
        pool.attach(self)

    def _arrays(self, n_total):
        self.n_total = n_total
        self.state = np.full(n_total, LEGACY, np.uint8)
        self.embed_counts = np.zeros(n_total, np.uint16)
        self.deleted = np.zeros(n_total, bool)
        grow = n_total - len(self.heat)
        self.heat = np.concatenate([self.heat, np.zeros(grow)])
        self.heat_clock = np.concatenate([self.heat_clock, np.zeros(grow)])
        self.segment_legacy = np.ones(math.ceil(n_total / self.cfg.segment_rows), bool)

    # ------------------------------------------------------------ calibration
    def upgrade(self, d_new, model, seed=0, force=False, translator=None, direct=()):
        """Calibrate: coarse regions, regional bridges and radii, then the certificate of
        view 0, after which E_{v+1} queries are served. With `translator`, every stored
        vector is mapped by that model instead of by regional bridges. Nodes in `direct`
        are re-embedded during calibration and never bridged (chained upgrades). `model`
        identifies E_{v+1}; overlays only count vectors of that version as exact."""
        cfg = self.cfg
        self.d_new, self.model, self.version = d_new, model, zlib.crc32(model.encode())
        self.wal = storage.Manifest(self.root / "manifest.log")
        self.incarnation = 1
        self.wal.append("incarnation", value=1)
        self.leases = Leases(self.incarnation, cfg.lease_seconds)
        self.graph = self._graph()
        old = self._old_vectors()
        _, self.region = bridges.coarse_quantizer(old, cfg.regions, seed)
        np.save(self.root / "regions.npy", self.region)
        self._arrays(self.n)
        self.gen, self.gen_ids = 0, {0: []}
        self.log = storage.NativeLog(self.root / "native" / "000000.bin", d_new)
        self.cal_overlay = self.live = _core.Overlay(self.n, d_new, self.version)
        direct = np.asarray(direct, np.int64)
        eligible = self.region.copy()
        eligible[direct] = cfg.regions
        splits = bridges.stratified_sample(eligible, cfg.regions, cfg.items_per_region, cfg.fit_per_region, seed)
        sample = np.concatenate([np.concatenate([f, c]) for f, c in splits] + [direct]).astype(np.int64)
        self._embed_blocking(sample, "calibration")
        exact = lambda ids: np.stack([self.cal_overlay.get(int(i)) for i in ids]) if len(ids) else np.zeros((0, d_new), np.float32)
        rng = np.random.default_rng(seed)
        with self.lock:
            recent_queries = np.array(list({k: v for k, v in self.recent}.values()), np.float32)
        if len(recent_queries) == 0:
            raise RuntimeError("calibration needs recent E_{v+1} queries; record them with observe()")
        self.bridges, radii, pooled = [], [], None
        for r, (fit, cal) in enumerate(splits):
            if translator is not None:
                mapped = translator.map
            else:
                if len(fit) >= 2:
                    br = bridges.fit_bridge(old[fit], exact(fit), cfg.bridge_rank, cfg.ridge_grid, cfg.ridge_folds, seed + r)
                else:
                    if pooled is None:
                        allfit = np.concatenate([f for f, _ in splits])
                        pooled = bridges.fit_bridge(old[allfit], exact(allfit), cfg.bridge_rank, cfg.ridge_grid, cfg.ridge_folds, seed)
                    br = pooled
                self.bridges.append(br)
                mapped = br
            radii.append(conformal.region_radii(mapped, old[cal], exact(cal), recent_queries, cfg.alpha, rng))
        self.eps = np.array([x["eps"] for x in radii], np.float32)
        self.gamma = np.array([x["gamma"] for x in radii], np.float32)
        self.dbar = np.array([x["dbar"] for x in radii], np.float64)
        self.diagnosis = conformal.diagnostic(radii, cfg.tau)
        if self.diagnosis["recommend_full_reembed"] and not force:
            self.wal.append("diagnostic", **self.diagnosis)
            return self.diagnosis
        if translator is None:
            bridges.save(self.root / "bridges.npz", self.bridges)
        np.savez(self.root / "radii.npz", eps=self.eps, gamma=self.gamma, dbar=self.dbar)
        self.wal.append("upgrade", d_new=d_new, model=model, regions=cfg.regions, **self.diagnosis)
        view = self._initial_view(old, translator)
        self.lam = self._calibrate_view(view, "calibration_query")
        self.epoch, self.lams = 0, {0: self.lam}
        self.wal.append("epoch", epoch=0, lam=_num(self.lam), merged_gen=-1, staged=None)
        self.holder.publish(view)
        self.edges = int(view.indegrees().sum())
        self.auditor = self._new_auditor(view)
        self.staged_log = storage.StagedLog(self.root / "staged" / "000000.log")
        return self.diagnosis

    def _graph(self):
        c = self.cfg
        return storage.load_graph(self.root, 1, c.max_degree, c.ef_construction, c.prune_alpha)

    def _old_vectors(self):
        rows, d_old = self.meta["segment_rows"], self.meta["d_old"]
        return np.concatenate([storage.read_segment(storage.segment_path(self.root, s, d_old), d_old)
                               for s in range(math.ceil(self.n / rows))])

    def _initial_view(self, old, translator):
        rows = self.cfg.segment_rows
        natives = np.nonzero(self.state == NATIVE)[0]
        flags = np.zeros(self.n, np.uint8)
        segs = []
        if translator is None:
            flags[natives] = F_NATIVE | F_REDIRECT
            segs = [_core.Segment(old.shape[1], True, old[s * rows:(s + 1) * rows]) for s in range(len(self.segment_legacy))]
            maps, frozen = [b.core() for b in self.bridges], self.cal_overlay
        else:
            flags[natives] = F_NATIVE
            for s in range(len(self.segment_legacy)):
                block = np.zeros((rows, self.d_new), np.float32)
                part = translator.map(old[s * rows:(s + 1) * rows])
                block[:len(part)] = part
                ids = np.arange(s * rows, s * rows + len(part))
                nat = ids[self.state[ids] == NATIVE]
                if len(nat):
                    block[nat - s * rows] = np.stack([self.cal_overlay.get(int(i)) for i in nat])
                storage.write_segment(self.root, s, block)
                self.wal.append("rotated", segment=s, width=self.d_new)
                segs.append(_core.Segment(self.d_new, False, block))
            self.segment_legacy[:] = False
            self.state[self.state == LEGACY] = BRIDGED
            maps, frozen = [], None
        self.live = _core.Overlay(self.n + self.insert_headroom(), self.d_new, self.version)
        with self.log_lock:
            self.gen_switch()
        return _core.View.initial(self.graph, rows, segs, self.region, maps, flags, frozen, self.live, self.d_new, self.version)

    def _calibrate_view(self, view, category):
        """lambda: conformal quantile over distinct recent queries of the largest
        region-normalized error among each query's candidates bridged in `view`."""
        queries = self._calibration_queries()
        if len(queries) == 0:
            return math.inf
        flags = view.flags()
        cands = [view.search(q, self.cfg.ef_search) for q in queries]
        bridged = [ids[(flags[ids] & F_NATIVE) == 0] for ids, _ in cands]
        need = np.unique(np.concatenate(bridged)).astype(np.int64)
        self._embed_until(need, category, view.live)
        scores = []
        for q, (ids, d) in zip(queries, cands):
            mask = (flags[ids] & F_NATIVE) == 0
            if not mask.any():
                scores.append(0.0)
                continue
            d_star = np.linalg.norm(np.stack([view.live.get(int(i)) for i in ids[mask]]) - q, axis=1)
            scores.append(conformal.query_score(d[mask], d_star, self._cert_eps(ids[mask])))
        return conformal.quantile(scores, self.cfg.alpha_q)

    def _calibration_queries(self):
        """m_q distinct queries sampled from the recent request window. Queries that calibrated
        the previous view are skipped: their re-embeddings are part of the view being calibrated."""
        with self.lock:
            window = list(self.recent)
        picked, seen = [], set(self.calibrated)
        for j in np.random.default_rng((self.requests, getattr(self, "epoch", -1) + 1)).permutation(len(window)):
            key, vec = window[j]
            if key not in seen:
                seen.add(key)
                picked.append((key, vec))
                if len(picked) == self.cfg.calibration_queries:
                    break
        self.calibrated = {k for k, _ in picked}
        return np.array([v for _, v in picked], np.float32)

    def observe(self, keys, vectors):
        """Record requests served before the upgrade; calibration samples from this window."""
        with self.lock:
            for key, vec in zip(keys, vectors):
                self.recent.append((key, np.asarray(vec, np.float32)))

    # ------------------------------------------------------------ claims and publication
    def claim(self, ids):
        now = time.monotonic()
        out = []
        with self.lock:
            for i in ids:
                i = int(i)
                if self.state[i] in (LEGACY, BRIDGED):
                    self.state[i] = RESOLVING
                    out.append((i, self.leases.issue(i, now)))
        return out

    def reap(self):
        """Reset expired claims; their late writes then fail the fence check."""
        now = time.monotonic()
        with self.lock:
            dead = self.leases.expired(now)
            self.expired_leases += len(dead)
            for i in dead:
                del self.leases.active[i]
                if self.state[i] == RESOLVING:
                    self.state[i] = LEGACY if self.segment_legacy[i // self.cfg.segment_rows] else BRIDGED
        return dead

    def publish(self, ids, fences, vectors, category):
        """Write and sync the vectors, append and sync the manifest record, then expose them."""
        now = time.monotonic()
        with self.lock:
            keep = [j for j, (i, f) in enumerate(zip(ids, fences)) if self.leases.valid(int(i), int(f), now)]
            self.stale_rejected += len(ids) - len(keep)
        if not keep:
            return 0
        ids = [int(ids[j]) for j in keep]
        fences = [int(fences[j]) for j in keep]
        vectors = np.ascontiguousarray(np.asarray(vectors, np.float32)[keep])
        with self.log_lock:
            gen = self.gen
            offset = self.log.append(ids, fences, vectors)
            crash_point("publish")
            self.wal.append("native", gen=gen, offset=offset, count=len(ids))
            with self.lock:
                for i, v in zip(ids, vectors):
                    self.live.put(i, v)
                    if self.pending_live is not None:
                        self.pending_live.put(i, v)
                    self.state[i] = NATIVE
                    self.leases.active.pop(i, None)
                    self.embed_counts[i] += 1
                    self.gen_ids[gen].append(i)
                    self.audit_queue.append(i)
                    self.audit_queue.extend(self.uncertain.pop(i, ()))
                if category not in CALIBRATION:
                    self.flips += len(ids)
                self.published.notify_all()
        return len(ids)

    def gen_switch(self):
        """Start the next native-log generation (caller holds log_lock)."""
        self.gen += 1
        self.gen_ids[self.gen] = []
        self.log.close()
        self.log = storage.NativeLog(self.root / "native" / f"{self.gen:06d}.bin", self.d_new)

    def _embed_blocking(self, ids, category):
        claims = self.claim(ids)
        if claims:
            self.pool.encode_now([c[0] for c in claims], [c[1] for c in claims], category)

    def _embed_until(self, ids, category, overlay):
        """Re-embed until every id has an exact vector in `overlay`; items claimed by others are awaited."""
        while True:
            missing = [int(i) for i in ids if not overlay.has(int(i))]
            if not missing:
                return
            self._embed_blocking(missing, category)
            with self.published:
                if any(not overlay.has(i) for i in missing):
                    self.published.wait(0.05)
            self.reap()

    # ------------------------------------------------------------ serving
    def search(self, q, key=None, mode=None, k=None, observe=True):
        """Algorithm 1: beam search on the frozen view, boundary resolution, certificate."""
        cfg = self.cfg
        k = k or cfg.k
        mode = mode or cfg.resolution_mode
        q = np.ascontiguousarray(q, dtype=np.float32)
        t0 = time.perf_counter()
        requested = resolved = waited = 0
        with self.holder.pin() as pin:
            ids, d_hat = pin.search(q, cfg.ef_search)
            epoch = pin.epoch
            if self.retired:
                if observe:
                    self._observe_request(key, q, ids)
                return dict(ids=ids[:k], bound=None, candidates=ids, epoch=epoch, requested=0, resolved=0, waited=0,
                            latency=time.perf_counter() - t0)
            in_view = pin.native_in_view(ids)
            lam = self.lams[epoch]
            exact, d = pin.annotate(q, ids, d_hat)
            eps = self.eps[self.region[ids]]
            if mode != "off":
                amb = _core.ambiguous(d, exact, eps, ids, k)[: cfg.resolve_limit]
                requested = len(amb)
                if amb:
                    claims = self.claim(ids[amb])
                    mine = {c[0] for c in claims}
                    if mode == "sync":
                        deadline = t0 + cfg.sync_deadline_ms / 1000
                        if claims:
                            self.pool.encode_sync([c[0] for c in claims], [c[1] for c in claims], deadline)
                        waited = self._wait([int(ids[j]) for j in amb], deadline)
                    elif claims:
                        self.pool.submit([c[0] for c in claims], [c[1] for c in claims], "resolution")
                    resolved = len(mine)
                    exact, d = pin.annotate(q, ids, d_hat)
            cert_eps = self._cert_eps(ids)
            top, m, z, bound = _core.certify(d, exact, conformal.widths(cert_eps, lam), ids, k)
            if self.count_rule == "m":
                bound = m / k
        if observe:
            self._observe_request(key, q, ids)
        return dict(ids=ids[top], bound=bound, m=m, z=z, candidates=ids, d_hat=d_hat, in_view=in_view, eps=cert_eps,
                    lam=lam, epoch=epoch, requested=requested, resolved=resolved, waited=waited,
                    latency=time.perf_counter() - t0)

    def _cert_eps(self, ids):
        eps = self.eps[self.region[ids]]
        return eps if self.region_scaling else np.where(np.isinf(eps), eps, 1.0).astype(np.float32)

    def _wait(self, nodes, deadline):
        with self.published:
            while True:
                done = sum(self.state[i] == NATIVE for i in nodes)
                left = deadline - time.perf_counter()
                if done == len(nodes) or left <= 0:
                    return done
                self.published.wait(left)

    def _observe_request(self, key, q, ids):
        with self.lock:
            self.requests += 1
            if self.heat_on:
                self.heat[ids] = self.heat_at(ids) + 1.0
                self.heat_clock[ids] = self.requests
            self.recent.append((key if key is not None else hash(q.tobytes()), q))

    def add_heat(self, ids):
        """Count one candidate list toward the heat of its nodes (warm-up traffic)."""
        with self.lock:
            self.requests += 1
            self.heat[ids] = self.heat_at(ids) + 1.0
            self.heat_clock[ids] = self.requests

    def heat_at(self, ids=slice(None)):
        """Decayed count of recent candidate lists containing each node."""
        return self.heat[ids] * np.exp2(-(self.requests - self.heat_clock[ids]) / self.cfg.heat_half_life_requests)

    # ------------------------------------------------------------ rotation
    def rotate(self, segments=None):
        """Overwrite each legacy segment with its bridged image. The rotated values equal the
        on-the-fly mapped values bit for bit, so the routing of the current view is unchanged."""
        rows = self.cfg.segment_rows
        with self.epoch_lock:
            pending = list(np.nonzero(self.segment_legacy)[0]) if segments is None else list(segments)
        for s in pending:
            with self.epoch_lock:
                if not self.segment_legacy[s]:
                    continue
                view = self.holder.current()
                block = view.rotate_segment(int(s))
                storage.write_segment(self.root, int(s), block)
                crash_point("rotate")
                self.wal.append("rotated", segment=int(s), width=self.d_new)
                self.holder.publish(view.with_segment(int(s), _core.Segment(self.d_new, False, block)))
                del view
                storage.segment_path(self.root, int(s), self.meta["d_old"]).unlink(missing_ok=True)
                with self.lock:
                    self.segment_legacy[s] = False
                    ids = np.arange(s * rows, min(self.n, (s + 1) * rows))
                    self.state[ids[self.state[ids] == LEGACY]] = BRIDGED

    # ------------------------------------------------------------ edge auditor
    def _new_auditor(self, view):
        return _core.Auditor(view, self.gamma, self.cfg.prune_alpha, self.cfg.auditor_pool)

    def audit(self, limit=64):
        """Rerun neighbor selection around newly native nodes; changes are staged for the next epoch."""
        with self.epoch_lock:
            with self.lock:
                batch = [self.audit_queue.popleft() for _ in range(min(limit, len(self.audit_queue)))]
            if not batch:
                return 0
            pending = []
            for p in batch:
                pending.extend(self.auditor.audit(int(p))[0])
            self.staged_log.append(batch, self.auditor.take_dirty())
            crash_point("stage")
            with self.lock:
                for p, c in pending:
                    if self.state[c] != NATIVE:
                        self.uncertain.setdefault(c, set()).add(p)
                    elif self.state[p] != NATIVE:
                        self.uncertain.setdefault(p, set()).add(p)
                    else:
                        self.final_audit.add(p)
            return len(batch)

    # ------------------------------------------------------------ epochs
    def epoch_due(self):
        return (self.flips >= self.cfg.epoch_node_fraction * self.n
                or self.auditor.changed >= self.cfg.epoch_edge_fraction * self.edges)

    def close_epoch(self, calibrate=True):
        """Form the next view from the overlay and staged edges, calibrate it while the
        current view serves, switch atomically, then merge the overlay in place.
        Without `calibrate`, the view keeps the current scale (ablations only)."""
        with self.epoch_lock:
            self.rotate()
            current = self.holder.current()
            n_next = self.n_total
            nxt_live = _core.Overlay(n_next + self.insert_headroom(), self.d_new, self.version)
            with self.log_lock:
                with self.lock:
                    self.pending_live = nxt_live
                    merged_gen = self.gen
                    merged = np.array(sorted({i for g, ids in self.gen_ids.items() if 0 < g <= merged_gen for i in ids}), np.int64)
                    seed = np.array(self.gen_ids.get(0, []), np.int64)
                    flips_before = self.flips
                self.gen_switch()
            flags = np.concatenate([current.flags(), np.zeros(n_next - current.n, np.uint8)])
            flags[merged] |= F_NATIVE | F_REDIRECT
            flags[self.deleted[:n_next]] |= F_DELETED
            l0, upper = self.auditor.staged()
            staged_file = self.root / "staged" / f"{self.epoch + 1:06d}.view.npz"
            storage.save_staged(staged_file, l0, upper)
            extra = self._extra_segments(current, n_next)
            view = current.next(self.epoch + 1, l0, upper, flags, self.region[:n_next], extra, current.live, nxt_live)
            del current
            self.auditor = self._new_auditor(view)
            new_staged = storage.StagedLog(self.root / "staged" / f"{self.epoch + 1:06d}.log")
            lam = self._calibrate_view(view, "calibration_query") if calibrate and not self.freeze_lambda else self.lam
            crash_point("switch")
            self.wal.append("epoch", epoch=self.epoch + 1, lam=_num(lam), merged_gen=merged_gen, staged=staged_file.name)
            self.lams[self.epoch + 1] = lam
            self.holder.publish(view)
            self.lams.pop(self.epoch, None)
            with self.lock:
                self.epoch, self.lam = self.epoch + 1, lam
                self.live, self.pending_live = nxt_live, None
                self.flips -= flips_before
                self.staged_log.close()
                self.staged_log = new_staged
            self.edges = int(view.indegrees().sum())
            self._merge(view, merged, merged_gen, seed)
            self._apply_edges(self.epoch, l0, upper)
            staged = self.auditor.staged()
            self.auditor = self._new_auditor(self.holder.current())
            self.auditor.load(*staged)

    def _merge(self, view, merged, gen, seed=()):
        """Write the frozen overlay's vectors into their segment rows. The served view still
        redirects these rows to the overlay, so no reader sees a partial write."""
        rows = self.cfg.segment_rows
        frozen = view.frozen
        flags = view.flags().copy()
        for s in np.unique(merged // rows):
            ids = merged[merged // rows == s]
            vecs = np.stack([frozen.get(int(i)) for i in ids])
            view.segment(int(s)).write_rows((ids - s * rows).astype(np.int32), vecs)
            storage.write_segment_rows(storage.segment_path(self.root, int(s), self.d_new), ids - s * rows, vecs)
            crash_point("merge")
            flags[ids] &= ~np.uint8(F_REDIRECT)
        self.holder.publish(view.with_flags(flags, None))
        mask = self._merged_mask()
        mask[merged] = True
        mask[np.asarray(seed, np.int64)] = True
        storage.write_atomic(self.root / "native" / "merged.npy", _npy(mask))
        self.wal.append("merged", gen=gen)
        with self.lock:
            for g in [g for g in self.gen_ids if g <= gen]:
                del self.gen_ids[g]
        for g in range(gen + 1):
            (self.root / "native" / f"{g:06d}.bin").unlink(missing_ok=True)

    def _apply_edges(self, epoch, l0, upper):
        """Fold the epoch's staged edges into the adjacency file in place, then reclaim the staged state."""
        storage.apply_edges(self.root, self.cfg.max_degree, l0, upper)
        crash_point("edges")
        self.wal.append("edges", epoch=epoch)
        (self.root / "staged" / f"{epoch:06d}.view.npz").unlink(missing_ok=True)
        (self.root / "staged" / f"{epoch - 1:06d}.log").unlink(missing_ok=True)

    def _merged_mask(self):
        path = self.root / "native" / "merged.npy"
        old = np.load(path) if path.exists() else np.zeros(0, bool)
        mask = np.zeros(self.n_total, bool)
        mask[:len(old)] = old
        return mask

    def _extra_segments(self, view, n_next):
        rows = self.cfg.segment_rows
        segs = []
        for s in range(view.segments, math.ceil(n_next / rows)):
            block = np.zeros((rows, self.d_new), np.float32)
            storage.write_segment(self.root, s, block)
            self.wal.append("rotated", segment=s, width=self.d_new)
            segs.append(_core.Segment(self.d_new, False, block))
        return segs

    # ------------------------------------------------------------ writes
    def insert_headroom(self):
        return max(self.cfg.segment_rows, int(self.cfg.epoch_node_fraction * self.n))

    def insert(self, vector):
        """An insert is embedded by E_{v+1}, enters the overlay as native, and becomes searchable next epoch."""
        vector = np.asarray(vector, np.float32)
        with self.epoch_lock:
            i = self.n_total
            if i >= self.live.capacity:
                self.close_epoch()
            with self.lock:
                self._grow(i + 1)
                fence = self.leases.issue(i, time.monotonic())
                self.state[i] = RESOLVING
            self.publish([i], [fence], vector[None], "insert")
            with self.lock:
                self.audit_queue.remove(i)
            _, _, nearest = self.auditor.link(i, vector, self.cfg.ef_construction)
            self.region[i] = self.region[nearest] if nearest >= 0 else 0
            return i

    def delete(self, i):
        """A delete becomes a tombstone; the auditor repairs the edges around it."""
        i = int(i)
        with self.lock:
            self.live.kill(i)
            if self.pending_live is not None:
                self.pending_live.kill(i)
            self.deleted[i] = True
            self.flips += 1
        self.wal.append("delete", node=i)
        if i < self.holder.current().n:
            self.audit_queue.extend(int(p) for p in self.holder.current().in_neighbors(i))

    def update(self, i, vector):
        self.delete(i)
        return self.insert(vector)

    def _grow(self, n_total):
        extra = n_total - len(self.state)
        if extra > 0:
            self.state = np.append(self.state, np.full(extra, RESOLVING, np.uint8))
            self.region = np.append(self.region, np.zeros(extra, np.uint16))
            self.deleted = np.append(self.deleted, np.zeros(extra, bool))
            self.embed_counts = np.append(self.embed_counts, np.zeros(extra, np.uint16))
            self.heat = np.append(self.heat, np.zeros(extra))
            self.heat_clock = np.append(self.heat_clock, np.zeros(extra))
            need = math.ceil(n_total / self.cfg.segment_rows) - len(self.segment_legacy)
            if need > 0:
                self.segment_legacy = np.append(self.segment_legacy, np.zeros(need, bool))
        self.n_total = n_total

    # ------------------------------------------------------------ convergence
    def all_native(self):
        with self.lock:
            return bool((self.state[~self.deleted] == NATIVE).all())

    def converged(self):
        with self.lock:
            return self.all_native() and not self.audit_queue and self.live.count == 0 and self.auditor.changed == 0

    def retire(self):
        """Once every node is native and staged repairs have drained, drop the bridges,
        radii and interval logic; the index is a plain E_{v+1} index."""
        if not self.all_native():
            raise RuntimeError("retire() needs every node native")
        with self.lock:
            self.audit_queue.extend(self.final_audit)
            self.final_audit.clear()
        while not self.converged():
            while self.audit():
                pass
            self.close_epoch()
        self.retired = True
        self.wal.append("retired", epoch=self.epoch)
        for name in ("bridges.npz", "radii.npz"):
            (self.root / name).unlink(missing_ok=True)

    def chain(self, new_root):
        """Start the next upgrade before this migration ends: the stored vectors become the old
        vectors of a new index at `new_root`. Returns the nodes that are not yet native; they
        would fall two versions behind and must be passed as `direct` to the new upgrade()."""
        self.close_epoch()
        view = self.holder.current()
        vectors = np.concatenate([view.segment(s).array() for s in range(view.segments)])[: view.n]
        MigratingIndex.create(new_root, self.cfg, self._graph(), vectors)
        with self.lock:
            return np.nonzero(self.state[: view.n] != NATIVE)[0]

    # ------------------------------------------------------------ recovery
    @classmethod
    def open(cls, root, cfg, pool):
        """Restart from the exact frontier: segments, merged natives, native logs and staged
        edges, as published by the manifest. Claims from before the crash are void."""
        self = cls(root, cfg)
        self.wal = storage.Manifest(self.root / "manifest.log")
        recs = list(self.wal.records())
        up = next(r for r in recs if r["kind"] == "upgrade")
        self.d_new, self.model, self.version = up["d_new"], up["model"], zlib.crc32(up["model"].encode())
        self.incarnation = 1 + sum(r["kind"] == "incarnation" for r in recs)
        self.wal.append("incarnation", value=self.incarnation)
        self.leases = Leases(self.incarnation, cfg.lease_seconds)
        self.retired = any(r["kind"] == "retired" for r in recs)
        rows = cfg.segment_rows
        rad = self.root / "radii.npz"
        if rad.exists():
            z = np.load(rad)
            self.eps, self.gamma, self.dbar = z["eps"], z["gamma"], z["dbar"]
        else:
            self.eps = self.gamma = np.zeros(cfg.regions, np.float32)
        path = self.root / "bridges.npz"
        self.bridges = bridges.load(path, cfg.regions) if path.exists() else []
        epochs = [r for r in recs if r["kind"] == "epoch"]
        rotated = {r["segment"] for r in recs if r["kind"] == "rotated"}
        committed = epochs[-1]["merged_gen"]
        merge_done = committed < 0 or any(r["kind"] == "merged" and r["gen"] == committed for r in recs)
        upto = {}
        for r in recs:
            if r["kind"] == "native":
                upto[r["gen"]] = max(upto.get(r["gen"], 0), r["offset"] + r["count"] * storage.NativeLog.record_size(self.d_new))
        natives = {}
        for p in sorted((self.root / "native").glob("*.bin")):
            g = int(p.stem)
            storage.NativeLog.truncate(p, upto.get(g, 0))
            natives[g] = storage.NativeLog(p, self.d_new).read()
        mpath = self.root / "native" / "merged.npy"
        merged_mask = np.load(mpath) if mpath.exists() else np.zeros(self.n, bool)
        if merge_done:
            natives = {g: v for g, v in natives.items() if g > committed or (committed < 0 and g == 0)}
        n_total = max([self.n, len(merged_mask)] + [int(ids.max()) + 1 for ids, _ in natives.values() if len(ids)])
        self._arrays(n_total)
        region = np.load(self.root / "regions.npy")
        self.region = np.concatenate([region, np.zeros(n_total - len(region), np.uint16)])
        for s in range(len(self.segment_legacy)):
            self.segment_legacy[s] = s not in rotated
            if s in rotated:
                self.state[s * rows:(s + 1) * rows] = BRIDGED
        self.state[:len(merged_mask)][merged_mask] = NATIVE
        for ids, _ in natives.values():
            self.state[ids] = NATIVE
            np.add.at(self.embed_counts, ids, 1)
        self.deleted[[r["node"] for r in recs if r["kind"] == "delete"]] = True
        in_view = [g for g in natives if (g <= committed) or (committed < 0 and g == 0)]
        frozen = _core.Overlay(n_total, self.d_new, self.version)
        self.live = _core.Overlay(n_total + self.insert_headroom(), self.d_new, self.version)
        for g, (ids, vecs) in natives.items():
            target = frozen if g in in_view else self.live
            for i, v in zip(ids, vecs):
                target.put(int(i), v)
        segs = []
        for s in range(len(self.segment_legacy)):
            width = self.meta["d_old"] if self.segment_legacy[s] else self.d_new
            segs.append(_core.Segment(width, bool(self.segment_legacy[s]),
                                      storage.read_segment(storage.segment_path(self.root, s, width), width)))
        last = epochs[-1]["epoch"]
        if last > 0 and not any(r["kind"] == "edges" and r["epoch"] == last for r in recs):
            l0, upper = storage.load_staged(self.root / "staged" / f"{last:06d}.view.npz")
            storage.apply_edges(self.root, cfg.max_degree, l0, upper)
            self.wal.append("edges", epoch=last)
        graph = self._graph()
        flags = np.zeros(graph.n, np.uint8)
        merged_mask = merged_mask[: graph.n]
        flags[: len(merged_mask)][merged_mask] = F_NATIVE
        for g in in_view:
            flags[natives[g][0]] |= F_NATIVE | F_REDIRECT
        flags[self.deleted[: graph.n]] |= F_DELETED
        view = _core.View.initial(graph, rows, segs, self.region[: graph.n], [b.core() for b in self.bridges], flags,
                                  frozen, self.live, self.d_new, self.version)
        view = view.next(last, {}, [], flags, self.region[: graph.n], [], frozen, self.live)
        self.epoch, self.lam = epochs[-1]["epoch"], _unnum(epochs[-1]["lam"])
        self.lams = {self.epoch: self.lam}
        self.gen = max(list(natives) + [committed, 0]) + 1
        self.gen_ids = {g: [int(i) for i in natives[g][0]] for g in natives if g not in in_view}
        self.gen_ids[self.gen] = []
        self.log = storage.NativeLog(self.root / "native" / f"{self.gen:06d}.bin", self.d_new)
        self.attach(pool)
        self.holder.publish(view)
        if not merge_done:
            self.gen_ids.update({g: [int(i) for i in natives[g][0]] for g in in_view})
            self._merge(view, np.concatenate([natives[g][0] for g in in_view]).astype(np.int64), committed)
        else:
            fl = view.flags().copy()
            fl[~self.segment_legacy[np.arange(view.n) // rows]] &= ~np.uint8(F_REDIRECT)
            self.holder.publish(view.with_flags(fl, frozen if (fl & F_REDIRECT).any() else None))
        self.edges = int(self.holder.current().indegrees().sum())
        self.auditor = self._new_auditor(self.holder.current())
        staged = self.root / "staged" / f"{self.epoch:06d}.log"
        now = self.holder.current()
        audited, l0, upper = storage.StagedLog(staged).read(lambda l, i: now.neighbors(i, l) if i < now.n else [])
        for later in (self.root / "staged").glob("*.log"):
            if int(later.stem) > self.epoch:
                later.unlink()
        self.auditor.load(l0, upper)
        self.staged_log = storage.StagedLog(staged)
        for g in natives:
            if g not in in_view:
                self.audit_queue.extend(int(i) for i in natives[g][0] if int(i) not in audited)
        return self

    # ------------------------------------------------------------ status
    def snapshot(self, store=None):
        """Atomic manifest snapshot, copied to object storage when `store` is given."""
        meta = dict(seq=self.wal.seq, epoch=self.epoch, lam=_num(self.lam), incarnation=self.incarnation,
                    native=int((self.state == NATIVE).sum()), n=int(self.n_total))
        data = json.dumps(meta).encode()
        storage.write_atomic(self.root / "snapshot" / "manifest.json", data)
        if store is not None:
            store.put(f"snapshots/{self.wal.seq:012d}/manifest.log", (self.root / "manifest.log").read_bytes())
            store.put(f"snapshots/{self.wal.seq:012d}/COMMIT", data)
        return meta

    def native_fraction(self):
        with self.lock:
            return float((self.state == NATIVE).mean())


def _npy(a):
    import io

    buf = io.BytesIO()
    np.save(buf, a)
    return buf.getvalue()


def _num(x):
    return "inf" if math.isinf(x) else float(x)


def _unnum(x):
    return math.inf if x == "inf" else float(x)
