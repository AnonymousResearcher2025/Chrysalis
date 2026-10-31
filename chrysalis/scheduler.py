import threading
import time

import numpy as np

from .index import BRIDGED, LEGACY

ORDERS = ("utility", "random", "heat", "radius")


class TokenBucket:
    """Dollar-denominated budget: refills at `rate` dollars per hour up to `capacity`."""

    def __init__(self, capacity, rate_per_hour, price_per_hour):
        self.capacity, self.rate, self.price = capacity, rate_per_hour / 3600.0, price_per_hour / 3600.0
        self.tokens, self.t, self.spent = capacity, time.monotonic(), 0.0
        self.lock = threading.Lock()

    def _refill(self):
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.t) * self.rate)
        self.t = now

    def reserve(self, gpu_seconds):
        with self.lock:
            self._refill()
            cost = gpu_seconds * self.price
            if cost > self.tokens:
                return False
            self.tokens -= cost
            self.spent += cost
            return True


class Scheduler:
    """Background re-embedding in decreasing h(i) * eps_r(i) * deg(i), batched by region."""

    def __init__(self, index, order="utility", bucket=None, ledger=None, seed=0):
        if order not in ORDERS:
            raise ValueError(order)
        self.index, self.order, self.bucket, self.ledger = index, order, bucket, ledger
        self.rng = np.random.default_rng(seed)
        self.random_key = None

    def priority(self, ids):
        ix = self.index
        if self.order == "random":
            if self.random_key is None or len(self.random_key) < ix.n_total:
                self.random_key = self.rng.random(ix.n_total)
            return self.random_key[ids]
        heat = ix.heat_at(ids)
        eps = ix.eps[ix.region[ids]].astype(np.float64)
        if self.order == "heat":
            return heat
        deg = ix.holder.current().indegrees().astype(np.float64)
        deg = np.concatenate([deg, np.zeros(max(0, ix.n_total - len(deg)))])[ids]
        if self.order == "radius":
            return eps
        with np.errstate(invalid="ignore"):
            p = heat * eps * deg
        p[(heat == 0) | (deg == 0)] = 0.0
        return p

    def pick(self, count):
        ix = self.index
        with ix.lock:
            ids = np.nonzero(np.isin(ix.state, (LEGACY, BRIDGED)) & ~ix.deleted)[0]
        if len(ids) == 0:
            return ids
        p = self.priority(ids)
        tie = ix.eps[ix.region[ids]].astype(np.float64)
        order = np.lexsort((ids, -tie, -p))[:count]
        top = ids[order]
        return top[np.argsort(ix.region[top], kind="stable")]

    def step(self, count=None):
        """Claim and enqueue the next `count` items; returns how many were enqueued."""
        ix = self.index
        count = count or ix.cfg.scheduler_batch
        if self.bucket is not None and not self.bucket.reserve(self.ledger.per_item("scheduled") * count):
            return 0
        claims = ix.claim(self.pick(count))
        if claims:
            ix.pool.submit([c[0] for c in claims], [c[1] for c in claims], "scheduled")
        return len(claims)

    def run(self, items, drain=True):
        """Re-embed `items` more nodes (all remaining if None), processing work in this thread."""
        ix = self.index
        done = 0
        while items is None or done < items:
            take = ix.cfg.scheduler_batch if items is None else min(ix.cfg.scheduler_batch, items - done)
            got = self.step(take)
            if drain:
                ix.pool.drain()
            if got == 0:
                break
            done += got
            maintain(ix)
        return done


def maintain(ix):
    """Lease reaping, edge auditing, and epoch closing when either trigger fires."""
    ix.reap()
    while ix.audit():
        if ix.epoch_due():
            ix.close_epoch()
    if ix.epoch_due():
        ix.close_epoch()
