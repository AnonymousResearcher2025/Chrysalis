"""GPU workers. Sync resolution calls the on-demand worker directly over RPC; async
resolution and scheduled work go through region queues on the index host, which
workers drain in region batches."""
import base64
import json
import threading
import time
from collections import OrderedDict, deque
from concurrent import futures

import numpy as np

from .index import BRIDGED, LEGACY, NATIVE, RESOLVING

PRIORITY = ("resolution", "calibration", "calibration_query", "scheduled")


class RegionQueues:
    """Pending (node, fence) pairs per category and region; batches never mix regions."""

    def __init__(self, region_of):
        self.region_of = region_of
        self.lock = threading.Condition()
        self.q = {c: OrderedDict() for c in PRIORITY}

    def put(self, ids, fences, category):
        with self.lock:
            for i, f in zip(ids, fences):
                self.q[category].setdefault(int(self.region_of(i)), deque()).append((int(i), int(f)))
            self.lock.notify_all()

    def take(self, limit, timeout=0.0):
        with self.lock:
            end = time.monotonic() + timeout
            while not any(self.q.values()):
                left = end - time.monotonic()
                if left <= 0:
                    return None
                self.lock.wait(left)
            for c in PRIORITY:
                if self.q[c]:
                    r, items = next(iter(self.q[c].items()))
                    batch = [items.popleft() for _ in range(min(limit, len(items)))]
                    if not items:
                        del self.q[c][r]
                    return c, batch
        return None

    def __len__(self):
        with self.lock:
            return sum(len(d) for q in self.q.values() for d in q.values())


class LocalPool:
    """Encoders in this process: `sync` is the on-demand device, `background` the spot devices."""

    def __init__(self, raw, sync, background=(), batch=128):
        self.raw, self.sync, self.background, self.batch = raw, sync, list(background) or [sync], batch
        self.exec = futures.ThreadPoolExecutor(max_workers=1)
        self.threads, self.stop = [], threading.Event()

    def attach(self, index):
        self.index = index
        self.queues = RegionQueues(lambda i: index.region[i])

    def _run(self, encoder, ids, fences, category):
        vecs = encoder.encode(self.raw.get(ids), role="item", category=category)
        return self.index.publish(ids, fences, vecs, category)

    def encode_now(self, ids, fences, category):
        jobs = [(ids[s:s + self.batch], fences[s:s + self.batch]) for s in range(0, len(ids), self.batch)]
        with futures.ThreadPoolExecutor(max_workers=len(self.background)) as ex:
            list(ex.map(lambda j: self._run(self.background[j[0] % len(self.background)], *j[1], category),
                        enumerate(jobs)))

    def encode_sync(self, ids, fences, deadline):
        fut = self.exec.submit(self._run, self.sync, ids, fences, "resolution_sync")
        try:
            fut.result(timeout=max(0.0, deadline - time.perf_counter()))
        except futures.TimeoutError:
            pass

    def submit(self, ids, fences, category):
        self.queues.put(ids, fences, category)

    def drain(self, max_items=None):
        """Process queued work in the caller's thread; returns the number of items encoded."""
        done = 0
        while max_items is None or done < max_items:
            job = self.queues.take(self.batch if max_items is None else min(self.batch, max_items - done))
            if job is None:
                break
            category, batch = job
            self._run(self.background[0], [b[0] for b in batch], [b[1] for b in batch], category)
            done += len(batch)
        return done

    def start(self):
        def loop(encoder):
            while not self.stop.is_set():
                job = self.queues.take(self.batch, timeout=0.1)
                if job:
                    category, batch = job
                    self._run(encoder, [b[0] for b in batch], [b[1] for b in batch], category)

        for enc in self.background:
            t = threading.Thread(target=loop, args=(enc,), daemon=True)
            t.start()
            self.threads.append(t)

    def close(self):
        self.stop.set()
        for t in self.threads:
            t.join()
        self.exec.shutdown(wait=False)


# ------------------------------------------------------------------ RPC
def _pack(obj):
    def enc(x):
        if isinstance(x, np.ndarray):
            return {"__nd__": base64.b64encode(np.ascontiguousarray(x, np.float32).tobytes()).decode(), "shape": x.shape}
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        raise TypeError(type(x))

    return json.dumps(obj, default=enc).encode()


def _unpack(data):
    def dec(d):
        if "__nd__" in d:
            return np.frombuffer(base64.b64decode(d["__nd__"]), np.float32).reshape(d["shape"])
        return d

    return json.loads(data, object_hook=dec)


def serve(address, methods, threads=16):
    import grpc

    handlers = {name: grpc.unary_unary_rpc_method_handler(lambda req, ctx, f=f: _pack(f(_unpack(req))),
                                                          request_deserializer=None, response_serializer=None)
                for name, f in methods.items()}
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=threads))
    server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler("chrysalis", handlers),))
    port = server.add_insecure_port(address)
    server.start()
    return server, port


class Client:
    def __init__(self, address):
        import grpc

        self.grpc = grpc
        self.channel = grpc.insecure_channel(address)
        self.calls = {}

    def call(self, method, timeout=None, **kw):
        if method not in self.calls:
            self.calls[method] = self.channel.unary_unary(f"/chrysalis/{method}")
        return _unpack(self.calls[method](_pack(kw), timeout=timeout))


class Batcher:
    """Coalesces concurrent sync encode requests into one GPU batch."""

    def __init__(self, encoder, raw, window_ms=0.5):
        self.encoder, self.raw, self.window = encoder, raw, window_ms / 1000
        self.lock = threading.Lock()
        self.pending = []
        self.cv = threading.Condition(self.lock)
        threading.Thread(target=self._loop, daemon=True).start()

    def encode(self, ids):
        slot = {"ids": ids, "done": threading.Event()}
        with self.cv:
            self.pending.append(slot)
            self.cv.notify()
        slot["done"].wait()
        return slot["out"]

    def _loop(self):
        while True:
            with self.cv:
                while not self.pending:
                    self.cv.wait()
            time.sleep(self.window)
            with self.cv:
                batch, self.pending = self.pending, []
            ids = [i for s in batch for i in s["ids"]]
            vecs = self.encoder.encode(self.raw.get(ids), role="item", category="resolution_sync")
            pos = 0
            for s in batch:
                s["out"] = vecs[pos:pos + len(s["ids"])]
                pos += len(s["ids"])
                s["done"].set()


def run_worker(address, host, encoder, raw, sync=False, batch=128, log=None):
    """A GPU worker. With `sync`, it also answers direct Encode calls for in-request resolution.
    `log` records every encoded and every acknowledged item (used by the fault audit)."""
    server = None
    if sync:
        batcher = Batcher(encoder, raw)
        server, _ = serve(address, {"Encode": lambda r: {"vectors": batcher.encode(r["ids"])}})
    client = Client(host)
    out = open(log, "a") if log else None
    while True:
        try:
            job = client.call("Pull", limit=batch, timeout=5)
        except Exception:
            time.sleep(0.5)
            continue
        if not job["ids"]:
            time.sleep(0.05)
            continue
        vecs = encoder.encode(raw.get(job["ids"]), role="item", category=job["category"])
        if out:
            out.write(json.dumps({"t": time.time(), "encoded": job["ids"]}) + "\n")
            out.flush()
        try:
            acked = client.call("Publish", ids=job["ids"], fences=job["fences"], vectors=vecs, category=job["category"],
                                timeout=30)["accepted"]
        except Exception:
            continue
        if out:
            out.write(json.dumps({"t": time.time(), "acked": job["ids"], "accepted": acked}) + "\n")
            out.flush()
    del server


class RemotePool:
    """Index-host side of the worker protocol."""

    def __init__(self, address, on_demand, batch=128):
        self.address, self.batch = address, batch
        self.on_demand = Client(on_demand)
        self.exec = futures.ThreadPoolExecutor(max_workers=8)

    def attach(self, index):
        self.index = index
        self.queues = RegionQueues(lambda i: index.region[i])
        self.server, self.port = serve(self.address, {
            "Pull": self._pull,
            "Publish": lambda r: {"accepted": index.publish(r["ids"], r["fences"], r["vectors"], r["category"])},
            "Search": lambda r: _result(index.search(r["q"], key=r.get("key"), mode=r.get("mode"))),
            "Status": lambda r: _status(index),
        })

    def _pull(self, req):
        job = self.queues.take(req["limit"])
        if job is None:
            return {"ids": [], "fences": [], "category": None}
        category, batch = job
        return {"ids": [b[0] for b in batch], "fences": [b[1] for b in batch], "category": category}

    def encode_now(self, ids, fences, category):
        self.queues.put(ids, fences, category)
        target = set(int(i) for i in ids)
        with self.index.published:
            while any(self.index.state[i] != NATIVE for i in target):
                self.index.published.wait(0.5)
                self.index.reap()
                released = [i for i in target if self.index.state[i] in (LEGACY, BRIDGED)]
                if released:
                    claims = self.index.claim(released)
                    self.queues.put([c[0] for c in claims], [c[1] for c in claims], category)

    def _sync(self, ids, fences):
        vecs = self.on_demand.call("Encode", ids=ids)["vectors"]
        self.index.publish(ids, fences, vecs, "resolution_sync")

    def encode_sync(self, ids, fences, deadline):
        fut = self.exec.submit(self._sync, ids, fences)
        try:
            fut.result(timeout=max(0.0, deadline - time.perf_counter()))
        except futures.TimeoutError:
            pass

    def submit(self, ids, fences, category):
        self.queues.put(ids, fences, category)


def _result(r):
    return {"ids": [int(i) for i in r["ids"]], "bound": r["bound"], "epoch": int(r["epoch"])}


def _status(index):
    with index.lock:
        return {"native": int((index.state == NATIVE).sum()), "resolving": int((index.state == RESOLVING).sum()),
                "n": int(index.n_total), "epoch": int(index.epoch), "converged": bool(index.retired),
                "stale_rejected": int(index.stale_rejected), "expired_leases": int(index.expired_leases),
                "incarnation": int(index.incarnation)}
