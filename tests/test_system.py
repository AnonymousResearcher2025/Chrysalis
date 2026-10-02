import math
import threading
import time
import numpy as np
import pytest
from chrysalis.index import Index
from chrysalis.math import normalize
from chrysalis.calibration import query_epoch
from chrysalis.backends import LocalQueue
from chrysalis.budget import Budget, Scheduler
from chrysalis.workers import LocalWorker, RpcServer, RpcClient


def make_index(path):
    rng = np.random.default_rng(7)
    x = normalize(rng.normal(size=(80, 4)))
    idx = Index.create(path, x, [0] * len(x), [str(i) for i in range(len(x))], 'old',
                       segment_size=11, graph_parameters=dict(M=8, efConstruction=30, alpha=1.2, seed=42))
    W = np.r_[np.eye(4), np.zeros((2, 4))].T
    maps = [dict(W=W.tolist(), b=[0] * 6, epsilon=.2, gamma=.3)]
    idx.configure('new', maps)
    return idx, x, normalize(x @ W)


def test_rotation_claim_fencing_and_restart(tmp_path):
    idx, old, exact = make_index(tmp_path)
    idx.rotate_one()
    idx.close()
    idx = Index(tmp_path)
    idx.rotate()
    for i in range(80):
        assert idx.graph.vector(i) == pytest.approx(exact[i])
    claims = []
    def claim(owner):
        claims.append(idx.claim(4, owner, 10, now=100))
    threads = [threading.Thread(target=claim, args=(str(i),)) for i in range(8)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert sum(c is not None for c in claims) == 1
    first = next(c for c in claims if c)
    second = idx.claim(4, 'next', 10, now=111)
    assert second['fence'] > first['fence']
    assert not idx.publish_native(first, exact[4], 'query', now=112)
    assert idx.publish_native(second, exact[4], 'query', now=112)
    assert not idx.publish_native(second, exact[4], 'query', now=112)
    idx.close()
    idx = Index(tmp_path)
    assert idx.store.node(4)['state'] == 'native'
    idx.close()


@pytest.mark.parametrize('stage', ['data_sync', 'manifest_sync', 'before_reclaim'])
def test_rotation_crash_frontier(tmp_path, stage):
    idx, _, exact = make_index(tmp_path)
    def fail(at):
        if at == stage:
            raise RuntimeError('injected crash')
    idx.store.crash = fail
    with pytest.raises(RuntimeError):
        idx.rotate_one()
    idx.close()
    idx = Index(tmp_path)
    idx.rotate()
    assert all(n['state'] == 'bridged' for n in idx.store.nodes())
    assert np.allclose([idx.graph.vector(i) for i in range(80)], exact)
    files = {n['file'] for n in idx.store.nodes()}
    assert len(list((tmp_path / 'vectors').glob('*.npy'))) == len(files)
    idx.close()


def test_frozen_calibration_and_no_early_publication(tmp_path):
    idx, _, exact = make_index(tmp_path)
    idx.rotate()
    seen = []
    def encoder(ids):
        seen.append(sum(n['state'] == 'native' for n in idx.store.nodes()))
        with pytest.raises(RuntimeError):
            idx.publish_retained()
        return exact[ids]
    epoch = query_epoch(idx, exact[:20], list(map(str, range(20))), encoder, .05, 8, change_threshold=.05)
    assert seen and set(seen) == {0}
    assert all(n['state'] == 'bridged' for n in idx.store.nodes())
    assert idx.fingerprint() == epoch['snapshot']
    result = idx.search(exact[0], mode='none', k=10)
    assert result['snapshot_applicable']
    idx.publish_retained()
    assert not idx.search(exact[0], mode='none')['snapshot_applicable']
    idx.close()


class FakeEncoder:
    spec = {'revision': 'new'}
    def __init__(self, x):
        self.x, self.calls = x, 0
    def encode(self, raw, category='background', role='corpus'):
        self.calls += len(raw)
        return self.x[list(map(int, raw))]


def test_async_sync_budget_convergence_and_repair(tmp_path):
    idx, _, exact = make_index(tmp_path)
    idx.rotate()
    encoder = FakeEncoder(exact)
    queue = LocalQueue(idx.store)
    result = idx.search(exact[1], rho=4, enqueue=queue.send)
    assert result['requested'] > 0
    assert sum(n['state'] == 'native' for n in idx.store.nodes()) == 0
    worker = LocalWorker(idx, encoder, queue)
    worker.drain()
    assert sum(n['state'] == 'native' for n in idx.store.nodes()) > 0
    result = idx.search(exact[20], mode='sync', resolver=lambda i: encoder.encode([str(i)])[0])
    assert result['requested'] <= 4
    budget = Budget(idx.store, 1, 0, 1)
    budget.pause()
    scheduler = Scheduler(idx, budget, queue)
    assert scheduler.schedule() == 0
    budget.pause(False)
    assert scheduler.schedule() > 0
    worker.budget = budget
    worker.drain()
    assert all(n['state'] == 'native' for n in idx.store.nodes())
    calls = encoder.calls
    idx.audit(all_nodes=True)
    assert idx.graph.check_reverse()
    assert encoder.calls == calls
    idx.retire()
    assert idx.store.get('bridges') is None
    idx.close()
    idx = Index(tmp_path)
    assert idx.store.get('meta')['retired']
    assert np.allclose([idx.graph.vector(i) for i in range(80)], exact)
    idx.close()


def test_bucket_reconciliation_and_duplicate_delivery(tmp_path):
    idx, _, _ = make_index(tmp_path)
    b = Budget(idx.store, 1, 0, 1)
    assert b.reserve(.5, 'job')
    assert not b.reserve(.5, 'job')
    assert b.reconcile('job', .2)
    assert not b.reconcile('job', .2)
    assert idx.store.get('budget')['tokens'] == pytest.approx(.8)
    assert b.reserve(.8, 'overrun')
    b.reconcile('overrun', 1)
    assert idx.store.get('budget')['tokens'] == pytest.approx(-.2)
    q = LocalQueue(idx.store)
    q.send(dict(id='same'))
    q.send(dict(id='same'))
    a = q.receive(now=1)[0]
    assert not q.receive(now=2)
    b = q.receive(now=130)[0]
    assert not q.ack(a)
    assert q.ack(b)
    idx.close()


def test_real_grpc_transport_and_microbatch():
    x = normalize(np.random.default_rng(2).normal(size=(8, 4)))
    encoder = FakeEncoder(x)
    server = RpcServer('127.0.0.1:0', encoder=encoder)
    client = RpcClient(f'127.0.0.1:{server.port}')
    assert np.allclose(client.encode(['0', '1', '2'], 'new'), x[:3])
    client.close()
    server.close()
