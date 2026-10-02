import threading
import time
import numpy as np
from chrysalis.backends import LocalQueue
from chrysalis.budget import Budget, Scheduler
from chrysalis.index import Index
from chrysalis.workers import LocalWorker
from tests.test_system import FakeEncoder, make_index


def test_duplicate_physical_workers_only_one_encoder(tmp_path):
    idx, _, exact = make_index(tmp_path)
    idx.rotate()
    c = idx.claim(0, 'queued-job')
    q = LocalQueue(idx.store)
    encoder = FakeEncoder(exact)
    original = encoder.encode
    def slow(*args, **kwargs):
        time.sleep(.05)
        return original(*args, **kwargs)
    encoder.encode = slow
    workers = [LocalWorker(idx, encoder, q), LocalWorker(idx, encoder, q)]
    # Independent messages intentionally duplicate the same logical claim.
    q.send(dict(id='one', claim=c, origin='query'))
    q.send(dict(id='two', claim=c, origin='query'))
    messages = q.receive(2)
    ts = [threading.Thread(target=w.process, args=(m,)) for w, m in zip(workers, messages)]
    for t in ts: t.start()
    for t in ts: t.join()
    assert encoder.calls == 1
    assert idx.store.node(0)['state'] == 'native'
    idx.close()


def test_reservation_claim_send_gap_recovery(tmp_path):
    idx, _, exact = make_index(tmp_path)
    idx.rotate()
    budget = Budget(idx.store, 1, 0, 1)
    assert budget.reserve(.1, 'lost-before-send', metadata=dict(ids=[0, 1], region=0, origin='scheduled'))
    idx.claim(0, 'lost-before-send')
    idx.close() # actual reserve and partial claim frontier persisted, no queue delivery
    idx = Index(tmp_path)
    budget = Budget(idx.store, 1, 0, 1)
    q = LocalQueue(idx.store)
    scheduler = Scheduler(idx, budget, q)
    scheduler.recover_pending()
    encoder = FakeEncoder(exact)
    worker = LocalWorker(idx, encoder, q, budget)
    worker.drain()
    assert idx.store.node(0)['state'] == idx.store.node(1)['state'] == 'native'
    assert idx.store.get('ledger')['lost-before-send']['reconciled']
    assert encoder.calls == 2
    idx.close()


def test_empty_regions_and_small_support():
    from chrysalis.calibration import sample_regions, regional
    from chrysalis.math import normalize
    old = normalize(np.random.default_rng(3).normal(size=(8, 4)))
    exact = normalize(np.pad(old, ((0, 0), (0, 2))))
    samples = sample_regions(np.zeros(8, dtype=int), 2, 4, 2, 42, region_count=2)
    maps, _ = regional(old, np.zeros(8), samples, lambda ids: exact[ids], exact,
                       rank=64, ridge=1e-6, alpha=.05, seed=42)
    assert maps[0]['status'] == 'insufficient-support'
    assert np.isinf(maps[0]['epsilon'])
    assert maps[1]['status'] == 'empty-region' and np.isinf(maps[1]['gamma'])
