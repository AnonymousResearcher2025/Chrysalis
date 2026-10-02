import numpy as np
import pytest
from chrysalis.backends import LocalQueue
from chrysalis.budget import Budget
from chrysalis.workers import CloudWorker, RpcClient, RpcServer, image_payload
from tests.test_system import FakeEncoder, make_index


def test_cloud_retained_reuse_over_real_rpc(tmp_path):
    idx, _, exact = make_index(tmp_path)
    idx.rotate()
    idx.retain({0: exact[0]})
    budget = Budget(idx.store, 1, 0, 1)
    budget.reserve(.01, 'batch')
    claims = [idx.claim(i, 'batch') for i in [0, 1]]
    queue = LocalQueue(idx.store)
    queue.send(dict(id='batch', claims=claims, raw_locators=[0, 1], origin='scheduled', version='new'))
    class Raw:
        called = []
        def get(self, locator):
            self.called.append(locator)
            return str(locator)
    raw = Raw()
    encoder = FakeEncoder(exact)
    encoder.account, encoder.device = None, 'cpu'
    server = RpcServer('127.0.0.1:0', index=idx, budget=budget)
    host = RpcClient(f'127.0.0.1:{server.port}')
    try:
        worker = CloudWorker(queue, raw, encoder, host)
        assert worker.once()
        assert encoder.calls == 1 and raw.called == [1]
        assert all(idx.store.node(i)['state'] == 'native' for i in [0, 1])
        assert idx.store.get('ledger')['batch']['reconciled']
        assert not queue.receive()
    finally:
        host.close(); server.close(); idx.close()


def test_cloud_query_accounts_direct_encoder_work(tmp_path):
    idx, _, exact = make_index(tmp_path)
    idx.rotate()
    budget = Budget(idx.store, 1, 0, 1)
    encoder = FakeEncoder(exact)
    direct_server = RpcServer('127.0.0.1:0', encoder=encoder)
    direct = RpcClient(f'127.0.0.1:{direct_server.port}')
    host_server = RpcServer('127.0.0.1:0', index=idx, budget=budget, query_worker=direct,
                            raw_provider=lambda i: str(i))
    client = RpcClient(f'127.0.0.1:{host_server.port}')
    try:
        result = client.call('Query', dict(raw='0', k=3, ef=8, rho=1, mode='sync'))
        assert result['requested'] == 1
        records = idx.store.get('work')
        assert {w['category'] for w in records} == {'query', 'boundary'}
        assert all(w['seconds'] > 0 and w['count'] == 1 for w in records)
        assert sum(w['dollars'] for w in records) > 0
    finally:
        client.close(); host_server.close(); direct.close(); direct_server.close(); idx.close()


def test_completion_event_recovers_reconciliation_gap(tmp_path):
    idx, _, _ = make_index(tmp_path)
    budget = Budget(idx.store, 1, 0, 1)
    budget.reserve(.1, 'job')
    idx.store.transaction({'work_event/complete-job': True,
                           'work': [dict(job='job', dollars=.03)]})
    server = RpcServer('127.0.0.1:0', index=idx, budget=budget)
    client = RpcClient(f'127.0.0.1:{server.port}')
    try:
        event = dict(event='complete-job', complete_job='job', category='job_complete', seconds=0, count=0, device='cpu')
        assert not client.call('Work', event)['recorded']
        assert idx.store.get('ledger')['job']['actual'] == pytest.approx(.03)
        assert idx.store.get('budget')['spent'] == pytest.approx(.03)
        client.call('Work', event)
        assert idx.store.get('budget')['spent'] == pytest.approx(.03)
    finally:
        client.close(); server.close(); idx.close()


def test_invalid_budget_and_worker_revision(tmp_path):
    idx, _, exact = make_index(tmp_path)
    with pytest.raises(ValueError):
        Budget(idx.store, 1, 0, float('nan'))
    budget = Budget(idx.store, 1, 0, 1)
    with pytest.raises(ValueError):
        budget.reserve(float('inf'), 'invalid')
    with pytest.raises(ValueError):
        budget.account('query', -1, 1, 'cpu')
    with pytest.raises(ValueError, match='durable budget'):
        Budget(idx.store, 1, 0, 2)
    queue = LocalQueue(idx.store)
    queue.send(dict(version='wrong', claims=[], raw_locators=[]))
    encoder = FakeEncoder(exact)
    worker = CloudWorker(queue, None, encoder, None)
    with pytest.raises(ValueError, match='revision'):
        worker.once()
    assert encoder.calls == 0
    idx.close()


def test_image_rpc_payload_hash(tmp_path):
    import base64
    import hashlib
    from PIL import Image
    path = tmp_path/'image.png'
    Image.new('RGB', (8, 8), (200, 120, 40)).save(path)
    payload = image_payload(path)
    value = base64.b64decode(payload['image_base64'], validate=True)
    assert value == path.read_bytes()
    assert hashlib.sha256(value).hexdigest() == payload['sha256']
