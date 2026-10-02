import hashlib
import io
import json
import numpy as np
import pytest
from botocore.stub import Stubber
from botocore.response import StreamingBody
from chrysalis.backends import SQSQueue, S3Raw
from chrysalis.budget import Budget
from chrysalis.calibration import query_epoch
from chrysalis.index import Index
from chrysalis.workers import RpcServer, RpcClient
from tests.test_system import make_index


def test_interrupted_epoch_restart(tmp_path):
    idx, _, exact = make_index(tmp_path)
    idx.rotate()
    counter = []
    def fail(ids):
        counter.append(ids)
        if len(counter) == 2:
            raise RuntimeError('worker preempted')
        return exact[ids]
    with pytest.raises(RuntimeError):
        query_epoch(idx, exact[:20], [str(i) for i in range(20)], fail, .05, 4, change_threshold=.05)
    assert idx.store.get('epoch_scoring')
    retained_before = set(idx.store.get('retained'))
    assert retained_before
    idx.close()
    idx = Index(tmp_path)
    with pytest.raises(RuntimeError):
        idx.search(exact[0], mode='none')
    called = []
    def encode(ids):
        assert not retained_before.intersection(map(str, ids))
        called.extend(ids)
        return exact[ids]
    query_epoch(idx, exact[:20], [str(i) for i in range(20)], encode, .05, 4, change_threshold=.05)
    assert not idx.store.get('epoch_scoring')
    idx.publish_retained()
    idx.close()


def test_rpc_host_fencing_and_idempotent_money(tmp_path):
    idx, _, exact = make_index(tmp_path)
    idx.rotate()
    budget = Budget(idx.store, 1, 0, {'scheduled': 1, 'boundary': 2, 'job_complete': 0})
    server = RpcServer('127.0.0.1:0', index=idx, budget=budget)
    client = RpcClient(f'127.0.0.1:{server.port}')
    claim = client.call('Claim', dict(id=0, owner='remote'))['claim']
    assert client.call('Claim', dict(id=0, owner='remote'))['claim'] == claim
    assert client.call('Claim', dict(id=0, owner='competitor'))['claim'] is None
    result = client.call('Publish', dict(claim=claim, vector=exact[0].tolist(), version='new', origin='query'))
    assert result['published']
    assert not client.call('Publish', dict(claim=claim, vector=exact[0].tolist(), version='new', origin='query'))['published']
    event = dict(event='unique', category='boundary', seconds=10, count=1, device='cpu')
    assert client.call('Work', event)['recorded']
    assert not client.call('Work', event)['recorded']
    assert len(idx.store.get('work')) == 1
    assert idx.store.get('work')[0]['dollars'] == pytest.approx(20 / 3600)
    client.close()
    server.close()
    idx.close()


def test_aws_sdk_protocol_with_stubs(monkeypatch):
    # This is an SDK wire test, never evidence of an executed AWS experiment.
    monkeypatch.setenv('AWS_ACCESS_KEY_ID', 'testing')
    monkeypatch.setenv('AWS_SECRET_ACCESS_KEY', 'testing')
    monkeypatch.setenv('AWS_EC2_METADATA_DISABLED', 'true')
    q = SQSQueue('https://sqs.us-east-1.amazonaws.com/123456789012/test')
    body = {'id': 'job', 'claims': [], 'origin': 'scheduled'}
    with Stubber(q.client) as stub:
        stub.add_response('send_message', {'MessageId': 'abc'}, {'QueueUrl': q.url, 'MessageBody': json.dumps(body)})
        assert q.send(body) == 'abc'
        stub.add_response('receive_message', {'Messages': [{'MessageId': 'abc', 'ReceiptHandle': 'receipt',
            'Body': json.dumps(body), 'Attributes': {'ApproximateReceiveCount': '2'}}]},
            {'QueueUrl': q.url, 'MaxNumberOfMessages': 1, 'WaitTimeSeconds': 1, 'VisibilityTimeout': 120,
             'AttributeNames': ['ApproximateReceiveCount']})
        msg = q.receive()[0]
        assert msg['attempts'] == 2
        stub.add_response('delete_message', {}, {'QueueUrl': q.url, 'ReceiptHandle': 'receipt'})
        q.ack(msg)
    raw = S3Raw('chrysalis-test', 'raw', 'old')
    value = json.dumps(dict(version='old', item='A real raw item')).encode()
    with Stubber(raw.client) as stub:
        stub.add_response('put_object', {'VersionId': 'v1'}, {'Bucket': raw.bucket, 'Key': 'raw/old/0.json', 'Body': value})
        locator = raw.put(0, 'A real raw item')
        stub.add_response('get_object', {'Body': StreamingBody(io.BytesIO(value), len(value))},
                          {'Bucket': raw.bucket, 'Key': locator['key'], 'VersionId': 'v1'})
        assert raw.get(locator) == 'A real raw item'


def test_snapshot_restore_and_dimensions(tmp_path):
    idx, _, exact = make_index(tmp_path / 'index')
    idx.rotate()
    with pytest.raises(ValueError):
        c = idx.claim(0, 'bad')
        idx.publish_native(c, [1, 0], 'query')
    hashes = idx.store.snapshot(tmp_path / 'snapshot')
    assert hashes and 'raw.jsonl' in hashes
    restored = Index(tmp_path / 'snapshot')
    assert np.allclose([restored.graph.vector(i) for i in range(80)], exact)
    assert restored.graph.check_reverse()
    restored.close()
    idx.close()
