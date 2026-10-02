import concurrent.futures
import json
import queue
import threading
import time
import uuid
import grpc
import numpy as np
import base64
import hashlib
import math
from pathlib import Path


def encode_json(x):
    return json.dumps(x).encode()


def decode_json(x):
    return json.loads(x)


class Batcher:
    """Aggregate concurrent direct gRPC resolutions for a short bounded window."""
    def __init__(self, encoder, max_batch=32, delay=.003):
        self.encoder, self.max_batch, self.delay = encoder, max_batch, delay
        self.queue = queue.Queue()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, raw, category='boundary', role='corpus'):
        future = concurrent.futures.Future()
        self.queue.put((raw, category, role, future))
        return future

    def _run(self):
        while not self.stop.is_set():
            try:
                batch = [self.queue.get(timeout=.05)]
            except queue.Empty:
                continue
            deadline = time.monotonic() + self.delay
            while len(batch) < self.max_batch:
                try:
                    batch.append(self.queue.get(timeout=max(0, deadline - time.monotonic())))
                except queue.Empty:
                    break
            groups = {}
            for request in batch:
                groups.setdefault((request[1], request[2]), []).append(request)
            for (category, role), requests in groups.items():
                try:
                    started = time.perf_counter()
                    vectors = self.encoder.encode([b[0] for b in requests], category=category, role=role)
                    seconds = (time.perf_counter() - started) / len(requests)
                    for vector, (_, _, _, f) in zip(vectors, requests):
                        f.set_result(dict(vector=vector.tolist(), seconds=seconds,
                                          device=getattr(self.encoder, 'device', 'cpu')))
                except Exception as e:
                    for _, _, _, f in requests:
                        f.set_exception(e)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=10)


class RpcServer:
    """Real gRPC, explicit JSON wire schema, optional mTLS in cloud deployments."""
    def __init__(self, address, *, encoder=None, index=None, budget=None, query_worker=None, queue_backend=None, raw_provider=None, credentials=None):
        self.server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=32),
                                  options=[('grpc.max_receive_message_length', 64 * 1024 * 1024)])
        handlers = {}
        self.batcher = Batcher(encoder) if encoder else None
        if encoder:
            def encode(req, ctx):
                if req['version'] != encoder.spec['revision']:
                    ctx.abort(grpc.StatusCode.FAILED_PRECONDITION, 'model version mismatch')
                role, category = req.get('role', 'corpus'), req.get('category', 'boundary')
                if role not in ('corpus', 'query'):
                    ctx.abort(grpc.StatusCode.INVALID_ARGUMENT, 'invalid embedding role')
                values = [self.batcher.submit(item, category=category, role=role) for item in req['raw']]
                measured = [f.result() for f in values]
                return dict(vectors=[r['vector'] for r in measured], work=dict(
                    category=category, seconds=sum(r['seconds'] for r in measured), count=len(measured),
                    device=measured[0]['device'] if measured else getattr(encoder, 'device', 'cpu'),
                    basis='encoder batch wall time apportioned across aggregated items'))
            handlers['Encode'] = encode
        if index:
            def claim(req, ctx):
                with index.lock:
                    n = index.store.node(req['id'])
                    if n['state'] == 'resolving' and n['owner'] == req['owner'] and n['expires'] > time.time():
                        c = dict(id=req['id'], owner=n['owner'], fence=n['fence'], expires=n['expires'])
                    else:
                        c = index.claim(req['id'], req['owner'], req.get('lease', 120))
                    return dict(claim=c, native=index.store.node(req['id'])['state'] == 'native')
            def publish(req, ctx):
                if req['version'] != index.store.get('meta')['new_version']:
                    ctx.abort(grpc.StatusCode.FAILED_PRECONDITION, 'successor mismatch')
                return dict(published=index.publish_native(req['claim'], req['vector'], req['origin']))
            handlers.update(Claim=claim, Publish=publish)
            def acquire(req, ctx):
                c = index.acquire(req['claim'], req['worker'], req.get('lease', 120))
                retained = index.retained(req['claim']['id']) if c else None
                return dict(claim=c, native=index.store.node(req['claim']['id'])['state'] == 'native',
                            retained_vector=retained.tolist() if retained is not None else None)
            handlers['Acquire'] = acquire
            if query_worker:
                def query(req, ctx):
                    version = index.store.get('meta')['new_version']
                    def remote_encode(raw, role='corpus', category='boundary'):
                        vectors, measurement = query_worker.encode_measured(raw, version, role=role, category=category)
                        if budget and measurement:
                            budget.account(measurement['category'], measurement['seconds'], measurement['count'], measurement['device'])
                        return vectors
                    q = remote_encode([req['raw']], role='query', category='query')[0]
                    return index.search(q, k=req.get('k', 10), ef=req.get('ef', 96), rho=req.get('rho', 4),
                        mode=req.get('mode', 'async'), enqueue=queue_backend.send if queue_backend else None,
                        resolver=lambda i: remote_encode([raw_provider(i) if raw_provider else index.raw(i)])[0])
                handlers['Query'] = query
            if budget:
                def work(req, ctx):
                    if not math.isfinite(req['seconds']) or req['seconds'] < 0 or req['count'] < 0:
                        ctx.abort(grpc.StatusCode.INVALID_ARGUMENT, 'invalid measured work')
                    with index.lock:
                        event_key = 'work_event/' + req['event']
                        if index.store.get(event_key):
                            # A crash may occur after the event write and before
                            # reconciliation. Redelivery must finish that step.
                            if req.get('complete_job'):
                                total = sum(w['dollars'] for w in index.store.get('work', []) if w.get('job') == req['complete_job'])
                                budget.reconcile(req['complete_job'], total)
                            return dict(recorded=False)
                        # One atomic ledger transaction avoids duplicate transport accounting.
                        dollars = req['seconds'] / 3600 * budget.price_for(req['category'])
                        records = index.store.get('work', [])
                        records.append(dict(category=req['category'], seconds=req['seconds'], count=req['count'],
                                            device=req['device'], dollars=dollars, job=req.get('job'),
                                            gpu_busy_seconds=req['seconds'] if req['device'].startswith('cuda') else 0))
                        index.store.transaction({'work': records, event_key: True})
                        if req.get('complete_job'):
                            total = sum(w['dollars'] for w in records if w.get('job') == req['complete_job'])
                            budget.reconcile(req['complete_job'], total)
                    return dict(recorded=True)
                handlers['Work'] = work
        generic = {name: grpc.unary_unary_rpc_method_handler(fn, request_deserializer=decode_json,
                                                           response_serializer=encode_json)
                   for name, fn in handlers.items()}
        self.server.add_generic_rpc_handlers([grpc.method_handlers_generic_handler('chrysalis.Worker', generic)])
        self.port = self.server.add_secure_port(address, credentials) if credentials else self.server.add_insecure_port(address)
        if self.port == 0:
            raise RuntimeError('gRPC bind failed')
        self.server.start()

    def close(self):
        self.server.stop(5).wait()
        if self.batcher:
            self.batcher.close()


class RpcClient:
    def __init__(self, address, credentials=None):
        self.channel = grpc.secure_channel(address, credentials) if credentials else grpc.insecure_channel(address)

    def call(self, name, payload, timeout=120):
        return self.channel.unary_unary('/chrysalis.Worker/' + name,
                                       request_serializer=encode_json, response_deserializer=decode_json)(payload, timeout=timeout)

    def encode(self, raw, version, role='corpus', category='boundary'):
        return self.encode_measured(raw, version, role, category)[0]

    def encode_measured(self, raw, version, role='corpus', category='boundary'):
        response = self.call('Encode', dict(raw=raw, version=version, role=role, category=category))
        return np.asarray(response['vectors'], dtype='float32'), response.get('work')

    def close(self):
        self.channel.close()


class LocalWorker:
    def __init__(self, index, encoder, transport, budget=None):
        self.index, self.encoder, self.transport, self.budget = index, encoder, transport, budget
        self.owner = uuid.uuid4().hex
        self.processing = threading.Lock()

    def process(self, message):
        with self.processing:
            return self._process(message)

    def _process(self, message):
        payload = message['payload']
        if self.encoder.spec['revision'] != self.index.store.get('meta')['new_version']:
            raise ValueError('worker encoder revision differs from successor')
        claims = payload.get('claims', [payload['claim']] if 'claim' in payload else [])
        active, raw, ready = [], [], []
        job = payload.get('id')
        for claim in claims:
            i = claim['id']
            n = self.index.store.node(i)
            if n['state'] == 'native':
                continue
            claim = self.index.acquire(claim, self.owner)
            if not claim:
                # Another live physical worker owns it; duplicate deliveries wait.
                return False
            retained = self.index.retained(i)
            if retained is not None:
                ready.append((claim, retained))
            else:
                active.append(claim)
                raw.append(self.index.raw(i))
        started = time.perf_counter()
        if raw:
            original_account = getattr(self.encoder, 'account', None)
            if self.budget:
                self.encoder.account = lambda category, seconds, count, device: self.budget.account(category, seconds, count, device, job=job)
            try:
                vectors = self.encoder.encode(raw, category=payload['origin'])
            finally:
                self.encoder.account = original_account
            ready.extend(zip(active, vectors))
        seconds = time.perf_counter() - started
        for claim, vector in ready:
            self.index.publish_native(claim, vector, payload['origin'])
        if all(self.index.store.node(c['id'])['state'] == 'native' for c in claims):
            if self.budget and job and job in self.index.store.get('ledger', {}):
                measurements = [w for w in self.index.store.get('work', []) if w.get('job') == job]
                dollars = sum(w['dollars'] for w in measurements) if measurements else seconds / 3600 * self.budget.price
                self.budget.reconcile(job, dollars)
            self.transport.ack(message)
            return True
        return False

    def drain(self):
        done = 0
        while messages := self.transport.receive(1):
            if not self.process(messages[0]):
                break
            done += 1
        return done


class CloudWorker:
    """SQS delivery -> versioned S3 raw -> actual encoder -> fenced host RPC."""
    def __init__(self, sqs, raw_backend, encoder, host):
        self.sqs, self.raw, self.encoder, self.host = sqs, raw_backend, encoder, host
        self.owner = uuid.uuid4().hex
        self.processing = threading.Lock()

    def once(self):
        with self.processing:
            return self._once()

    def _once(self):
        messages = self.sqs.receive(1)
        if not messages:
            return False
        message = messages[0]
        p = message['payload']
        if p['version'] != self.encoder.spec['revision']:
            raise ValueError('queued successor differs from worker encoder revision')
        if len(p['claims']) != len(p['raw_locators']):
            raise ValueError('SQS raw locator/claim lengths differ')
        records, ready = [], []
        for c, locator in zip(p['claims'], p['raw_locators']):
            result = self.host.call('Acquire', dict(claim=c, worker=self.owner))
            if result['native']:
                continue
            if not result['claim']:
                return False
            c = result['claim']
            if result.get('retained_vector') is not None:
                ready.append((c, result['retained_vector']))
            else:
                records.append((c, self.raw.get(locator)))
        if records:
            before_account = self.encoder.account
            job = p.get('id')
            self.encoder.account = lambda category, seconds, count, device: self.host.call('Work', dict(
                event=uuid.uuid4().hex, job=job, category=category, seconds=seconds, count=count, device=device))
            try:
                vectors = self.encoder.encode([r for _, r in records], category=p['origin'])
            finally:
                self.encoder.account = before_account
            ready.extend((c, x.tolist()) for (c, _), x in zip(records, vectors))
        for c, x in ready:
            result = self.host.call('Publish', dict(claim=c, vector=x, origin=p['origin'], version=p['version']))
            if not result['published']:
                return False
        if p.get('id'):
            self.host.call('Work', dict(event='complete-' + p['id'], complete_job=p['id'], category='job_complete',
                                      seconds=0, count=0, device=self.encoder.device))
        # Redelivery is harmless; host fencing validates every publication. Query
        # claims may be active before enqueue: initial workers may use that claim.
        self.sqs.ack(message)
        return True


def image_payload(path):
    """Transport immutable image bytes to a remote direct encoder."""
    data = Path(path).read_bytes()
    return dict(image_base64=base64.b64encode(data).decode('ascii'), sha256=hashlib.sha256(data).hexdigest())
