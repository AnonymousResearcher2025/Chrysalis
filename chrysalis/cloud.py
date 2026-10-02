"""Deployable AWS protocol wiring; credentials are supplied through boto3's chain."""
import json
from pathlib import Path
import time
import uuid
from .backends import SQSQueue, S3Raw
from .budget import Budget, Scheduler
from .embeddings import Encoder
from .index import Index
from .workers import RpcClient, RpcServer, CloudWorker, image_payload
from .settings import load as load_settings


class CloudQueue:
    def __init__(self, index, queue, locators):
        self.index, self.queue, self.locators = index, queue, locators

    def send(self, payload):
        claims = payload.get('claims', [payload['claim']] if 'claim' in payload else [])
        payload = dict(payload, claims=claims,
                       raw_locators=[self.locators[str(c['id'])] for c in claims],
                       version=self.index.store.get('meta')['new_version'])
        return self.queue.send(payload)


def credentials(args, server=False):
    import grpc
    if not args.ca:
        if not args.local_insecure:
            raise ValueError('cloud gRPC requires TLS material or explicit --local-insecure for development')
        return None
    ca = Path(args.ca).read_bytes()
    if server:
        return grpc.ssl_server_credentials([(Path(args.key).read_bytes(), Path(args.cert).read_bytes())],
                                           root_certificates=ca, require_client_auth=True)
    return grpc.ssl_channel_credentials(ca, private_key=Path(args.key).read_bytes(), certificate_chain=Path(args.cert).read_bytes())


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['upload', 'host', 'worker', 'snapshot', 'restore'])
    p.add_argument('--index')
    p.add_argument('--bucket', required=True)
    p.add_argument('--prefix', required=True)
    p.add_argument('--raw-version', required=True)
    p.add_argument('--images', action='store_true')
    p.add_argument('--region', default='us-east-1')
    p.add_argument('--endpoint-url')
    p.add_argument('--sqs-url')
    p.add_argument('--address', default='127.0.0.1:50052')
    p.add_argument('--encoder-address', help='direct on-demand gRPC encoder endpoint for Query/sync resolution')
    p.add_argument('--model', default='mpnet')
    p.add_argument('--models', default='configs/models.lock.json')
    p.add_argument('--device', default='cuda')
    p.add_argument('--price-per-hour', type=float)
    p.add_argument('--capacity', type=float)
    p.add_argument('--refill-per-hour', type=float)
    p.add_argument('--ca')
    p.add_argument('--cert')
    p.add_argument('--key')
    p.add_argument('--local-insecure', action='store_true')
    p.add_argument('--output')
    p.add_argument('--snapshot-id')
    a = p.parse_args()
    raw = S3Raw(a.bucket, a.prefix, a.raw_version, a.region, a.endpoint_url)
    if a.mode == 'restore':
        raw.restore_snapshot(a.snapshot_id, a.output)
        return
    index = Index(a.index) if a.mode != 'worker' else None
    try:
        if a.mode == 'upload':
            locators = {}
            for i in range(index.store.get('meta')['count']):
                locators[str(i)] = raw.put_image(i, index.raw(i)) if a.images else raw.put(i, index.raw(i))
                index.store.transaction({'s3_raw': locators})
            print('uploaded versioned raw items', len(locators))
        elif a.mode == 'snapshot':
            print(raw.snapshot(index, a.output, a.snapshot_id or uuid.uuid4().hex))
        elif a.mode == 'host':
            if None in (a.price_per_hour, a.capacity, a.refill_per_hour):
                raise ValueError('explicit prices and token bucket required')
            bucket = Budget(index.store, a.capacity, a.refill_per_hour, a.price_per_hour)
            locators = index.store.get('s3_raw')
            if not locators or not a.sqs_url:
                raise ValueError('upload versioned raw items and provide --sqs-url before hosting')
            lock = load_settings(a.models)
            if index.store.get('meta')['new_version'] != lock[a.model]['revision']:
                raise ValueError('--model must match configured successor')
            queue = CloudQueue(index, SQSQueue(a.sqs_url, a.region, a.endpoint_url), locators)
            def raw_provider(i):
                value = raw.get(locators[str(i)])
                return image_payload(value) if lock[a.model]['kind'] == 'clip' else value
            query_worker = RpcClient(a.encoder_address, credentials(a)) if a.encoder_address else None
            server = RpcServer(a.address, index=index, budget=bucket, query_worker=query_worker, queue_backend=queue,
                               raw_provider=raw_provider,
                               credentials=credentials(a, server=True))
            scheduler = Scheduler(index, bucket, queue)
            try:
                while True:
                    with index.lock:
                        if index.store.recover():
                            index.reload()
                    scheduler.schedule()
                    index.audit()
                    time.sleep(1)
            finally:
                server.close()
                if query_worker:
                    query_worker.close()
        else:
            host = RpcClient(a.address, credentials(a))
            lock = load_settings(a.models)
            def account(category, seconds, count, device):
                host.call('Work', dict(event=uuid.uuid4().hex, category=category, seconds=seconds, count=count, device=device))
            encoder = Encoder(a.model, lock, a.device, account=account)
            worker = CloudWorker(SQSQueue(a.sqs_url, a.region, a.endpoint_url), raw, encoder, host)
            while True:
                worker.once()
    finally:
        if index:
            index.close()


if __name__ == '__main__':
    main()
