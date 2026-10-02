"""Local and real AWS transports implementing at-least-once job delivery."""
import hashlib
import json
from pathlib import Path
import time
import uuid


class LocalQueue:
    def __init__(self, store, visibility=120):
        self.store, self.visibility = store, visibility

    def send(self, payload):
        ident = payload.get('id', uuid.uuid4().hex)
        with self.store.lock:
            key = 'job/' + ident
            if self.store.get(key) is None:
                self.store.transaction({key: dict(payload=payload, visible=0, receipt=None, attempts=0)})
        return ident

    def receive(self, count=1, now=None):
        now = time.time() if now is None else now
        out = []
        with self.store.lock:
            for key in sorted(self.store.keys('job/')):
                job = self.store.get(key)
                if job.get('done') or job['visible'] > now:
                    continue
                job.update(receipt=uuid.uuid4().hex, visible=now + self.visibility, attempts=job['attempts'] + 1)
                self.store.transaction({key: job})
                out.append(dict(key=key, receipt=job['receipt'], payload=job['payload'], attempts=job['attempts']))
                if len(out) >= count:
                    break
        return out

    def ack(self, message):
        with self.store.lock:
            job = self.store.get(message['key'])
            if job['receipt'] == message['receipt']:
                job['done'] = True
                self.store.transaction({message['key']: job})
                return True
        return False


class SQSQueue:
    def __init__(self, url, region='us-east-1', endpoint_url=None, visibility=120):
        import boto3
        self.client = boto3.client('sqs', region_name=region, endpoint_url=endpoint_url)
        self.url, self.visibility = url, visibility

    def send(self, payload):
        return self.client.send_message(QueueUrl=self.url, MessageBody=json.dumps(payload))['MessageId']

    def receive(self, count=1, now=None):
        result = self.client.receive_message(QueueUrl=self.url, MaxNumberOfMessages=min(10, count),
                                             WaitTimeSeconds=1, VisibilityTimeout=self.visibility,
                                             AttributeNames=['ApproximateReceiveCount'])
        return [dict(receipt=m['ReceiptHandle'], payload=json.loads(m['Body']),
                     attempts=int(m['Attributes'].get('ApproximateReceiveCount', 1)))
                for m in result.get('Messages', [])]

    def ack(self, message):
        self.client.delete_message(QueueUrl=self.url, ReceiptHandle=message['receipt'])
        return True


class S3Raw:
    def __init__(self, bucket, prefix, version, region='us-east-1', endpoint_url=None):
        import boto3
        self.client = boto3.client('s3', region_name=region, endpoint_url=endpoint_url)
        self.bucket, self.prefix, self.version = bucket, prefix.rstrip('/'), version

    def put(self, i, item):
        body = json.dumps(dict(version=self.version, item=item)).encode()
        result = self.client.put_object(Bucket=self.bucket, Key=f'{self.prefix}/{self.version}/{i}.json', Body=body)
        return dict(bucket=self.bucket, key=f'{self.prefix}/{self.version}/{i}.json',
                    version_id=result.get('VersionId'), sha256=hashlib.sha256(body).hexdigest())

    def put_image(self, i, path):
        body = Path(path).read_bytes()
        key = f'{self.prefix}/{self.version}/{i}.image'
        result = self.client.put_object(Bucket=self.bucket, Key=key, Body=body)
        locator = dict(bucket=self.bucket, key=key, version_id=result.get('VersionId'), sha256=hashlib.sha256(body).hexdigest())
        return self.put(i, dict(image_locator=locator))

    def get(self, locator):
        args = dict(Bucket=locator['bucket'], Key=locator['key'])
        if locator.get('version_id'):
            args['VersionId'] = locator['version_id']
        body = self.client.get_object(**args)['Body'].read()
        if hashlib.sha256(body).hexdigest() != locator['sha256']:
            raise ValueError('raw S3 checksum mismatch')
        record = json.loads(body)
        if record['version'] != self.version:
            raise ValueError('raw S3 version mismatch')
        item = record['item']
        if isinstance(item, dict) and 'image_locator' in item:
            loc = item['image_locator']
            args = dict(Bucket=loc['bucket'], Key=loc['key'])
            if loc.get('version_id'):
                args['VersionId'] = loc['version_id']
            image = self.client.get_object(**args)['Body'].read()
            if hashlib.sha256(image).hexdigest() != loc['sha256']:
                raise ValueError('S3 image checksum mismatch')
            cache = Path('data/s3-images')
            cache.mkdir(parents=True, exist_ok=True)
            path = cache / (loc['sha256'] + '.image')
            if not path.exists():
                with path.open('wb') as f:
                    f.write(image); f.flush()
                    import os
                    os.fsync(f.fileno())
            return str(path.resolve())
        return item

    def snapshot(self, index, directory, snapshot_id):
        directory = Path(directory)
        hashes = index.store.snapshot(directory)
        for relative in hashes:
            self.client.upload_file(str(directory / relative), self.bucket,
                                    f'{self.prefix}/snapshots/{snapshot_id}/{relative}')
        # Commit marker uploaded last; incomplete uploads never count as snapshots.
        self.client.upload_file(str(directory / 'SNAPSHOT.json'), self.bucket,
                                f'{self.prefix}/snapshots/{snapshot_id}/SNAPSHOT.json')
        return snapshot_id

    def restore_snapshot(self, snapshot_id, destination):
        destination = Path(destination).resolve()
        if destination.exists():
            raise FileExistsError(destination)
        base = f'{self.prefix}/snapshots/{snapshot_id}/'
        body = self.client.get_object(Bucket=self.bucket, Key=base + 'SNAPSHOT.json')['Body'].read()
        hashes = json.loads(body)
        destination.mkdir(parents=True)
        for relative, digest in hashes.items():
            path = (destination / relative).resolve()
            if not path.is_relative_to(destination):
                raise ValueError('unsafe snapshot path')
            path.parent.mkdir(parents=True, exist_ok=True)
            self.client.download_file(self.bucket, base + relative, str(path))
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError('snapshot checksum mismatch')
        (destination / 'SNAPSHOT.json').write_bytes(body)
