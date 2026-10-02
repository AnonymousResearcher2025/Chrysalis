"""RocksDB authoritative manifest with data-first durable publication.

Single index-host process owns RocksDB's exclusive lock. All external workers
publish through that host. A process-level RLock serializes durable transactions
and their C++ mirrors; the GIL is not the ownership protocol.
"""
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import uuid
import numpy as np
from rocksdict import Rdict, Options, WriteBatch, WriteOptions, Checkpoint


def dumps(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=True)


def sync_dir(path):
    if os.name != 'nt':
        fd = os.open(path, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    # Windows: newly created data files are flushed via fsync/FlushFileBuffers;
    # rename directory durability under sudden power loss is not independently
    # proven here. Orphan recovery handles interrupted process publication.


class Store:
    def __init__(self, root, crash=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / 'vectors').mkdir(exist_ok=True)
        opts = Options()
        opts.create_if_missing(True)
        self.db = Rdict(str(self.root / 'manifest'), opts)
        self.write_options = WriteOptions()
        self.write_options.sync = True
        self.write_options.disable_wal = False
        self.lock = threading.RLock()
        self.crash = crash or (lambda stage: None)
        self.peak_bytes = self.get('peak_bytes', 0)

    def get(self, key, default=None):
        value = self.db.get(key)
        return default if value is None else json.loads(value)

    def keys(self, prefix):
        return [k for k in self.db.keys() if isinstance(k, str) and k.startswith(prefix)]

    def transaction(self, updates, deletes=()):
        with self.lock:
            batch = WriteBatch()
            for k, v in updates.items():
                batch[k] = dumps(v)
            for k in deletes:
                batch.delete(k)
            self.db.write(batch, self.write_options)

    def write_array(self, x, kind='segment'):
        name = f'vectors/{kind}-{uuid.uuid4().hex}.npy'
        path = self.root / name
        with path.open('wb') as f:
            np.save(f, np.asarray(x, dtype=np.float32), allow_pickle=False)
            f.flush()
            os.fsync(f.fileno())
        sync_dir(path.parent)
        self.measure()
        self.crash('data_sync')
        return name

    def vector(self, node):
        arr = np.load(self.root / node['file'], allow_pickle=False)
        return arr[node['row']].copy()

    def node(self, i):
        n = self.get(f'node/{i}')
        if n is None:
            raise KeyError(i)
        return n

    def nodes(self):
        return [self.node(i) for i in range(self.get('meta')['count'])]

    def publish(self, updates):
        self.transaction(updates)
        self.crash('manifest_sync')

    def reclaim(self):
        # Only known managed vector directory, never arbitrary manifest paths.
        refs = {n['file'] for n in self.nodes()} if self.get('meta') else set()
        retained = self.get('retained', {})
        refs.update(v['file'] for v in retained.values())
        for path in (self.root / 'vectors').glob('*.npy'):
            rel = path.relative_to(self.root).as_posix()
            if rel not in refs:
                self.crash('before_reclaim')
                path.unlink()
        sync_dir(self.root / 'vectors')
        self.measure()

    def claim(self, i, owner, seconds, now=None):
        if seconds <= 0:
            raise ValueError('positive lease required')
        now = time.time() if now is None else now
        with self.lock:
            n = self.node(i)
            if n['state'] == 'native':
                return None
            if n['state'] == 'legacy':
                raise RuntimeError('rotation required before claim')
            if n['state'] == 'resolving' and n['expires'] > now:
                return None
            n.update(state='resolving', owner=owner, fence=n['fence'] + 1, expires=now + seconds)
            self.transaction({f'node/{i}': n})
            return dict(id=i, owner=owner, fence=n['fence'], expires=n['expires'])

    def native(self, claim, vector, origin, now=None, reclaim=True):
        now = time.time() if now is None else now
        i = claim['id']
        with self.lock:
            n = self.node(i)
            if n['state'] == 'native':
                return False
            if (n['state'] != 'resolving' or n['owner'] != claim['owner'] or
                    n['fence'] != claim['fence'] or n['expires'] <= now):
                return False
            path = self.write_array([vector], 'native')
            previous_origin = n['origin']
            n.update(file=path, row=0, state='native', version=self.get('meta')['new_version'],
                     origin=origin, owner=None, expires=0)
            meta = self.get('meta')
            meta['flips'] += 1
            meta['routing_revision'] = meta.get('routing_revision', 0) + 1
            counts = meta.setdefault('origin_counts', {})
            counts[previous_origin] = counts.get(previous_origin, 0) - 1
            counts[origin] = counts.get(origin, 0) + 1
            queue = sorted(set(self.get('repair_queue', []) + [i]))
            self.publish({f'node/{i}': n, 'meta': meta, 'repair_queue': queue})
            if reclaim:
                self.reclaim()
            return True

    def acquire(self, claim, worker, seconds=120, now=None):
        """Bind a queued claim to one physical worker; duplicates cannot share it."""
        now = time.time() if now is None else now
        with self.lock:
            i = claim['id']
            n = self.node(i)
            if n['state'] == 'native':
                return None
            if n['state'] == 'resolving' and n['expires'] > now:
                if n['owner'] == worker:
                    return dict(id=i, owner=worker, fence=n['fence'], expires=n['expires'])
                if n['owner'] != claim['owner'] or n['fence'] != claim['fence']:
                    return None
                n.update(owner=worker, fence=n['fence'] + 1, expires=now + seconds)
                self.transaction({f'node/{i}': n})
                return dict(id=i, owner=worker, fence=n['fence'], expires=n['expires'])
            return self.claim(i, worker, seconds, now)

    def recover(self, now=None):
        now = time.time() if now is None else now
        with self.lock:
            updates = {}
            if self.get('meta'):
                for i, n in enumerate(self.nodes()):
                    if n['state'] == 'resolving' and n['expires'] <= now:
                        n.update(state='bridged', owner=None, expires=0, fence=n['fence'] + 1)
                        updates[f'node/{i}'] = n
            self.transaction(updates)
            self.reclaim()
            return len(updates)

    def measure(self):
        total = sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())
        self.peak_bytes = max(self.peak_bytes, total)
        # Avoid recursive measurement. WAL, temporary segments, maps, raw and
        # retained files all live underneath root and are counted.
        self.transaction({'peak_bytes': self.peak_bytes})
        return total

    def snapshot(self, destination):
        """Checkpoint + referenced vectors under host lock; restores as one directory."""
        import shutil
        dest = Path(destination)
        if dest.exists():
            raise FileExistsError(dest)
        with self.lock:
            dest.mkdir(parents=True)
            Checkpoint(self.db).create_checkpoint(str(dest / 'manifest'))
            shutil.copytree(self.root / 'vectors', dest / 'vectors')
            for name in ('raw.jsonl', 'inputs.json'):
                if (self.root / name).exists():
                    shutil.copy2(self.root / name, dest / name)
            hashes = {str(p.relative_to(dest)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in dest.rglob('*') if p.is_file()}
            (dest / 'SNAPSHOT.json').write_text(dumps(hashes), encoding='utf-8')
        return hashes

    def close(self):
        self.db.flush_wal(True)
        self.db.close()

    def compact(self):
        with self.lock:
            self.measure()
            self.db.flush()
            self.db.compact_range(None, None)
            self.db.flush_wal(True)
            self.measure()
