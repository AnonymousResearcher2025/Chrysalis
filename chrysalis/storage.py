import json
import os
import struct
import zlib
from pathlib import Path

import numpy as np


def crash_point(name):
    """Fault injection: exit abruptly when CHRYSALIS_CRASH names this point."""
    target = os.environ.get("CHRYSALIS_CRASH")
    if not target:
        return
    point, _, count = target.partition("@")
    if point != name:
        return
    seen = int(os.environ.get("CHRYSALIS_CRASH_SEEN", "0")) + 1
    os.environ["CHRYSALIS_CRASH_SEEN"] = str(seen)
    if seen >= int(count or 1):
        os._exit(137)


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_atomic(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


class Manifest:
    """Write-ahead log; one JSON record per line followed by its CRC32."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.touch(exist_ok=True)
        self.seq = 0
        good = 0
        with open(self.path, "rb") as f:
            for line in f:
                rec = self._parse(line)
                if rec is None:
                    break
                good += len(line)
                self.seq = rec["seq"]
        with open(self.path, "r+b") as f:
            f.truncate(good)
        self.f = open(self.path, "ab")

    @staticmethod
    def _parse(line):
        body, sep, crc = line.rstrip(b"\n").rpartition(b"\t")
        if not sep or not line.endswith(b"\n") or zlib.crc32(body) != int(crc or b"-1"):
            return None
        return json.loads(body)

    def append(self, kind, **fields):
        self.seq += 1
        body = json.dumps(dict(seq=self.seq, kind=kind, **fields), separators=(",", ":")).encode()
        self.f.write(body + b"\t" + str(zlib.crc32(body)).encode() + b"\n")
        self.f.flush()
        os.fsync(self.f.fileno())
        return self.seq

    def records(self, after=0):
        with open(self.path, "rb") as f:
            for line in f:
                rec = self._parse(line)
                if rec is None:
                    return
                if rec["seq"] > after:
                    yield rec

    def close(self):
        self.f.close()


class NativeLog:
    """Append-only exact vectors: (node, fence, crc, float32[dim]) per record."""

    HEADER = struct.Struct("<qqI")

    def __init__(self, path, dim):
        self.path = Path(path)
        self.dim = dim
        self.size = self.HEADER.size + 4 * dim
        self.path.touch(exist_ok=True)
        valid = 0
        for _ in self._scan():
            valid += self.size
        with open(self.path, "r+b") as f:
            f.truncate(valid)
        self.f = open(self.path, "ab")

    def _scan(self):
        with open(self.path, "rb") as f:
            while True:
                rec = f.read(self.size)
                if len(rec) < self.size:
                    return
                node, fence, crc = self.HEADER.unpack_from(rec)
                body = rec[self.HEADER.size:]
                if zlib.crc32(body, node & 0xFFFFFFFF) != crc:
                    return
                yield node, fence, np.frombuffer(body, dtype=np.float32)

    @classmethod
    def record_size(cls, dim):
        return cls.HEADER.size + 4 * dim

    @staticmethod
    def truncate(path, size):
        """Drop records the manifest never published."""
        with open(path, "r+b") as f:
            f.truncate(min(size, os.path.getsize(path)))

    def append(self, nodes, fences, vectors):
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        buf = bytearray()
        for node, fence, v in zip(nodes, fences, vectors):
            body = v.tobytes()
            buf += self.HEADER.pack(int(node), int(fence), zlib.crc32(body, int(node) & 0xFFFFFFFF)) + body
        offset = self.f.tell()
        self.f.write(buf)
        self.f.flush()
        os.fsync(self.f.fileno())
        return offset

    def read(self):
        ids, vecs = [], []
        for node, _, v in self._scan():
            ids.append(node)
            vecs.append(v)
        return np.array(ids, dtype=np.int64), (np.stack(vecs) if vecs else np.zeros((0, self.dim), np.float32))

    def close(self):
        self.f.close()


def segment_path(root, s, width):
    return Path(root) / "segments" / f"{s:06d}.w{width}.f32"


def write_segment(root, s, rows):
    rows = np.ascontiguousarray(rows, dtype=np.float32)
    path = segment_path(root, s, rows.shape[1])
    write_atomic(path, rows.tobytes())
    return path


def write_segment_rows(path, local, rows):
    rows = np.ascontiguousarray(rows, dtype=np.float32)
    width = rows.shape[1]
    with open(path, "r+b") as f:
        for j, row in zip(local, rows):
            f.seek(int(j) * 4 * width)
            f.write(row.tobytes())
        f.flush()
        os.fsync(f.fileno())


def read_segment(path, width):
    return np.fromfile(path, dtype=np.float32).reshape(-1, width)


def _npz(**arrays):
    import io

    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def _lists(pairs):
    """(keys, offsets, flat) encoding of {key: list}."""
    keys = list(pairs)
    lens = [len(pairs[k]) for k in keys]
    flat = np.concatenate([np.asarray(pairs[k], np.int32) for k in keys]) if keys else np.zeros(0, np.int32)
    return keys, np.concatenate([[0], np.cumsum(lens)]).astype(np.int64), flat


def save_graph(root, graph):
    """Adjacency of an index: fixed layer-0 rows (updated in place) and the upper layers."""
    write_atomic(Path(root) / "adjacency.i32", np.ascontiguousarray(graph.layer0()).tobytes())
    levels, lists, entry, maxl = graph.upper()
    save_upper(root, np.asarray(levels, np.int8), {(l, i): v for l, i, v in lists}, entry, maxl)


def save_upper(root, levels, upper, entry, maxl):
    keys, off, flat = _lists(upper)
    write_atomic(Path(root) / "upper.npz", _npz(levels=levels, level=np.array([k[0] for k in keys], np.int32),
                                                node=np.array([k[1] for k in keys], np.int32), off=off, flat=flat,
                                                entry=np.array(entry), maxl=np.array(maxl)))


def load_upper(root):
    z = np.load(Path(root) / "upper.npz")
    upper = {(int(l), int(i)): z["flat"][z["off"][j]:z["off"][j + 1]].tolist()
             for j, (l, i) in enumerate(zip(z["level"], z["node"]))}
    return z["levels"].copy(), upper, int(z["entry"]), int(z["maxl"])


def load_graph(root, dim, degree, ef_construction, alpha):
    from . import _core

    rows = np.fromfile(Path(root) / "adjacency.i32", np.int32).reshape(-1, degree + 1)
    levels, upper, entry, maxl = load_upper(root)
    levels = np.concatenate([levels, np.zeros(len(rows) - len(levels), np.int8)])
    return _core.Graph.from_adjacency(dim, degree, ef_construction, alpha, rows, levels,
                                      [(l, i, v) for (l, i), v in upper.items()], entry, maxl)


def apply_edges(root, degree, l0, upper_changes):
    """Write staged layer-0 lists into their rows in place and fold upper-layer changes."""
    path = Path(root) / "adjacency.i32"
    width = 4 * (degree + 1)
    with open(path, "r+b") as f:
        for node, lst in sorted(l0.items()):
            row = np.full(degree + 1, -1, np.int32)
            row[0] = len(lst)
            row[1:1 + len(lst)] = lst
            f.seek(int(node) * width)
            f.write(row.tobytes())
        f.flush()
        os.fsync(f.fileno())
    if upper_changes:
        levels, upper, entry, maxl = load_upper(root)
        for l, i, v in upper_changes:
            upper[(int(l), int(i))] = list(v)
        save_upper(root, levels, upper, entry, maxl)


def save_staged(path, l0, upper):
    keys, off, flat = _lists(l0)
    ukeys, uoff, uflat = _lists({(l, i): v for l, i, v in upper})
    write_atomic(path, _npz(node=np.array(keys, np.int64), off=off, flat=flat,
                            ulevel=np.array([k[0] for k in ukeys], np.int32), unode=np.array([k[1] for k in ukeys], np.int64),
                            uoff=uoff, uflat=uflat))


def load_staged(path):
    z = np.load(path)
    l0 = {int(i): z["flat"][z["off"][j]:z["off"][j + 1]].tolist() for j, i in enumerate(z["node"])}
    upper = [(int(l), int(i), z["uflat"][z["uoff"][j]:z["uoff"][j + 1]].tolist())
             for j, (l, i) in enumerate(zip(z["ulevel"], z["unode"]))]
    return l0, upper


class StagedLog:
    """Staged edge changes of one epoch as length-prefixed binary batches of
    (layer, node, added, removed) deltas against the epoch's view."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.touch(exist_ok=True)
        self.f = open(self.path, "ab")

    def append(self, audited, deltas):
        keys = [(l, i) for l, i, _, _ in deltas]
        _, aoff, aflat = _lists({k: d[2] for k, d in zip(keys, deltas)})
        _, roff, rflat = _lists({k: d[3] for k, d in zip(keys, deltas)})
        blob = _npz(audited=np.asarray(audited, np.int64), layer=np.array([k[0] for k in keys], np.int32),
                    node=np.array([k[1] for k in keys], np.int64), aoff=aoff, aflat=aflat, roff=roff, rflat=rflat)
        self.f.write(struct.pack("<QI", len(blob), zlib.crc32(blob)) + blob)
        self.f.flush()
        os.fsync(self.f.fileno())

    def read(self, base):
        """Audited nodes and staged lists; `base(layer, node)` gives the view's list."""
        import io

        audited, delta = set(), {}
        with open(self.path, "rb") as f:
            while True:
                head = f.read(12)
                if len(head) < 12:
                    break
                size, crc = struct.unpack("<QI", head)
                blob = f.read(size)
                if len(blob) < size or zlib.crc32(blob) != crc:
                    break
                z = np.load(io.BytesIO(blob))
                audited.update(int(x) for x in z["audited"])
                for j, (l, i) in enumerate(zip(z["layer"], z["node"])):
                    delta[(int(l), int(i))] = (z["aflat"][z["aoff"][j]:z["aoff"][j + 1]].tolist(),
                                               z["rflat"][z["roff"][j]:z["roff"][j + 1]].tolist())
        l0, upper = {}, []
        for (l, i), (added, removed) in delta.items():
            gone = set(removed)
            lst = [x for x in base(l, i) if x not in gone] + added
            if l == 0:
                l0[i] = lst
            else:
                upper.append((l, i, lst))
        return audited, l0, upper

    def close(self):
        self.f.close()


class ObjectStore:
    """Local directory or S3 prefix (s3://bucket/prefix)."""

    def __init__(self, url):
        self.url = url
        if url.startswith("s3://"):
            import boto3

            bucket, _, prefix = url[5:].partition("/")
            self.s3, self.bucket, self.prefix = boto3.client("s3"), bucket, prefix.rstrip("/")
        else:
            self.s3, self.root = None, Path(url)
            self.root.mkdir(parents=True, exist_ok=True)

    def put(self, key, data):
        if self.s3:
            self.s3.put_object(Bucket=self.bucket, Key=f"{self.prefix}/{key}", Body=data)
        else:
            (self.root / key).parent.mkdir(parents=True, exist_ok=True)
            write_atomic(self.root / key, data)

    def get(self, key):
        if self.s3:
            return self.s3.get_object(Bucket=self.bucket, Key=f"{self.prefix}/{key}")["Body"].read()
        return (self.root / key).read_bytes()


class RawStore:
    """Raw corpus items by id: a JSON-lines file with a byte-offset index.
    Text items carry "text"; image items carry "image", a path relative to the corpus directory."""

    def __init__(self, directory):
        self.dir = Path(directory)
        self.offsets = np.load(self.dir / "corpus.offsets.npy")
        self.fd = os.open(self.dir / "corpus.jsonl", os.O_RDONLY)

    def __len__(self):
        return len(self.offsets) - 1

    def get(self, ids):
        out = []
        for i in ids:
            start, end = int(self.offsets[i]), int(self.offsets[i + 1])
            rec = json.loads(os.pread(self.fd, end - start, start))
            out.append(rec["text"] if "text" in rec else str(self.dir / rec["image"]))
        return out

    @staticmethod
    def index(directory):
        offsets, pos = [0], 0
        with open(Path(directory) / "corpus.jsonl", "rb") as f:
            for line in f:
                pos += len(line)
                offsets.append(pos)
        np.save(Path(directory) / "corpus.offsets.npy", np.array(offsets, dtype=np.int64))
