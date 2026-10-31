"""Corpus and query preparation for P1 to P4 (Table 2)."""
import gzip
import hashlib
import io
import json
import tarfile
from concurrent import futures
from pathlib import Path

import numpy as np

from .config import ROOT
from .storage import RawStore

SOURCES = json.loads((ROOT / "configs" / "datasets.json").read_text())


def download(url, dest):
    import requests

    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 22):
                f.write(chunk)
    tmp.rename(dest)
    return dest


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def _finish(out, kind, **meta):
    out = Path(out)
    name = "corpus.jsonl" if kind == "corpus" else "queries.jsonl"
    if kind == "corpus":
        RawStore.index(out)
    meta.update(file=name, sha256=_sha256(out / name))
    (out / "manifest.json").write_text(json.dumps(meta, indent=2))


def _write(out, name, rows):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(out / name, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def wiki1m(out, cache):
    """1M passages drawn uniformly without replacement from DPR's 21M 100-word Wikipedia passages."""
    src = SOURCES["wiki1m"]
    path = download(src["url"], Path(cache) / "psgs_w100.tsv.gz")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        total = sum(1 for _ in f) - 1
    keep = np.sort(np.random.default_rng(src["seed"]).choice(total, src["size"], replace=False))

    def rows():
        j = 0
        with gzip.open(path, "rt", encoding="utf-8") as f:
            next(f)
            for line_no, line in enumerate(f):
                if j < len(keep) and line_no == keep[j]:
                    pid, text, title = line.rstrip("\n").split("\t")
                    yield {"id": j, "text": f"{title}. {text}", "title": title, "source": pid}
                    j += 1

    n = _write(out, "corpus.jsonl", rows())
    _finish(out, "corpus", source=src["url"], sampled=n, population=total, seed=src["seed"])


def nq(out, cache):
    """10K Natural Questions questions drawn from the DPR NQ dev and test splits."""
    src = SOURCES["nq"]
    questions = []
    for url in src["urls"]:
        with open(download(url, Path(cache) / Path(url).name), encoding="utf-8") as f:
            questions += [line.split("\t")[0] for line in f if line.strip()]
    pick = np.random.default_rng(src["seed"]).choice(len(questions), src["size"], replace=False)
    n = _write(out, "queries.jsonl", ({"id": int(j), "text": questions[i]} for j, i in enumerate(pick)))
    _finish(out, "queries", source=src["urls"], sampled=n, population=len(questions), seed=src["seed"])


def msmarco(out, cache, queries_out):
    """The 8,841,823 MS MARCO passages and the 6,980 dev (small) queries."""
    src = SOURCES["msmarco"]
    path = download(src["url"], Path(cache) / "collectionandqueries.tar.gz")
    with tarfile.open(path) as tar:
        coll = io.TextIOWrapper(tar.extractfile(src["collection"]), encoding="utf-8")
        n = _write(out, "corpus.jsonl", ({"id": j, "text": t, "source": int(p)}
                                         for j, (p, t) in enumerate(l.rstrip("\n").split("\t", 1) for l in coll)))
        dev = io.TextIOWrapper(tar.extractfile(src["dev"]), encoding="utf-8")
        q = _write(queries_out, "queries.jsonl", ({"id": int(a), "text": b} for a, b in (l.rstrip("\n").split("\t", 1) for l in dev)))
    _finish(out, "corpus", source=src["url"], passages=n)
    _finish(queries_out, "queries", source=src["url"], member=src["dev"], queries=q)


def trec_dl(out, cache):
    """TREC Deep Learning 2019 and 2020 passage-ranking test queries."""
    src = SOURCES["trec-dl"]
    rows = []
    for url in src["urls"]:
        with gzip.open(download(url, Path(cache) / Path(url).name), "rt", encoding="utf-8") as f:
            rows += [{"id": int(a), "text": b, "track": Path(url).name.split("-")[1]}
                     for a, b in (l.rstrip("\n").split("\t", 1) for l in f if l.strip())]
    n = _write(out, "queries.jsonl", rows)
    _finish(out, "queries", source=src["urls"], queries=n)


def laion1m(metadata, out, captions_out, threads=64):
    """1M images and 10K held-out captions sampled from LAION-400M metadata (parquet with URL
    and TEXT columns). Images are stored with their shorter side resized as configured."""
    import pyarrow.parquet as pq
    import requests
    from PIL import Image

    src = SOURCES["laion1m"]
    table = [pq.read_table(p, columns=["URL", "TEXT"]) for p in sorted(Path(metadata).glob("*.parquet"))]
    urls = [u for t in table for u in t.column("URL").to_pylist()]
    texts = [x for t in table for x in t.column("TEXT").to_pylist()]
    order = np.random.default_rng(src["seed"]).permutation(len(urls))
    out = Path(out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    def fetch(i):
        try:
            r = requests.get(urls[i], timeout=10)
            r.raise_for_status()
            with Image.open(io.BytesIO(r.content)) as im:
                im = im.convert("RGB")
                scale = src["short_side"] / min(im.size)
                im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.BICUBIC)
                name = f"images/{i:09d}.jpg"
                im.save(out / name, quality=95)
            return i, name
        except Exception:
            return i, None

    kept, pos = [], 0
    with futures.ThreadPoolExecutor(threads) as ex:
        while len(kept) < src["size"] and pos < len(order):
            chunk = order[pos:pos + 4 * threads * 64]
            pos += len(chunk)
            for i, name in ex.map(fetch, chunk):
                if name and len(kept) < src["size"]:
                    kept.append((int(i), name))
    n = _write(out, "corpus.jsonl", ({"id": j, "image": name, "caption": texts[i], "source": i}
                                     for j, (i, name) in enumerate(kept)))
    used = {i for i, _ in kept}
    held = [int(i) for i in order[pos:] if texts[i]][: src["captions"]]
    if len(held) < src["captions"]:
        held += [int(i) for i in order[:pos] if int(i) not in used and texts[i]][: src["captions"] - len(held)]
    q = _write(captions_out, "queries.jsonl", ({"id": j, "text": texts[i], "source": i} for j, i in enumerate(held)))
    _finish(out, "corpus", source=str(metadata), images=n, seed=src["seed"], short_side=src["short_side"])
    _finish(captions_out, "queries", source=str(metadata), captions=q, seed=src["seed"])


def read_queries(path):
    with open(Path(path) / "queries.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def prepare(name, root, cache, laion_metadata=None):
    root, cache = Path(root), Path(cache)
    if name == "wiki1m":
        wiki1m(root / "wiki1m", cache)
    elif name == "nq":
        nq(root / "nq", cache)
    elif name == "msmarco":
        msmarco(root / "msmarco", cache, root / "msmarco-dev")
    elif name == "trec-dl":
        trec_dl(root / "trec-dl", cache)
    elif name == "laion1m":
        laion1m(laion_metadata, root / "laion1m", root / "laion-captions")
    elif name == "glove":
        from .config import models

        spec = models()["glove"]
        download(spec["url"], root / "glove" / Path(spec["url"]).name)
    else:
        raise ValueError(name)
