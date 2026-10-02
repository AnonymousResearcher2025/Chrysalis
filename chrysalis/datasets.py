"""Explicit, hashed data selection. No claim to recover paper dataset releases."""
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def prepare(corpus, queries, destination, *, dataset_id, release, selection, seed, counts):
    if not dataset_id or not release or not selection:
        raise ValueError('dataset ID, release and selection procedure required')
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=False)
    c, q = read_jsonl(corpus), read_jsonl(queries)
    for row in c:
        if row.get('image_sha256'):
            if sha256(row['raw']) != row['image_sha256']:
                raise ValueError('image byte hash mismatch')
    for rows in (c, q):
        if len({r['id'] for r in rows}) != len(rows):
            raise ValueError('duplicate raw identities')
        if any(not isinstance(r['raw'], str) or not r['raw'] for r in rows):
            raise ValueError('nonempty text or immutable image path required')
    if sum(counts.values()) > len(q):
        raise ValueError('insufficient disjoint query identities')
    rng = np.random.default_rng(seed)
    order, start, pools = rng.permutation(len(q)), 0, {}
    for name, count in counts.items():
        pools[name] = [q[i] for i in order[start:start + count]]
        start += count
    corpus_out = dest / 'corpus.jsonl'
    corpus_out.write_text(''.join(json.dumps(r) + '\n' for r in c), encoding='utf-8')
    queries_out = dest / 'queries.json'
    queries_out.write_text(json.dumps(pools, indent=2), encoding='utf-8')
    manifest = dict(dataset_id=dataset_id, release=release, selection=selection, seed=seed,
                    corpus_count=len(c), query_counts=counts, corpus_sha256=sha256(corpus_out),
                    queries_sha256=sha256(queries_out), source_corpus_sha256=sha256(corpus),
                    source_queries_sha256=sha256(queries), reconstruction=True)
    (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def load(destination):
    dest = Path(destination)
    m = json.loads((dest / 'manifest.json').read_text())
    if sha256(dest / 'corpus.jsonl') != m['corpus_sha256'] or sha256(dest / 'queries.json') != m['queries_sha256']:
        raise ValueError('dataset hash mismatch')
    c = read_jsonl(dest / 'corpus.jsonl')
    for row in c:
        if row.get('image_sha256') and sha256(row['raw']) != row['image_sha256']:
            raise ValueError('image byte hash mismatch')
    return c, json.loads((dest / 'queries.json').read_text()), m


def import_tsv(source, destination, *, raw_column=1, id_column=0):
    """MARCO collection.tsv/dev queries; Wikipedia TSV only with explicit construction."""
    opener = gzip.open if str(source).endswith('.gz') else open
    with opener(source, 'rt', encoding='utf-8') as f, open(destination, 'w', encoding='utf-8') as out:
        for line in f:
            columns = line.rstrip('\n').split('\t')
            out.write(json.dumps(dict(id=columns[id_column], raw=columns[raw_column])) + '\n')


def rare_entities(corpus_annotations, query_annotations):
    """Explicit supplied annotations only. Count document frequency over corpus IDs.

    This counting rule is a reconstruction choice; requires exact annotation
    provenance (extractor/revision/casing/aliases) in the input manifest.
    """
    from collections import Counter
    freq = Counter(e for r in corpus_annotations for e in set(r['entities']))
    if not freq:
        raise ValueError('no annotated entities')
    cutoff = float(np.quantile(list(freq.values()), .1, method='higher'))
    rare = {e for e, n in freq.items() if n <= cutoff}
    return [r['id'] for r in query_annotations if rare.intersection(r['entities'])], dict(freq)
