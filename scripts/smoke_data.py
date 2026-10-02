"""Prepare AG News smoke input from an immutable dataset revision."""
import argparse
import json
from pathlib import Path
import re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from huggingface_hub import HfApi, hf_hub_download
from chrysalis.datasets import load, prepare, sha256

RELEASE = 'eb185aade064a813bc0b7f42de02595523103ca4'
DATASET = 'fancyzhx/ag_news'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('destination', nargs='?', default='data/smoke')
    parser.add_argument('--revision', default=RELEASE)
    args = parser.parse_args()
    if not re.fullmatch(r'[0-9a-f]{40}', args.revision):
        raise ValueError('immutable dataset commit SHA required')
    dest = Path(args.destination)
    if dest.exists():
        _, _, manifest = load(dest)
        if manifest['dataset_id'] != DATASET or manifest['release'] != args.revision:
            raise ValueError('existing dataset differs from requested release')
        print(json.dumps(manifest, indent=2))
        return
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise SystemExit('Install dataset preparation dependencies: pip install -r requirements-datasets.txt') from error
    cache = dest.parent / 'agnews-source'
    cache.mkdir(parents=True, exist_ok=True)
    names = sorted(p for p in HfApi().list_repo_files(DATASET, repo_type='dataset', revision=args.revision)
                   if p.startswith('data/train-') and p.endswith('.parquet'))
    rows, sources = [], []
    for name in names:
        path = hf_hub_download(DATASET, name, repo_type='dataset', revision=args.revision, cache_dir=str(cache/'hf'))
        sources.append(dict(path=name, sha256=sha256(path)))
        for batch in parquet.ParquetFile(path).iter_batches(batch_size=300):
            for value in batch.to_pylist():
                rows.append(dict(id=f'train:{len(rows)}', raw=value['text']))
                if len(rows) == 300:
                    break
            if len(rows) == 300:
                break
        if len(rows) == 300:
            break
    if len(rows) != 300:
        raise ValueError('pinned dataset lacks the required 300 train rows')
    corpus, queries = cache/'corpus.jsonl', cache/'queries.jsonl'
    corpus.write_text(''.join(json.dumps(x) + '\n' for x in rows[:160]), encoding='utf-8')
    queries.write_text(''.join(json.dumps(x) + '\n' for x in rows[160:]), encoding='utf-8')
    manifest = prepare(corpus, queries, dest, dataset_id=DATASET, release=args.revision,
                       selection='First 160 train rows corpus; next 140 train texts query pools; AG News reconstruction smoke only',
                       seed=42, counts=dict(residual=20, replay=20, offline=20, evaluation=32))
    (dest/'download-manifest.json').write_text(json.dumps(dict(revision=args.revision, sources=sources), indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
