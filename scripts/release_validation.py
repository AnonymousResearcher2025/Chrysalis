"""Validate release evidence and optionally check a real encoder/RPC operation.

This is a release check, not a rerun of the paper or the smoke benchmark campaign.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

parser = argparse.ArgumentParser()
parser.add_argument('--encoder-check', action='store_true')
parser.add_argument('--dataset', help='prepared raw dataset for fresh input/encoder verification')
parser.add_argument('--run', help='recorded run directory; defaults to local full run or shipped evidence')
parser.add_argument('--output', default='build/release-validation.json')
args = parser.parse_args()
record = dict(check_type='release validation; no benchmark campaign', paper_experiments_rerun=False)
run = Path(args.run) if args.run else root/('runs/smoke-final' if (root/'runs/smoke-final').exists() else 'evidence/smoke')
old = json.loads((run/'configuration.json').read_text())['dataset']
dataset = Path(args.dataset) if args.dataset else root/'data/release-check'
if dataset.exists():
    from chrysalis.datasets import load
    _, _, new = load(dataset)
    assert all(old[k] == new[k] for k in ['corpus_sha256', 'queries_sha256', 'source_queries_sha256'])
    record['immutable_input_matches_recorded_run'] = True
    if (dataset/'download-manifest.json').exists():
        record['dataset_source'] = json.loads((dataset/'download-manifest.json').read_text())
else:
    record['immutable_input_matches_recorded_run'] = None
    record['input_check_status'] = 'not requested; prepare raw data and pass --dataset for fresh verification'
tests = ET.parse(root/'docs/release-tests.xml').getroot()
record['tests'] = [suite.attrib for suite in tests.findall('testsuite')]
assert all(int(suite['failures']) == 0 and int(suite['errors']) == 0 for suite in record['tests'])
record['recorded_installed_wheel_check'] = json.loads((root/'docs/installed-wheel-check.json').read_text(encoding='utf-8-sig'))
hashes = json.loads((run/'source-hashes.json').read_text())
for name, expected in hashes.items():
    path = run/'executed-sources'/Path(name.replace('\\', '/'))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
record['primary_executed_sources_verified'] = len(hashes)
if args.encoder_check:
    if not dataset.exists():
        raise ValueError('--encoder-check requires a prepared --dataset')
    import numpy as np
    import torch
    from chrysalis.embeddings import Encoder
    from chrysalis.workers import RpcServer, RpcClient
    from chrysalis.settings import load
    torch.set_num_threads(4)
    encoder = Encoder('mpnet', load('configs/models.lock.json'), cache=str(root/'models'))
    raw = [json.loads(line)['raw'] for line in (dataset/'corpus.jsonl').read_text(encoding='utf-8').splitlines()[:2]]
    server = RpcServer('127.0.0.1:0', encoder=encoder)
    client = RpcClient(f'127.0.0.1:{server.port}')
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda item: client.encode_measured([item], encoder.spec['revision'], role='query', category='query'), raw))
        vectors = np.concatenate([r[0] for r in results])
        assert vectors.shape == (2, 768) and np.isfinite(vectors).all()
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6)
        record['actual_encoder_rpc_check'] = dict(model=encoder.spec, raw_inputs=2,
            device='cpu', output_shape=list(vectors.shape), unit_normalized=True,
            work=[r[1] for r in results])
    finally:
        client.close(); server.close()
record['current_source_hashes'] = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
    for folder in ['chrysalis', 'cpp', 'configs', 'scripts', 'tests']
    for p in (root/folder).rglob('*') if p.suffix in ('.py', '.cpp', '.json') and '__pycache__' not in p.parts}
destination = Path(args.output)
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(record, indent=2))
print(json.dumps({k:v for k,v in record.items() if k != 'current_source_hashes'}, indent=2))
