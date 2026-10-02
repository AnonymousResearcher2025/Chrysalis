"""Measured post-convergence serving without adapters, intervals or durable heat."""
import json
from pathlib import Path
import sys
import hashlib
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chrysalis.datasets import load
from chrysalis.embeddings import Encoder
from chrysalis.evaluation import Oracle, replay
from chrysalis.index import Index
import torch
torch.set_num_threads(4)
root = Path(sys.argv[1]).resolve()
dataset = sys.argv[2]
cfg = json.loads((root/'configuration.json').read_text())
conf = cfg['config']
corpus, pools, _ = load(dataset)
encoder = Encoder('mpnet', cfg['models'])
oracle = Oracle(__import__('numpy').load(root/'oracle/vectors.npy'))
pool = pools['evaluation']
for seed in conf['replay_seeds']:
    idx = Index(root/f'SeedAsync-{seed}')
    assert idx.store.get('meta')['retired']
    replay([r['raw'] for r in pool], [r['id'] for r in pool], encoder,
           lambda q:idx.search(q, k=conf['k'], ef=conf['ef']), oracle,
           root/f'queries-NativeRetiredFastPath-{seed}.jsonl', k=conf['k'], length=conf['replay_length'], seed=seed,
           label='NativeRetiredFastPath')
    idx.close()
(root/'retired-source-hashes.json').write_text(json.dumps({
    name:hashlib.sha256((Path(__file__).resolve().parents[1]/name).read_bytes()).hexdigest()
    for name in ['chrysalis/index.py','scripts/evaluate_retired.py']},indent=2))
