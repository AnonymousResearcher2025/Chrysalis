"""Capture available run sources only when they match the recorded execution SHA.

Already captured sources are preserved; changed files are never relabeled as the
executed version. This tool does not recover the historical lost prototype.
"""
import hashlib
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
run = Path(sys.argv[1]).resolve()
expected = json.loads((run / 'source-hashes.json').read_text())
dest = run / 'executed-sources'
for name, digest in expected.items():
    path = root / Path(name)
    target = dest / Path(name)
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == digest:
        continue
    value = path.read_bytes()
    # Later metric/CLI additions may differ; this helper must never label them
    # as executed unless their original snapshot is actually available.
    if hashlib.sha256(value).hexdigest() != digest:
        print('source changed after execution:', name)
        continue
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(value)
