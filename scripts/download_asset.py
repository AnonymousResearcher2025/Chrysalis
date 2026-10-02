"""Streaming checksum-pinned public download; never silently overwrite an asset."""
import argparse
import hashlib
from pathlib import Path
import requests
p = argparse.ArgumentParser()
p.add_argument('--url', required=True)
p.add_argument('--sha256', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
path = Path(a.output).resolve()
if path.exists():
    raise FileExistsError(path)
path.parent.mkdir(parents=True, exist_ok=True)
tmp = path.with_name(path.name + '.partial')
if tmp.exists():
    raise FileExistsError(tmp)
h = hashlib.sha256()
with requests.get(a.url, stream=True, timeout=60) as r, tmp.open('wb') as f:
    r.raise_for_status()
    for block in r.iter_content(1024 * 1024):
        h.update(block); f.write(block)
    f.flush()
    import os
    os.fsync(f.fileno())
if h.hexdigest().lower() != a.sha256.lower():
    raise ValueError('checksum mismatch; retained .partial for investigation')
tmp.replace(path)
print(path)
