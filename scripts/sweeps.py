"""Actual parameter sweep, with explicit growing per-region sample budget."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--dataset', required=True)
p.add_argument('--output', required=True)
p.add_argument('--config', default='configs/provisional.json')
p.add_argument('--pair', default='P2')
p.add_argument('--device', default='cuda')
p.add_argument('--regions', default='1,64,256,1024')
p.add_argument('--alphas', default='0.1,0.05,0.01')
a = p.parse_args()
root = Path(a.output)
root.mkdir(parents=True, exist_ok=False)
base = json.loads(Path(a.config).read_text())
for R, alpha in [(int(r), .05) for r in a.regions.split(',')] + [(base['R'], float(v)) for v in a.alphas.split(',') if float(v) != .05]:
    cfg = dict(base, R=R, alpha=alpha)
    name = f'R{R}-alpha{alpha}'
    path = root / (name + '.json')
    path.write_text(json.dumps(cfg, indent=2))
    subprocess.run([sys.executable, '-m', 'chrysalis.cli', 'experiment', '--dataset', a.dataset,
                    '--output', str(root / name), '--config', str(path), '--pair', a.pair, '--device', a.device], check=True)
