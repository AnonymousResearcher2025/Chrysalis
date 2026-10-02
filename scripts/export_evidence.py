"""Export inspectable durable ledgers without replacing recorded measurements."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chrysalis.storage import Store

root = Path(sys.argv[1]).resolve()
out = root / 'durable-evidence'
out.mkdir(exist_ok=True)
for label in ['base', 'seed-snapshot'] + [f'{mode}-{s}' for mode in
                                        ['SeedNoResolution', 'SeedAsync', 'SeedSync'] for s in range(1, 6)]:
    store = Store(root / label)
    value = {key: store.get(key) for key in ('meta', 'config', 'inputs', 'epoch', 'epoch_scores',
                                            'work', 'budget', 'ledger', 'repair_queue', 'ambiguous_edges')}
    value['nodes'] = store.nodes()
    value['heat'] = {key: store.get(key) for key in store.keys('heat/')}
    (out / (label + '.json')).write_text(json.dumps(value, indent=2))
    store.close()
source = Path(__file__).resolve().parents[1]
(root / 'postprocessing-source-hashes.json').write_text(json.dumps({
    name: hashlib.sha256((source / name).read_bytes()).hexdigest()
    for name in ['chrysalis/evaluation.py', 'scripts/export_evidence.py']}, indent=2))
