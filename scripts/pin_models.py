"""Resolve public model commits once; experiments consume only the saved lock."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from huggingface_hub import HfApi
from chrysalis.embeddings import MODEL_SPECS

dest = Path(sys.argv[1] if len(sys.argv) > 1 else 'configs/models.lock.json')
if dest.exists():
    raise FileExistsError('refusing to silently repin models')
api = HfApi()
lock = {}
for key, (repo, kind, limit) in MODEL_SPECS.items():
    info = api.model_info(repo)
    lock[key] = dict(id=repo, revision=info.sha, kind=kind, max_tokens=limit,
                     normalization='unit_l2', bridge_normalization='none')
dest.write_text(json.dumps(lock, indent=2), encoding='utf-8')
print(dest)
