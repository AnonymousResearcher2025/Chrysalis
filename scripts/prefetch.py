import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chrysalis.embeddings import Encoder
lock = json.loads(Path('configs/models.lock.json').read_text())
for key in sys.argv[1:] or ['minilm', 'mpnet']:
    enc = Encoder(key, lock, cache='models')
    print(key, enc.encode(['A real sentence about vector index migration.']).shape, flush=True)
