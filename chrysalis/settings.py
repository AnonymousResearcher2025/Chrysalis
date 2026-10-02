"""Resolve source-checkout or wheel-installed configuration resources."""
import json
from pathlib import Path
import sysconfig


def load(path):
    candidate = Path(path)
    if not candidate.exists() and not candidate.is_absolute() and candidate.parent == Path('configs'):
        candidate = Path(sysconfig.get_path('data')) / 'share/chrysalis/configs' / candidate.name
    return json.loads(candidate.read_text(encoding='utf-8'))
