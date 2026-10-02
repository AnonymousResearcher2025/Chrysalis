import importlib.metadata as meta
import json
import platform
from pathlib import Path
import sys
names = ['numpy', 'scipy', 'scikit-learn', 'rocksdict', 'torch', 'sentence-transformers',
         'transformers', 'huggingface-hub', 'grpcio', 'boto3', 'Pillow', 'requests',
         'pybind11', 'pytest', 'ziglang', 'tokenizers', 'safetensors', 'botocore']
out = dict(python=sys.version, platform=platform.platform(),
           dependencies={n: meta.version(n) for n in names},
           reconstruction=True, historical_prototype=False)
dest = Path(sys.argv[1] if len(sys.argv) > 1 else 'docs/executed-environment.json')
dest.write_text(json.dumps(out, indent=2))
print(dest)
