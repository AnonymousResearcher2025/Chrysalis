# Chrysalis

[Repository](https://github.com/AnonymousResearcher2025/Chrysalis) · [MIT license](LICENSE)

C++17/Python implementation of certified in-place vector-index migration across embedding model versions. Includes regional calibration, interval search, embedding workers, durable migration, budgeted scheduling, graph repair, baselines, and evaluation tools.


## Build and test

Requires Python 3.12.

**Linux:** requires a C++17 compiler.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/build.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

**Windows PowerShell:** uses the pinned Zig compiler.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-windows-build.txt
$env:ZIG_GLOBAL_CACHE_DIR = "$PWD/build/zig-global"
$env:ZIG_LOCAL_CACHE_DIR = "$PWD/build/zig-local"
.venv/Scripts/python.exe scripts/build.py
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
.venv/Scripts/python.exe -m pytest -q
```

## Implementation

- `cpp/`: native graph index and interval kernel.
- `chrysalis/`: migration, calibration, persistence, workers, scheduling, repair, and evaluation.
- `configs/`: model and experiment settings.
- `scripts/`: build, dataset preparation, and experiment utilities.
