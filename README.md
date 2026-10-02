# Chrysalis

[Repository](https://github.com/AnonymousResearcher2025/Chrysalis) · [MIT license](LICENSE)

A working C++17/Python reconstruction of the supplied short paper. It performs
regional calibration, streaming vector migration, layered ANN traversal, interval
resolution, frozen-snapshot certificates, budgeted convergence and local graph
repair. RocksDB owns the manifest; local queues and real AWS/S3/SQS/gRPC adapters
share durable lease/fencing semantics. This is newly authored source, not the lost
prototype. Historical results are not claimed as reproduced.

The release has **33 passing tests**, a built and independently installed native
wheel, and recorded real MiniLM-to-MPNet migration evidence. Build, recovery,
worker and dataset checks are in [release readiness](docs/RELEASE_READINESS.md).
The authors' earlier experiment results remain the results reported in the paper;
the current release's validation is recorded separately. No rerun of the paper's
benchmark campaign is required to use or publish this implementation.

Read [artifact availability](docs/ARTIFACT.md), [specification](docs/SPECIFICATION.md), [traceability](docs/TRACEABILITY.md),
[architecture](docs/ARCHITECTURE.md), [provenance](docs/PROVENANCE.md), and
[executed report](docs/FINAL_REPORT.md). `configs/provisional.json` provides paper
defaults and explicit implementation choices where settings were not specified.

## Clean checkout

Use a local filesystem. On this host, Box's on-demand filesystem produced missing
file/directory errors in Python and the compiler; validation used local scratch.
The public archive includes `RELEASE-SHA256.txt`. Verify all packaged bytes
with `python scripts/verify_artifact.py /path/to/Chrysalis-GitHub-source.zip`.
See [anonymization](docs/ANONYMIZATION.md) for the public redaction record. The full
companion archive is private and must not be uploaded during anonymous review.

Linux with a C++17 compiler and Python 3.12:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/build.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python -m chrysalis.cli --help
```

Windows PowerShell with Python 3.12 (portable Zig compiler; no Visual Studio needed):

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-windows-build.txt
$env:ZIG_GLOBAL_CACHE_DIR = "$PWD/build/zig-global"
$env:ZIG_LOCAL_CACHE_DIR = "$PWD/build/zig-local"
.venv/Scripts/python.exe scripts/build.py
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
.venv/Scripts/python.exe -m pytest -q
```

CMake is an alternate build (`find_package` discovers pybind11 through Python).
`pip install . --no-build-isolation` also builds the extension via setuptools;
it uses Zig when available on Windows and the platform compiler elsewhere. The
wheel includes its default configurations and resolves them outside the checkout.
The executed Windows build used portable Zig; hosted Linux/Windows CI was
configured but not executed in this environment.

## Real-input smoke

The checked-in model lock contains immutable reconstruction revisions for P1-P4.
Do not silently repin it. `scripts/pin_models.py` creates a new lock only if its
destination does not already exist. The following downloads real AG News raw
texts and the actual MiniLM/MPNet checkpoints, then runs the full workflow:

```sh
python -m pip install -r requirements-datasets.txt
python scripts/smoke_data.py data/smoke
python scripts/prefetch.py minilm mpnet
python -m chrysalis.cli experiment --dataset data/smoke --output runs/smoke \
  --config configs/smoke.json --pair P1 --device cpu
python -m chrysalis.cli tables runs/smoke
```

The GitHub source bundle includes raw query records and measured tables under
`evidence/smoke`. Regenerate them offline with
`python -m chrysalis.cli tables evidence/smoke`. The separate full evidence archive
also includes converged indexes in `runs/smoke-final`. Regenerate its tables:
`python -m chrysalis.cli tables runs/smoke-final`. To repeat the separate current
retired-native measurement with the cached pinned successor encoder, run
`python scripts/evaluate_retired.py runs/smoke-final data/smoke`.

The downloader pins dataset commit eb185aade064a813bc0b7f42de02595523103ca4 and
records source-file hashes. Repeating preparation with an existing matching
manifest verifies it rather than silently replacing it. Downloaded model weights,
large raw corpora, runtime environments and compiler outputs are excluded from
the GitHub source tree.

For offline cached execution set `HF_HUB_OFFLINE=1`; put `HF_HOME` under a writable
local cache. In PowerShell, put a command on one line instead of using shell `\`
continuation. The smoke uses R1, M4, efC32, ef4, 160 raw texts and 20-query replay
windows. Skinny beams keep an unresolved remainder in a tiny corpus; their ANN
recall is deliberately not representative of M32/ef96. All five Zipf replay seeds
are used. Oracle vectors are independently generated and never supplied to the
serving/calibration/worker APIs. Model inference is CPU float32 or GPU FP16, with
native vectors normalized and bridge outputs left unnormalized.

## Explicit dataset preparation and phases

Input corpus/query JSONL rows are `{"id":"stable-id","raw":"raw text or immutable image path"}`.
Image corpus rows additionally contain `image_sha256`. The exact Wikipedia release,
NQ split, passage construction, LAION release/caption selection and rare entities
are unknown in the supplied paper. Supply those explicitly; the tool will record
hashes and selection procedures rather than invent an original configuration.
`datasets.import_tsv` imports MARCO collection/dev files and already constructed
Wiki passage TSV. `scripts/import_nq.py` imports NQ JSONL.gz query identities.
`scripts/download_asset.py` requires a known SHA256. For LAION, prepare a selected
JSONL of locally retained image bytes and caption-query rows, recording failed URLs
and the actual selection in the manifest. No unspecified corpus release is fetched
by default.

```sh
python -m chrysalis.cli prepare --corpus corpus.jsonl --queries queries.jsonl \
  --output data/my-input --dataset-id YOUR_DATASET --release IMMUTABLE_RELEASE \
  --selection 'Exact construction, split, selection and failure handling' \
  --counts '{"residual":512,"replay":512,"offline":512,"evaluation":1000}'
python -m chrysalis.cli embed --dataset data/my-input --model minilm --output data/old.npy
python -m chrysalis.cli build --dataset data/my-input --vectors data/old.npy \
  --index runs/index --version predecessor-version --config configs/provisional.json
python -m chrysalis.cli calibrate --dataset data/my-input --index runs/index --model mpnet \
  --config configs/provisional.json --price-per-hour YOUR_PRICE --capacity YOUR_BUDGET \
  --refill-per-hour YOUR_REFILL
python -m chrysalis.cli migrate --index runs/index --model mpnet --all --repair \
  --price-per-hour YOUR_PRICE --capacity YOUR_BUDGET --refill-per-hour YOUR_REFILL
python -m chrysalis.cli status --index runs/index
python -m chrysalis.cli retire --index runs/index
```

Pause/resume/recover/repair/snapshot commands accept `--index`. `epoch` restarts a
frozen query replay, reusing retained embeddings after interruption. A diagnostic
abort recommends FullReembed; `--override-diagnostic` explicitly permits exploratory
migration, and is recorded. `experiment` requires explicit price/capacity/refill
fields in its configuration; there are no recovered AWS price assumptions. The
CPU smoke explicitly prices work at $0 and does not measure an AWS invoice.

To evaluate an independently produced new-space oracle:

```sh
python -m chrysalis.cli embed --dataset data/my-input --model mpnet --output data/oracle.npy
python -m chrysalis.cli evaluate --index runs/index --dataset data/my-input \
  --oracle data/oracle.npy --output runs/queries-explicit.jsonl --model mpnet --mode async
```

Keep oracle exports outside serving directories. `experiment` automatically builds
all baselines, selects projection/truncation on offline queries, selects the global
adapter on paired validation, creates actual second-index backfill/cutover, and
runs no-resolution/async/sync/scheduled/native/no-repair/repair variants.
Use `scripts/sweeps.py` for the region and alpha sweep, and
`scripts/rare_entities.py` with explicit versioned entity annotations. Experiment
Table 2-5 CSVs are new measurements. They never interpolate or hardcode the paper.

## Cloud protocol

Use your own AWS credentials, versioned S3 bucket and SQS standard queue. Supply
prices and mTLS certificates. The reconstruction does not create or bill AWS
instances automatically. The paper's r6i.4xlarge index host, four g5.2xlarge A10G
workers (one on-demand/three spot), c6i.2xlarge client and gp3 disks are recorded
testbed specifications; that deployment has not been run here.

```sh
python -m chrysalis.cloud upload --index runs/index --bucket BUCKET --prefix PREFIX --raw-version RAW_VERSION
python -m chrysalis.cli worker --listen 127.0.0.1:50051 --model mpnet --device cuda
python -m chrysalis.cloud host --index runs/index --bucket BUCKET --prefix PREFIX \
  --raw-version RAW_VERSION --sqs-url QUEUE_URL --address HOST:50052 \
  --encoder-address ENCODER:50051 --price-per-hour PRICE --capacity BUDGET \
  --refill-per-hour REFILL --ca ca.pem --cert host.pem --key host.key
python -m chrysalis.cloud worker --bucket BUCKET --prefix PREFIX --raw-version RAW_VERSION \
  --sqs-url QUEUE_URL --address HOST:50052 --model mpnet --device cuda \
  --ca ca.pem --cert worker.pem --key worker.key
```

The standalone CLI encoder example is loopback development. For remote production
encoders add `--ca ca.pem --cert encoder.pem --key encoder.key` to `cli worker`;
do not expose its unauthenticated loopback example publicly. Host `Query` encodes
raw queries through the successor endpoint and uses direct resolution or SQS.
`RpcClient.call('Query', {raw,k,ef,rho,mode})` returns certificates and metadata.
`chrysalis.cloud snapshot/restore` use checksum-committed S3 checkpoints.
Image raw uploads require `--images` so workers retrieve bytes, not local paths.

RocksDB is single-host authoritative. The service lock is deliberately coarse;
performance and peak-storage guarantees are measured limits, not assumed paper
results. Certificates cover examined candidates on a frozen calibration snapshot
under exchangeability. Evolving-index soundness and full-corpus recall are separate
measurements. A native transition or repair can invalidate routing applicability.
