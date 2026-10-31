# Chrysalis

Chrysalis migrates a vector index from one embedding model to the next in place, while it keeps serving. A short calibration fits a linear bridge per region of the corpus and bounds its distance error with split-conformal prediction. Search treats each bridged distance as an interval, re-embeds the candidates whose intervals straddle the top-k boundary first, and returns every answer with a lower bound on its recall over the examined candidates. Routing reads a view that is frozen within each epoch and recalibrated between epochs. A budgeted scheduler re-embeds the remaining items on spot GPUs, and an edge auditor repairs graph links that the new geometry invalidates. Every state change goes through a write-ahead manifest, so a migration resumes from its exact frontier after a crash.

## Repository layout

| Path | Contents |
| --- | --- |
| `cpp/core.cpp` | C++ core: HNSW-style layered graph with DiskANN's pruning rule, immutable epoch views with lock-free switching, versioned overlays, interval and certificate kernels, interval edge auditor |
| `chrysalis/index.py` | Migration state machine (legacy, bridged, resolving, native): calibration, rotation, interval search with boundary resolution, epochs, writes, chained upgrades, recovery |
| `chrysalis/bridges.py`, `conformal.py` | Coarse regions, regional bridges (Procrustes or rank-limited ridge affine with cross-validated penalty), conformal radii, certificate scale |
| `chrysalis/scheduler.py` | Background re-embedding by heat x radius x in-degree, dollar token bucket |
| `chrysalis/storage.py` | Manifest, native-vector logs, segments, staged edges, adjacency, object storage |
| `chrysalis/workers.py` | GPU workers: region queues, direct RPC for sync resolution, request batching |
| `chrysalis/encoders.py` | Pinned encoder checkpoints, GPU-time ledger |
| `chrysalis/baselines.py` | Global, FullReembed, DualIndex graph builds, B1 rank merge, B2 and B2+ (HMoE), B3 (label-free FastFill) |
| `chrysalis/data.py` | Corpus and query preparation for P1 to P4 |
| `chrysalis/evaluation.py`, `experiments.py` | Ground truth, per-request metrics, and every experiment of Section 4 |
| `configs/paper.json` | All parameters of Section 4.1 |
| `configs/models.json`, `datasets.json` | Model checkpoints with pinned revisions, model pairs, data sources |
| `tests/` | Unit and end-to-end tests on synthetic data |

## Installation

Linux, Python 3.10 to 3.12, a C++17 compiler, and NVIDIA GPUs with CUDA 12 drivers for the encoders.

```sh
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e ".[gpu,baselines,test]"
pip install --no-deps "embedding-translation @ git+https://github.com/DBgroup-Edinburgh/embedding-translation@5e62baf1544264847ce3d8feeabd1ded6499b064"
pytest
```

The last `pip` line installs the reference implementation of HMoE (B2, B2+, and the confidence baseline of Table 6) without its training-pipeline dependencies; the `baselines` extra provides the packages it needs at run time. On a machine without a GPU, use the `cpu` extra instead of `gpu`.

## Data

```sh
chrysalis prepare wiki1m     # 1M passages from DPR's 21M Wikipedia passages (P1, P4)
chrysalis prepare nq         # 10K Natural Questions questions (DPR dev and test splits)
chrysalis prepare msmarco    # 8,841,823 MS MARCO passages and 6,980 dev queries (P2)
chrysalis prepare trec-dl    # TREC Deep Learning 2019 and 2020 passage queries (P2)
chrysalis prepare laion1m --laion-metadata /path/to/laion400m-meta   # 1M images and 10K held-out captions (P3)
chrysalis prepare glove      # GloVe 6B.300d (P4)
```

Sources, sample sizes and seeds are in `configs/datasets.json`. LAION-400M metadata (parquet files with `URL` and `TEXT` columns) must be obtained separately. Each prepared directory contains a `manifest.json` with its source, counts and SHA-256.

## Experiments

Every command writes per-request logs (JSON lines) and a summary under `results/<pair>/<experiment>/`. Encodings of the corpus and queries, ground truth and graphs are cached under `work/cache/<pair>/`. `--devices` lists the GPUs; the first one serves synchronous resolution. Parameters come from `configs/paper.json`; `--set key=value` overrides one.

| Result | Command |
| --- | --- |
| Table 3, quality during migration | `chrysalis run quality --pair P1` (also `P2`, `P3`) |
| Table 4, matched budget | `chrysalis run budget --pair P1` (also `P2`, `P3`) |
| Figure 2 | `chrysalis run figure2 --pair P3` (after `budget`) |
| Table 5, coverage and region sweep | `chrysalis run coverage --pair P2` |
| Table 6, certificate reliability, early recalibration | `chrysalis run reliability --pair P2 --workloads zipf,uniform,cold,trec --recalibrate`, `chrysalis run reliability --pair P3 --workloads zipf,uniform,cold --recalibrate` |
| Table 7, certificate ablation | `chrysalis run ablation --pair P2 --workloads zipf`, `chrysalis run ablation --pair P3 --workloads cold` |
| Table 8, cost and storage | `chrysalis run cost --pair P1 --devices cuda:0,cuda:1,cuda:2,cuda:3` (also `P2`, `P3`) |
| Table 9, scheduler ablation | `chrysalis run scheduler --pair P1` (also `P2`, `P3`) |
| Table 10, serving cost | `chrysalis run latency --pair P2 --devices cuda:0,cuda:1,cuda:2,cuda:3` |
| Table 11, fault injection | `chrysalis run faults --pair P2 --devices cuda:0 --spot cuda:1,cuda:2,cuda:3` |
| Section 4.6, cross-family fallback | `chrysalis run fallback --pair P4 --glove data/glove/glove.6B.zip` |
| Section 4.6, rare entities | `chrysalis run rare --pair P1` |
| Section 4.6, structure repair | `chrysalis run repair --pair P2` (also `P3`) |

Each run replays the evaluation queries as one seeded Zipf stream; heat comes only from the warm-up prefix, metrics only from later requests, and calibration queries are distinct queries sampled from the recent requests, excluding those that calibrated the previous view. Results average the replay seeds and calibration seeds of `configs/paper.json`. Ground truth is exact kNN in the new space by brute force with Faiss on GPU.

## Deployment

The index host serves search over RPC, hosts the region queues, and runs the scheduler, the edge auditor and epoch closing. GPU workers pull region batches from the host; the on-demand worker also answers direct encode calls for synchronous resolution.

```sh
chrysalis worker --pair P2 --devices cuda:0 --host HOST:7100 --sync --address 0.0.0.0:7101   # on-demand worker
chrysalis worker --pair P2 --devices cuda:0 --host HOST:7100                                 # each spot worker
chrysalis host   --pair P2 --index /data/index --on-demand WORKER:7101 --snapshots s3://bucket/prefix
```

`chrysalis host --recover` restarts from the manifest. The testbed of Section 4.1 is one r6i.4xlarge index host, four g5.2xlarge workers (one on-demand, three spot), a c6i.2xlarge client, gp3 EBS volumes and S3 for manifest snapshots, all in us-east-1.

## License

MIT, see `LICENSE`.
