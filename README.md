# Chrysalis

Chrysalis migrates a vector index from one embedding model to the next in place, while it keeps serving. A short calibration fits a linear bridge per region of the corpus and bounds its distance error with split-conformal prediction. Search treats each bridged distance as an interval, re-embeds the candidates whose intervals straddle the top-k boundary first, and returns every answer with a lower bound on its recall over the examined candidates. Routing reads a view that is frozen within each epoch and recalibrated between epochs. A budgeted scheduler re-embeds the remaining items on spot GPUs, and an edge auditor repairs graph links that the new geometry invalidates. Every state change goes through a write-ahead manifest, so a migration resumes from its exact frontier after a crash.


## Installation

Linux, Python 3.10 to 3.12, a C++17 compiler, and NVIDIA GPUs with CUDA 12 drivers for the encoders.

```sh
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e ".[gpu,baselines,test]"
pip install --no-deps "embedding-translation @ git+https://github.com/DBgroup-Edinburgh/embedding-translation@5e62baf1544264847ce3d8feeabd1ded6499b064"
pytest
```

The last `pip` line installs the reference implementation of HMoE (B2, B2+, and the confidence baseline of Table 6) without its training-pipeline dependencies; the `baselines` extra provides the packages it needs at run time.

## Data

```sh
chrysalis prepare wiki1m     # 1M passages from DPR's 21M Wikipedia passages (P1, P4)
chrysalis prepare nq         # 10K Natural Questions questions (DPR dev and test splits)
chrysalis prepare msmarco    # 8,841,823 MS MARCO passages and 6,980 dev queries (P2)
chrysalis prepare trec-dl    # TREC Deep Learning 2019 and 2020 passage queries (P2)
chrysalis prepare laion1m --laion-metadata /path/to/laion400m-meta   # 1M images and 10K held-out captions (P3)
chrysalis prepare glove      # GloVe 6B.300d (P4)
```

## License

MIT, see `LICENSE`.
