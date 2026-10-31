import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from . import config, data, evaluation, experiments


def _cfg(args):
    overrides = {}
    for item in args.set or []:
        key, _, value = item.partition("=")
        overrides[key] = json.loads(value)
    return config.load(args.config, **overrides)


def _pair(args, cfg):
    return experiments.Pair(args.pair, args.data, args.work, cfg, devices=args.devices.split(","), glove=args.glove)


EXPERIMENTS = {
    "quality": lambda p, o, a: experiments.quality(p, o),
    "budget": lambda p, o, a: experiments.matched_budget(p, o),
    "figure2": lambda p, o, a: experiments.plot_budget(Path(a.out) / p.name / "budget"),
    "coverage": lambda p, o, a: experiments.coverage(p, o),
    "reliability": lambda p, o, a: experiments.reliability(p, o, a.workloads.split(","), recalibrate=a.recalibrate),
    "ablation": lambda p, o, a: experiments.ablation(p, o, a.workloads),
    "cost": lambda p, o, a: experiments.cost(p, o, qps=a.qps),
    "scheduler": lambda p, o, a: experiments.scheduler_ablation(p, o),
    "latency": lambda p, o, a: experiments.latency(p, o),
    "fallback": lambda p, o, a: experiments.fallback(p, o),
    "rare": lambda p, o, a: experiments.rare_entities(p, o),
    "repair": lambda p, o, a: experiments.structure_repair(p, o),
    "faults": lambda p, o, a: experiments.faults(p, o, a),
}


def host(args):
    """Index host: serves Search over RPC, hosts the region queues, runs the scheduler, the
    auditor and epoch closing, and snapshots the manifest after each epoch."""
    from .index import MigratingIndex
    from .scheduler import Scheduler, maintain
    from .storage import ObjectStore
    from .workers import RemotePool

    cfg = _cfg(args)
    pair = _pair(args, cfg)
    root = Path(args.index)
    pool = RemotePool(args.address, args.on_demand, cfg.encode_batch)
    store = ObjectStore(args.snapshots) if args.snapshots else None
    if args.recover and (root / "manifest.log").exists():
        ix = MigratingIndex.open(root, cfg, pool)
    else:
        MigratingIndex.create(root, cfg, pair.old_graph(), np.asarray(pair.old_vectors()))
        ix = MigratingIndex(root, cfg)
        ix.attach(pool)
        q_new, q_old, old = pair.queries("new"), pair.queries("old"), pair.old_graph()
        warm = evaluation.zipf_stream(len(q_new), cfg.stream_requests, cfg.zipf_exponent, args.seed)[: cfg.warmup_requests]
        ix.observe([("eval", int(r)) for r in warm], q_new[warm])
        for r in warm:
            ix.add_heat(old.search(np.asarray(q_old[r]), cfg.ef_search)[0])
        ix.heat_on = False
        ix.upgrade(pair.d_new, pair.model_id, seed=args.seed)
    sched = Scheduler(ix, "utility")
    (root / "ready").write_text(str(os.getpid()))
    epoch = ix.epoch
    while True:
        crash = root / "crash"
        if crash.exists():
            os.environ["CHRYSALIS_CRASH"] = crash.read_text().strip()
            crash.unlink()
        if ix.segment_legacy.any():
            ix.rotate([int(np.nonzero(ix.segment_legacy)[0][0])])
        if not ix.retired:
            sched.step()
            maintain(ix)
            if ix.all_native() and not len(pool.queues):
                ix.retire()
        if ix.epoch != epoch:
            epoch = ix.epoch
            ix.snapshot(store)
        time.sleep(0.01)


def worker(args):
    from . import encoders
    from .storage import RawStore
    from .workers import run_worker

    models = config.models()
    spec = models["pairs"][args.pair]
    enc = encoders.load(spec["new"], models, args.devices.split(",")[0], _cfg(args).encode_batch, glove_path=args.glove)
    run_worker(args.address, args.host, enc, RawStore(Path(args.data) / spec["corpus"]), sync=args.sync,
               batch=_cfg(args).encode_batch, log=args.log)


def main(argv=None):
    p = argparse.ArgumentParser(prog="chrysalis")
    sub = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--pair", default="P2", choices=["P1", "P2", "P3", "P4"])
    common.add_argument("--data", default="data")
    common.add_argument("--work", default="work")
    common.add_argument("--devices", default="cuda:0", help="comma-separated; the first serves sync resolution")
    common.add_argument("--glove", help="glove.6B.zip (P4)")
    common.add_argument("--config", default=str(config.ROOT / "configs" / "paper.json"))
    common.add_argument("--set", action="append", help="override a parameter, e.g. --set regions=64")

    pr = sub.add_parser("prepare", help="download and prepare a corpus or query set")
    pr.add_argument("name", choices=["wiki1m", "nq", "msmarco", "trec-dl", "laion1m", "glove"])
    pr.add_argument("--data", default="data")
    pr.add_argument("--cache", default="downloads")
    pr.add_argument("--laion-metadata", help="directory of LAION-400M metadata parquet files")

    ru = sub.add_parser("run", parents=[common], help="run one experiment of Section 4")
    ru.add_argument("experiment", choices=sorted(EXPERIMENTS))
    ru.add_argument("--out", default="results")
    ru.add_argument("--workloads", default="zipf,uniform,cold,trec")
    ru.add_argument("--recalibrate", action="store_true", help="also measure an early recalibration epoch on cold traffic")
    ru.add_argument("--qps", type=float, default=100.0)
    ru.add_argument("--spot", default="", help="faults: comma-separated devices of the spot workers")

    ho = sub.add_parser("host", parents=[common], help="index host process")
    ho.add_argument("--index", required=True)
    ho.add_argument("--address", default="0.0.0.0:7100")
    ho.add_argument("--on-demand", required=True, help="address of the on-demand (sync) worker")
    ho.add_argument("--snapshots", help="s3://bucket/prefix or a directory for manifest snapshots")
    ho.add_argument("--recover", action="store_true")
    ho.add_argument("--seed", type=int, default=1)

    wo = sub.add_parser("worker", parents=[common], help="GPU worker process")
    wo.add_argument("--host", required=True)
    wo.add_argument("--address", default="0.0.0.0:7101")
    wo.add_argument("--sync", action="store_true", help="also serve direct Encode calls (on-demand worker)")
    wo.add_argument("--log")

    args = p.parse_args(argv)
    if args.cmd == "prepare":
        data.prepare(args.name, args.data, args.cache, args.laion_metadata)
    elif args.cmd == "run":
        cfg = _cfg(args)
        out = Path(args.out) / args.pair / args.experiment
        print(json.dumps(EXPERIMENTS[args.experiment](_pair(args, cfg), out, args), indent=2))
    elif args.cmd == "host":
        host(args)
    else:
        worker(args)


if __name__ == "__main__":
    main()
