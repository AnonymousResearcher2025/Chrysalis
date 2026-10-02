import argparse
import json
from pathlib import Path
import time
import numpy as np


def config(path):
    from .settings import load
    return load(path)


def main():
    p = argparse.ArgumentParser(description='Chrysalis reconstruction: explicit inputs and measured outputs')
    sub = p.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare')
    for field in ('corpus', 'queries', 'output', 'dataset-id', 'release', 'selection'):
        prepare.add_argument('--' + field, required=True)
    prepare.add_argument('--seed', type=int, default=42)
    prepare.add_argument('--counts', default='{"residual":512,"replay":512,"offline":512,"evaluation":1000}')
    embed = sub.add_parser('embed')
    embed.add_argument('--dataset', required=True)
    embed.add_argument('--model', required=True)
    embed.add_argument('--output', required=True)
    build = sub.add_parser('build')
    for field in ('dataset', 'vectors', 'index', 'version'):
        build.add_argument('--' + field, required=True)
    build.add_argument('--config', default='configs/provisional.json')
    for cmd in ('calibrate', 'epoch', 'migrate'):
        s = sub.add_parser(cmd)
        s.add_argument('--index', required=True)
        s.add_argument('--dataset', required=cmd != 'migrate')
        s.add_argument('--model', default='mpnet')
        s.add_argument('--config', default='configs/provisional.json')
        s.add_argument('--price-per-hour', type=float, required=True)
        s.add_argument('--capacity', type=float, required=True)
        s.add_argument('--refill-per-hour', type=float, required=True)
        if cmd == 'calibrate':
            s.add_argument('--override-diagnostic', action='store_true', help='explicit exploratory run outside diagnostic acceptance')
        if cmd == 'migrate':
            s.add_argument('--all', action='store_true')
            s.add_argument('--repair', action='store_true')
    evaluate = sub.add_parser('evaluate')
    for field in ('index', 'dataset', 'oracle', 'output'):
        evaluate.add_argument('--' + field, required=True)
    evaluate.add_argument('--model', default='mpnet')
    evaluate.add_argument('--config', default='configs/provisional.json')
    evaluate.add_argument('--mode', choices=['none', 'sync', 'async'], default='async')
    experiment = sub.add_parser('experiment')
    experiment.add_argument('--dataset', required=True)
    experiment.add_argument('--output', required=True)
    experiment.add_argument('--config', default='configs/provisional.json')
    experiment.add_argument('--pair', choices=['P1', 'P2', 'P3', 'P4'], default='P1')
    experiment.add_argument('--override-diagnostic', action='store_true')
    experiment.add_argument('--glove')
    report = sub.add_parser('tables')
    report.add_argument('run_directory')
    for name in ('status', 'pause', 'resume', 'recover', 'repair', 'retire', 'snapshot'):
        s = sub.add_parser(name)
        s.add_argument('--index', required=True)
        if name == 'snapshot':
            s.add_argument('--output', required=True)
    worker = sub.add_parser('worker')
    worker.add_argument('--listen', default='127.0.0.1:50051')
    worker.add_argument('--model', default='mpnet')
    worker.add_argument('--ca')
    worker.add_argument('--cert')
    worker.add_argument('--key')
    for s in (embed, evaluate, experiment, worker, *(sub.choices[c] for c in ('calibrate', 'epoch', 'migrate'))):
        s.add_argument('--models', default='configs/models.lock.json')
        s.add_argument('--device', default='cpu')
        s.add_argument('--cache', default='models')
    args = p.parse_args()
    from . import datasets
    from .embeddings import Encoder
    if args.command == 'prepare':
        print(datasets.prepare(args.corpus, args.queries, args.output, dataset_id=args.dataset_id,
                               release=args.release, selection=args.selection, seed=args.seed, counts=json.loads(args.counts)))
        return
    if args.command == 'tables':
        from .evaluation import tables
        print(json.dumps(tables(args.run_directory), indent=2))
        return
    if args.command == 'experiment':
        from .experiment import run
        run(args.dataset, args.output, config(args.config), config(args.models), args.pair, args.device,
            args.cache, args.override_diagnostic, args.glove)
        return
    if args.command == 'embed':
        corpus, _, manifest = datasets.load(args.dataset)
        lock = config(args.models)
        encoder = Encoder(args.model, lock, args.device, cache=args.cache)
        x = encoder.encode([r['raw'] for r in corpus], category='explicit_embedding_export')
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
        np.save(path, x)
        Path(str(path) + '.json').write_text(json.dumps(dict(dataset=manifest, model=lock[args.model],
                                                         vector_sha256=datasets.sha256(path)), indent=2))
        return
    if args.command == 'worker':
        from .workers import RpcServer
        credentials = None
        if args.ca:
            import grpc
            credentials = grpc.ssl_server_credentials([(Path(args.key).read_bytes(), Path(args.cert).read_bytes())],
                                                       root_certificates=Path(args.ca).read_bytes(), require_client_auth=True)
        server = RpcServer(args.listen, encoder=Encoder(args.model, config(args.models), args.device, cache=args.cache), credentials=credentials)
        print('worker serving', args.listen, flush=True)
        try:
            server.server.wait_for_termination()
        finally:
            server.close()
        return
    from .index import Index
    from .budget import Budget, Scheduler
    if args.command == 'build':
        from .calibration import partition
        cfg = config(args.config)
        corpus, _, manifest = datasets.load(args.dataset)
        recorded = config(str(args.vectors) + '.json')
        if recorded['vector_sha256'] != datasets.sha256(args.vectors) or recorded['dataset'] != manifest:
            raise ValueError('input vector provenance/hash mismatch')
        x = np.load(args.vectors, allow_pickle=False)
        regions, centers = partition(x, cfg['R'], cfg['seed'])
        index = Index.create(args.index, x, regions, [r['raw'] for r in corpus], args.version,
                             segment_size=cfg['segment_size'], graph_parameters=dict(M=cfg['M'], efConstruction=cfg['efConstruction'],
                                                                                    alpha=cfg['alpha_prune'], seed=cfg['seed']))
        index.store.transaction({'inputs': manifest, 'centers': centers.tolist(), 'config': cfg, 'old_embedding_provenance': recorded})
        index.close()
        return
    index = Index(args.index)
    try:
        if args.command in ('calibrate', 'epoch', 'migrate', 'evaluate'):
            cfg = config(args.config)
            lock = config(args.models)
            if args.command == 'evaluate':
                encoder = Encoder(args.model, lock, args.device, cache=args.cache)
            else:
                budget = Budget(index.store, args.capacity, args.refill_per_hour, args.price_per_hour)
                encoder = Encoder(args.model, lock, args.device, cache=args.cache, account=budget.account)
            if args.command != 'calibrate' and index.store.get('meta')['new_version'] != lock[args.model]['revision']:
                raise ValueError('worker revision differs from index successor')
            from .backends import LocalQueue
            from .workers import LocalWorker
            queue = LocalQueue(index.store)
            worker = LocalWorker(index, encoder, queue, budget if args.command != 'evaluate' else None)
            def encode_ids(ids):
                return encoder.encode([index.raw(i) for i in ids], category='calibration')
            if args.command in ('calibrate', 'epoch'):
                from .calibration import sample_regions, regional, diagnostic, query_epoch
                corpus, pools, manifest = datasets.load(args.dataset)
                if manifest != index.store.get('inputs'):
                    raise ValueError('dataset differs from index construction input')
                if args.command == 'calibrate':
                    ns = index.store.nodes()
                    old = np.asarray([index.store.vector(n) for n in ns])
                    regions = np.asarray([n['region'] for n in ns])
                    samples = sample_regions(regions, cfg['fit_per_region'], cfg['calibration_per_region'], cfg['offline_per_region'], cfg['seed'], cfg['R'])
                    queries = encoder.encode([r['raw'] for r in pools['residual']], role='query', category='calibration_query')
                    maps, paired = regional(old, regions, samples, encode_ids, queries, rank=cfg['rank'], ridge=cfg['ridge'], alpha=cfg['alpha'], seed=cfg['seed'])
                    d = diagnostic(maps, cfg['tau'])
                    index.store.transaction({'samples': samples, 'diagnostic': d})
                    if d['recommend_full_reembed'] and not args.override_diagnostic:
                        print(json.dumps(d)); return
                    index.configure(lock[args.model]['revision'], maps)
                    index.retain(paired)
                    index.rotate()
                replay = pools['replay'][:cfg['m_q']]
                if len(replay) != cfg['m_q']:
                    raise ValueError('insufficient m_q query identities')
                query = encoder.encode([r['raw'] for r in replay], role='query', category='calibration_query')
                e = query_epoch(index, query, [r['id'] for r in replay], encode_ids, cfg['alpha_q'], cfg['ef'], change_threshold=cfg['epoch_change_threshold'])
                index.publish_retained()
                print(json.dumps(e))
            elif args.command == 'migrate':
                index.rotate()
                index.publish_retained()
                scheduler = Scheduler(index, budget, queue)
                while True:
                    jobs = scheduler.schedule()
                    done = worker.drain()
                    if args.repair:
                        index.audit()
                    if not args.all or all(n['state'] == 'native' for n in index.store.nodes()) or not (jobs or done):
                        break
                print(dict(native=sum(n['state'] == 'native' for n in index.store.nodes()), count=index.store.get('meta')['count'],
                           paused_or_budget_blocked=not (jobs or done)))
            else:
                from .evaluation import Oracle, replay
                corpus, pools, manifest = datasets.load(args.dataset)
                serving_inputs = index.store.get('inputs')
                if serving_inputs and manifest != serving_inputs:
                    raise ValueError('evaluation dataset mismatch')
                oracle = Oracle(np.load(args.oracle, allow_pickle=False))
                pool = pools['evaluation']
                def serve(q):
                    return index.search(q, k=cfg['k'], ef=cfg['ef'], rho=cfg['rho'], mode=args.mode,
                                        enqueue=queue.send, resolver=lambda i: encoder.encode([index.raw(i)], category='boundary')[0])
                print(replay([r['raw'] for r in pool], [r['id'] for r in pool], encoder, serve, oracle, args.output,
                             k=cfg['k'], length=cfg['replay_length'], seed=cfg['seed'], label='explicit-evaluation',
                             after_query=worker.drain if args.mode == 'async' else None))
        elif args.command == 'status':
            from collections import Counter
            print(json.dumps(dict(meta=index.store.get('meta'), states=dict(Counter(n['state'] for n in index.store.nodes())),
                                  origins=dict(Counter(n['origin'] for n in index.store.nodes())), budget=index.store.get('budget'),
                                  live_bytes=index.store.measure(), peak_bytes=index.store.peak_bytes), indent=2))
        elif args.command in ('pause', 'resume'):
            b = index.store.get('budget')
            if not b:
                raise RuntimeError('no initialized budget')
            b['paused'] = args.command == 'pause'
            index.store.transaction({'budget': b})
        elif args.command == 'recover':
            print('expired claims reset', index.store.recover())
            index.reload()
        elif args.command == 'repair':
            print('rewritten endpoints', index.audit(all_nodes=True))
        elif args.command == 'retire':
            index.retire()
        elif args.command == 'snapshot':
            print(index.store.snapshot(args.output))
    finally:
        index.close()


if __name__ == '__main__':
    main()
