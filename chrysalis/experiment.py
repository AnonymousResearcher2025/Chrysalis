"""Reconstructable experiment driver. Ground truth lives only in Oracle."""
import json
from pathlib import Path
import shutil
import time
import numpy as np
from . import _core
from . import datasets
from .baselines import noadapt_maps, global_adapter, full_reembed, DualIndex
from .budget import Budget, Scheduler
from .calibration import partition, sample_regions, regional, diagnostic, query_epoch, epoch_due
from .embeddings import Encoder, GloveEncoder, PAIRS
from .evaluation import Oracle, replay, graph_serve, tables
from .index import Index
from .workers import LocalWorker, RpcServer, RpcClient
from .backends import LocalQueue


def run(dataset, output, config, model_lock, pair='P1', device='cpu', cache='models', override_diagnostic=False, glove=None):
    for field in ('budget_capacity', 'dollars_per_hour', 'price_per_hour'):
        if field not in config or config[field] is None:
            raise ValueError('explicit ' + field + ' required in experiment config')
    import torch
    torch.set_num_threads(config.get('cpu_threads', 4))
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    import hashlib
    source_root = Path(__file__).resolve().parents[1]
    source_hashes = {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for folder in ('chrysalis', 'cpp', 'configs', 'scripts', 'tests')
                     for p in (source_root / folder).rglob('*') if p.suffix in ('.py', '.cpp', '.json')}
    (root / 'source-hashes.json').write_text(json.dumps(source_hashes, indent=2))
    corpus, pools, manifest = datasets.load(dataset)
    for name in ('residual', 'replay', 'offline', 'evaluation'):
        if not pools.get(name):
            raise ValueError('four disjoint nonempty query pools required')
    all_ids = [r['id'] for pool in pools.values() for r in pool]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError('query pools overlap')
    raw = [r['raw'] for r in corpus]
    (root / 'configuration.json').write_text(json.dumps(dict(config=config, models=model_lock, dataset=manifest,
                                                            pair=pair, device=device, diagnostic_override=override_diagnostic), indent=2))
    oldkey, newkey = PAIRS[pair]
    old_encoder = GloveEncoder(glove, raw) if oldkey == 'glove' else Encoder(oldkey, model_lock, device, cache=cache)
    if oldkey == 'glove':
        (root / 'glove-provenance.json').write_text(json.dumps(old_encoder.spec, indent=2))
    new_encoder = Encoder(newkey, model_lock, device, cache=cache)
    started = time.perf_counter()
    old = old_encoder.encode(raw, category='initial_old_index')
    region_ids, centers = partition(old, config['R'], config['seed'])
    np.save(root / 'centers.npy', centers)
    params = dict(M=config['M'], efConstruction=config['efConstruction'], alpha=config['alpha_prune'], seed=config['seed'])
    index = Index.create(root / 'base', old, region_ids, raw, 'old-' + oldkey,
                         segment_size=config['segment_size'], graph_parameters=params)
    index.store.transaction({'config': config, 'inputs': manifest})
    upgrade_started = time.perf_counter()
    budget = Budget(index.store, config['budget_capacity'], config['dollars_per_hour'], config['price_per_hour'])
    new_encoder.account = budget.account
    def encode_items(ids):
        return new_encoder.encode([index.raw(i) for i in ids], category='calibration')
    residual_q = new_encoder.encode([r['raw'] for r in pools['residual']], role='query', category='calibration_query')
    samples = sample_regions(region_ids, config['fit_per_region'], config['calibration_per_region'], config['offline_per_region'], config['seed'], config['R'])
    (root / 'splits.json').write_text(json.dumps(samples, indent=2))
    maps, paired = regional(old, region_ids, samples, encode_items, residual_q,
                            rank=config['rank'], ridge=config['ridge'], alpha=config['alpha'], seed=config['seed'])
    diagnosis = diagnostic(maps, config['tau'])
    (root / 'diagnostic.json').write_text(json.dumps(diagnosis, indent=2))
    print('regional calibration', diagnosis, flush=True)
    if diagnosis['recommend_full_reembed'] and not override_diagnostic:
        index.store.transaction({'diagnostic': diagnosis, 'fallback': 'FullReembed required; migration not started'})
        index.close()
        return dict(status='diagnostic_abort', diagnosis=diagnosis)
    index.configure(model_lock[newkey]['revision'], maps)
    index.retain(paired)
    index.rotate()
    replay_pool = pools['replay'][:config['m_q']]
    if len(replay_pool) < config['m_q']:
        raise ValueError('insufficient replay queries; never silently reduce m_q')
    replay_q = new_encoder.encode([r['raw'] for r in replay_pool], role='query', category='calibration_query')
    epoch = query_epoch(index, replay_q, [r['id'] for r in replay_pool], encode_items,
                        config['alpha_q'], config['ef'], change_threshold=config['epoch_change_threshold'])
    index.publish_retained()
    calibration_rotation_wall_seconds = time.perf_counter() - upgrade_started
    seed_fraction = sum(n['state'] == 'native' for n in index.store.nodes()) / len(raw)
    print('epoch', epoch['id'], 'epsilon_cert', epoch['epsilon_cert'], 'native_seed', seed_fraction, flush=True)
    index.store.snapshot(root / 'seed-snapshot')
    # Offline coverage embeddings are not retained/published and remain separate.
    coverage = []
    offline_q = new_encoder.encode([r['raw'] for r in pools['offline']], role='query', category='offline_validation')
    for r, split in enumerate(samples):
        ids = split['offline']
        if not ids:
            coverage.append(dict(region=r, support=0, coverage=None, status='no_offline_support'))
            continue
        exact = new_encoder.encode([raw[i] for i in ids], category='offline_validation')
        q = offline_q[np.random.default_rng(config['seed'] + r).integers(len(offline_q), size=len(ids))]
        estimate = old[ids] @ np.asarray(maps[r]['W']) + np.asarray(maps[r]['b'])
        errors = abs(np.linalg.norm(q - estimate, axis=1) - np.linalg.norm(q - exact, axis=1))
        coverage.append(dict(region=r, support=len(ids), coverage=float(np.mean(errors <= maps[r]['epsilon'])),
                             relative_width=maps[r]['epsilon'] / maps[r]['dbar'] if maps[r]['dbar'] else None,
                             alpha=config['alpha'], R=config['R']))
    (root / 'offline-coverage.json').write_text(json.dumps(coverage, indent=2))
    # Oracle is independently generated from raw corpus, saved outside index roots.
    oracle_vectors = new_encoder.encode(raw, category='evaluation_ground_truth')
    (root / 'oracle').mkdir()
    np.save(root / 'oracle' / 'vectors.npy', oracle_vectors)
    oracle = Oracle(oracle_vectors)
    eval_raw = [r['raw'] for r in pools['evaluation']]
    eval_ids = [r['id'] for r in pools['evaluation']]
    basework = index.store.get('work', [])
    economics = []
    summaries = {}
    # NoAdapt compares projection/truncation on OFFLINE validation queries, never eval.
    legacy = _core.Graph(**params)
    legacy.build(old.tolist(), [0] * len(old))
    choices = noadapt_maps(oracle_vectors.shape[1], old.shape[1], config['seed'])
    def score_adapter(fn):
        serve = graph_serve(legacy, config['k'], config['ef'], fn)
        return np.mean([oracle.assess(q, serve(q), config['k'])['recall_full'] for q in offline_q])
    chosen = max(choices, key=lambda name: (score_adapter(choices[name]), name))
    operational = sorted(paired)
    adapter, adapter_info = global_adapter(old[operational], np.asarray([paired[i] for i in operational]),
                                           rank=config['rank'], ridge=config['ridge'], seed=config['seed'],
                                           mlp_steps=config.get('mlp_steps', 200))
    (root / 'baseline-selection.json').write_text(json.dumps(dict(noadapt=chosen, global_adapter=adapter_info), indent=2))
    from .storage import sync_dir
    import os
    def sync_array(path, x):
        with Path(path).open('wb') as f:
            np.save(f, x, allow_pickle=False); f.flush(); os.fsync(f.fileno())
        sync_dir(Path(path).parent)
    def sync_json(path, value):
        with Path(path).open('w') as f:
            json.dump(value, f); f.flush(); os.fsync(f.fileno())
        sync_dir(Path(path).parent)
    native_graph, fresh = full_reembed(raw, new_encoder, params, persist=lambda x: sync_array(root / 'full-vectors.npy', x))
    sync_json(root / 'full-topology.json', native_graph.topology())
    dual = DualIndex(legacy, len(raw), params)
    (root / 'dual').mkdir()
    sync_array(root / 'dual' / 'old-vectors.npy', old)
    sync_json(root / 'dual' / 'old-topology.json', legacy.topology())
    dual_peak = [0]
    def persist_dual(x, start):
        sync_array(root / 'dual' / f'{start}.npy', x)
        dual_peak[0] = max(dual_peak[0], sum(p.stat().st_size for p in (root / 'dual').rglob('*') if p.is_file()))
    dual_graph = dual.backfill(raw, new_encoder, persist=persist_dual)
    sync_json(root / 'dual' / 'topology.json', dual_graph.topology())
    # Durable atomic cutover marker after all second-index state is persisted.
    with (root / 'dual' / 'alias.json').open('w') as f:
        json.dump(dict(active='new', count=len(raw)), f)
        f.flush(); os.fsync(f.fileno())
    sync_dir(root / 'dual')
    dual_peak[0] = max(dual_peak[0], sum(p.stat().st_size for p in (root / 'dual').rglob('*') if p.is_file()))
    final_fresh_bytes = (root / 'full-vectors.npy').stat().st_size + (root / 'full-topology.json').stat().st_size
    for category, footprint in [('fullreembed', final_fresh_bytes), ('dualindex', dual_peak[0])]:
        work = [w for w in index.store.get('work', []) if w['category'] == category]
        economics.append(dict(configuration=category, device=device, encoder_seconds=sum(w['seconds'] for w in work),
                              gpu_busy_hours=sum(w['gpu_busy_seconds'] for w in work) / 3600,
                              dollars=sum(w['dollars'] for w in work), peak_storage_bytes=footprint,
                              storage_basis='vector+graph files; Chrysalis RocksDB metadata counted separately'))
    graph_variants = dict(NoAdapt=graph_serve(legacy, config['k'], config['ef'], choices[chosen]),
                          GlobalAdapter=graph_serve(legacy, config['k'], config['ef'], adapter),
                          FullReembed=graph_serve(native_graph, config['k'], config['ef']),
                          DualIndex=graph_serve(dual_graph, config['k'], config['ef']))
    for label, serve in graph_variants.items():
        for seed in config['replay_seeds']:
            summaries[f'{label}:{seed}'] = replay(eval_raw, eval_ids, new_encoder, serve, oracle,
                                                   root / f'queries-{label}-{seed}.jsonl', k=config['k'],
                                                   length=config['replay_length'], seed=seed, label=label)
    for seed in config['replay_seeds']:
        for mode, label in [('none', 'SeedNoResolution'), ('async', 'SeedAsync'), ('sync', 'SeedSync')]:
            path = root / f'{label}-{seed}'
            shutil.copytree(root / 'seed-snapshot', path)
            serving = Index(path)
            price = Budget(serving.store, config['budget_capacity'], config['dollars_per_hour'], config['price_per_hour'])
            new_encoder.account = price.account
            queue = LocalQueue(serving.store)
            worker = LocalWorker(serving, new_encoder, queue, price)
            server = RpcServer('127.0.0.1:0', encoder=new_encoder)
            client = RpcClient(f'127.0.0.1:{server.port}')
            variant_started = time.perf_counter()
            from collections import deque
            bound_window_size = config.get('mean_bound_window', config['replay_length'])
            bound_window = deque(maxlen=bound_window_size)
            target = {}
            def observe(row):
                if row.get('bound') is None:
                    return
                bound_window.append(row['bound'])
                if not target and len(bound_window) == bound_window_size and np.mean(bound_window) >= .95:
                    work_now = serving.store.get('work', [])
                    migration_now = [w for w in work_now if w['category'] not in ('query', 'offline_validation', 'evaluation_ground_truth')]
                    target.update(time_to_mean_bound_095_seconds=calibration_rotation_wall_seconds + time.perf_counter() - variant_started,
                                  encoder_seconds_to_mean_bound_095=sum(w['seconds'] for w in migration_now),
                                  gpu_busy_hours_to_mean_bound_095=sum(w['gpu_busy_seconds'] for w in migration_now) / 3600,
                                  first_target_label=row['label'], mean_bound_window=bound_window_size,
                                  target_metric='sliding served-query mean; calibration elapsed + per-replica elapsed')
            def serve(q):
                return serving.search(q, k=config['k'], ef=config['ef'], rho=config['rho'], mode=mode,
                                      enqueue=queue.send, resolver=lambda i: client.encode([serving.raw(i)], model_lock[newkey]['revision'])[0])
            summaries[f'{label}:{seed}'] = replay(eval_raw, eval_ids, new_encoder, serve, oracle,
                                                  root / f'queries-{label}-{seed}.jsonl', k=config['k'],
                                                  length=config['replay_length'], seed=seed, label=label,
                                                  after_query=worker.drain if mode == 'async' else None,
                                                  observer=observe if mode == 'async' else None)
            if mode == 'async':
                scheduler = Scheduler(serving, price, queue, estimated_seconds_per_item=1)
                # Scheduled f=25% is quota measured separately from seed/query work.
                scheduled_base = sum(n['origin'] == 'scheduled' for n in serving.store.nodes())
                remaining = sum(n['state'] != 'native' for n in serving.store.nodes())
                denominator = len(raw) if config['fraction_denominator'] == 'whole' else len(raw) * (1 - seed_fraction)
                quota = min(remaining, int(.25 * denominator))
                scheduler.schedule(batch_size=max(1, quota), max_items=quota)
                worker.drain()
                summaries[f'Scheduled25:{seed}'] = replay(eval_raw, eval_ids, new_encoder, serve, oracle,
                                                          root / f'queries-Scheduled25-{seed}.jsonl', k=config['k'],
                                                          length=config['replay_length'], seed=seed, label='Scheduled25', after_query=worker.drain, observer=observe)
                while any(n['state'] != 'native' for n in serving.store.nodes()):
                    scheduled = scheduler.schedule()
                    processed = worker.drain()
                    if not scheduled and not processed:
                        raise RuntimeError('convergence budget exhausted or active claims pending; resume using migrate command')
                calls_before = sum(w['count'] for w in serving.store.get('work', []))
                summaries[f'NativeNoRepair:{seed}'] = replay(eval_raw, eval_ids, new_encoder,
                    lambda q: serving.search(q, k=config['k'], ef=config['ef'], mode='none'), oracle,
                    root / f'queries-NativeNoRepair-{seed}.jsonl', k=config['k'], length=config['replay_length'], seed=seed, label='NativeNoRepair')
                edges = serving.audit(all_nodes=True)
                # Repair accepts only stored vectors and never receives an encoder.
                serving.retire()
                summaries[f'NativeRepair:{seed}'] = replay(eval_raw, eval_ids, new_encoder,
                    lambda q: serving.search(q, k=config['k'], ef=config['ef'], mode='none'), oracle,
                    root / f'queries-NativeRepair-{seed}.jsonl', k=config['k'], length=config['replay_length'], seed=seed, label='NativeRepair')
                work = serving.store.get('work', [])
                migration = [w for w in work if w['category'] not in ('evaluation_ground_truth', 'offline_validation', 'query', 'initial_old_index', 'fullreembed', 'dualindex')]
                economics.append(dict(configuration='Chrysalis', seed=seed, device=device,
                                      calibration_rotation_wall_seconds=calibration_rotation_wall_seconds,
                                      **target,
                                      gpu_busy_hours=sum(w['gpu_busy_seconds'] for w in migration) / 3600,
                                      encoder_seconds=sum(w['seconds'] for w in migration), dollars=sum(w['dollars'] for w in migration),
                                      query_encoder_seconds=sum(w['seconds'] for w in work if w['category'] == 'query'),
                                      query_dollars=sum(w['dollars'] for w in work if w['category'] == 'query'),
                                      peak_storage_bytes=serving.store.peak_bytes, final_storage_bytes=serving.store.measure(),
                                      seed_fraction=seed_fraction, native_fraction=1., rewritten_endpoints=edges,
                                      recalibration_due=epoch_due(serving), scheduled_fraction=sum(n['origin'] == 'scheduled' for n in serving.store.nodes()) / denominator,
                                      query_resolved=sum(n['origin'] == 'query' for n in serving.store.nodes())))
            client.close()
            server.close()
            serving.close()
    index.close()
    economics.append(dict(configuration='run', wall_seconds=time.perf_counter() - started, device=device,
                          historical_results_reproduced=False, diagnostic_override=override_diagnostic))
    (root / 'economics.json').write_text(json.dumps(economics, indent=2))
    (root / 'summaries.json').write_text(json.dumps(summaries, indent=2))
    result = tables(root)
    print(json.dumps(result, indent=2), flush=True)
    return result
