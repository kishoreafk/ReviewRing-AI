"""Run one isolated, reproducible experiment from the repository root.

The Colab and local shell entry points both call this runner. Use --smoke for a
bounded end-to-end check, including calibration, replay, rings and reports.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))

from reviewring.cli import resolve_config
from reviewring.utils.runtime import save_json


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--track', choices=['all', 'yelp', 'amazon'], default='all')
    p.add_argument('--seeds', nargs='+', type=int, default=[17, 42, 73])
    p.add_argument('--epochs', type=int, help='override expert and fusion epoch ceilings')
    p.add_argument('--smoke', action='store_true', help='seed 42, two epochs, graph + adaptive fusion')
    p.add_argument('--prepared-data', action='store_true', help='copy existing configured processed data instead of rebuilding it')
    p.add_argument('--experiment-dir', type=Path, help='new empty output directory; defaults to experiments/<UTC-id>')
    return p


def make_config(track, experiment, epochs=None, prepared_data=False):
    import yaml
    name = 'yelp_static' if track == 'yelp' else 'amazon_replay'
    config = resolve_config(str(REPO / 'configs' / f'{name}.yaml'))
    original = Path(config['paths']['processed_dir'])
    processed = experiment / 'processed' / track
    if prepared_data:
        if not original.is_dir():
            raise FileNotFoundError(f'No prepared data at {original}; omit --prepared-data to rebuild it')
        shutil.copytree(original, processed)
    config['paths']['processed_dir'] = str(processed)
    config['paths']['artifacts_dir'] = str(experiment / 'artifacts')
    if epochs is not None:
        config['training']['max_epochs'] = epochs
        config['training']['fusion_max_epochs'] = epochs
    config_path = experiment / 'configs' / f'{name}.yaml'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    return config


def run_experiment(args):
    import torch
    from reviewring.pipelines import report, track_a, track_b_data, track_b_eval, track_b_train

    if args.epochs is not None and args.epochs < 1:
        raise ValueError('--epochs must be positive')
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError('--seeds must be distinct')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    experiment = (args.experiment_dir or REPO / 'experiments' / f'{stamp}-{uuid4().hex[:8]}').resolve()
    experiment.mkdir(parents=True, exist_ok=False)
    # Limit small matrix workload overhead; respect an explicit user setting.
    torch.set_num_threads(int(os.environ.get('REVIEWRING_THREADS', '2')))
    seeds = [42] if args.smoke else args.seeds
    epochs = 2 if args.smoke else args.epochs
    state = {'status': 'running', 'smoke': args.smoke, 'seeds': seeds,
             'experiment_dir': str(experiment), 'runs': [], 'python': sys.version,
             'torch': str(torch.__version__), 'training_device': 'cpu'}
    rev = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True, capture_output=True)
    state['git_commit'] = rev.stdout.strip() if rev.returncode == 0 else None
    status = subprocess.run(['git', 'status', '--porcelain'], cwd=REPO, text=True, capture_output=True)
    state['working_tree_dirty'] = bool(status.stdout.strip()) if status.returncode == 0 else None
    manifest = experiment / 'experiment.json'
    save_json(manifest, state)
    save_json(REPO / 'latest_experiment.json', {'experiment_dir': str(experiment)})
    print(f'Experiment outputs: {experiment}', flush=True)

    def remember(summary):
        state['runs'].append({'run_id': summary['run_id'], 'model': summary['model'],
                              'track': summary['track'], 'seed': summary['seed']})
        save_json(manifest, state)

    try:
        if args.track in ('all', 'yelp'):
            config = make_config('yelp', experiment, epochs, args.prepared_data)
            if not args.prepared_data:
                track_a.prepare(config)
                track_a.split(config)
            for seed in seeds:
                for model in (['graph'] if args.smoke else ['prior', 'numeric', 'graph']):
                    summary = track_a.train(config, model, seed)
                    remember(summary)
                    report.track_a_report(config, summary['run_id'])
        if args.track in ('all', 'amazon'):
            config = make_config('amazon', experiment, epochs, args.prepared_data)
            if not args.prepared_data:
                for stage in ('prepare', 'split', 'simulate', 'features', 'graph'):
                    getattr(track_b_data, stage)(config)
            models = ['adaptive_fusion'] if args.smoke else [
                'text', 'time', 'graph', 'fixed_fusion', 'global_fusion',
                'adaptive_fusion', 'adaptive_fusion_noctx']
            main_seed = 42 if 42 in seeds else seeds[0]
            for seed in seeds:
                for model in models:
                    summary = track_b_train.train(config, model, seed)
                    remember(summary)
                    run_id = summary['run_id']
                    if model == 'adaptive_fusion' and seed == main_seed:
                        for stage in ('replay', 'rings', 'explain'):
                            getattr(track_b_eval, stage)(config, run_id)
                    track_b_eval.report(config, run_id)
        subprocess.run([sys.executable, str(REPO / 'scripts' / 'collect_results.py'),
                        '--artifacts-dir', str(experiment / 'artifacts'),
                        '--output', str(experiment / 'RESULTS.md')], check=True, cwd=REPO)
        state['status'] = 'complete'
    except BaseException as exc:
        state['status'] = 'failed'
        state['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        save_json(manifest, state)
    print(f'Completed. Results: {experiment / "RESULTS.md"}', flush=True)
    return experiment


def main():
    os.chdir(REPO)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    run_experiment(parser().parse_args())


if __name__ == '__main__':
    main()
