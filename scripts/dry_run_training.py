"""Run a bounded training check against existing processed data (no downloads).

From the repository root:
    python scripts/dry_run_training.py --config configs/amazon_replay.yaml --model adaptive_fusion
Outputs are isolated under artifacts/dry_runs; metrics are smoke-test results.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from reviewring.cli import resolve_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/amazon_replay.yaml')
    parser.add_argument('--model', default='adaptive_fusion')
    parser.add_argument('--epochs', type=int, default=2)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    import torch
    torch.set_num_threads(2)
    config = resolve_config(args.config)
    config['training'].update(max_epochs=args.epochs, fusion_max_epochs=args.epochs,
                              patience=args.epochs, fusion_patience=args.epochs)
    config['paths']['artifacts_dir'] = str(Path('artifacts/dry_runs').resolve())
    if config['track'] == 'raw_review_replay':
        from reviewring.pipelines.track_b_train import train
    elif config['track'] == 'yelp_static':
        from reviewring.pipelines.track_a import train
    else:
        parser.error('unsupported track')
    summary = train(config, args.model)
    print(json.dumps({'run_id': summary['run_id'], 'metrics': summary['metrics']}, indent=2))


if __name__ == '__main__':
    main()
