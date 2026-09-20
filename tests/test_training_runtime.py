"""Training regression tests, including complete tiny Track B dry runs."""
import json

import numpy as np
import pandas as pd
import pytest
import torch

from reviewring.pipelines.track_b_train import FUSION_MODELS, VALID_MODELS, train
from reviewring.training.calibrate import PlattCalibrator
from reviewring.training.train import bce_with_logits_masked, pos_weight


def test_unknown_and_empty_loss_backward():
    logits = torch.tensor([1., float('nan')], requires_grad=True)
    labels = torch.tensor([1., -1.])
    loss = bce_with_logits_masked(logits, labels, torch.ones(2, dtype=torch.bool))
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad[1] == 0
    logits.grad = None
    loss = bce_with_logits_masked(logits, labels, torch.zeros(2, dtype=torch.bool))
    loss.backward()
    assert loss.item() == 0
    assert torch.equal(logits.grad, torch.zeros(2))
    assert pos_weight(np.ones(3)) == 1


def test_calibration_missing_scores_and_empty_predictions():
    cal = PlattCalibrator().fit(np.array([-2., -1., 1., 2., np.nan]), np.array([0, 0, 1, 1, 1]))
    assert cal.fitted
    p = cal.predict_proba(np.array([-1., np.nan, 1.]))
    assert np.isnan(p[1]) and np.isfinite(p[[0, 2]]).all()
    assert cal.predict_proba(np.array([])).shape == (0,)
    cal.fit(np.array([np.nan]), np.array([1]))
    assert not cal.fitted


@pytest.mark.parametrize('model', sorted(VALID_MODELS))
def test_track_b_training_dry_run(tmp_path, model):
    # One unavailable review in every partition; both classes remain available.
    n = 32
    rng = np.random.default_rng(12)
    ids = [f'r{i}' for i in range(n)]
    missing = np.arange(n) % 8 == 0
    pd.DataFrame({'review_id': ids, 'text_missing': missing,
                  'timestamp': pd.date_range('2024-01-01', periods=n, tz='UTC')}).to_parquet(tmp_path / 'reviews.parquet')
    pd.DataFrame({'review_id': ids, 'label': np.arange(n) % 2}).to_parquet(tmp_path / 'labels.parquet')
    pd.DataFrame({'node_idx': np.arange(n), 'node_key': ids,
                  'role': np.repeat(['train', 'validation', 'calibration', 'test'], 8)}).to_parquet(tmp_path / 'splits.parquet')
    pd.DataFrame({'campaign_id': []}).to_parquet(tmp_path / 'campaigns.parquet')
    np.save(tmp_path / 'embeddings.npy', rng.normal(size=(n, 6)).astype('float32'))
    np.savez(tmp_path / 'features.npz', history=rng.normal(size=(n, 9)).astype('float32'),
             history_missing=np.repeat(missing[:, None], 9, axis=1))
    dst = np.flatnonzero(~missing)
    np.savez(tmp_path / 'edges.npz', edge_index=np.array([dst - 1, dst]), relation=np.zeros(len(dst), dtype='int64'))
    config = {'seed': 42, 'paths': {'processed_dir': str(tmp_path), 'artifacts_dir': str(tmp_path / 'runs')},
              'model': {'hidden_dim': 8, 'layers': 2, 'dropout': .1},
              'training': {'max_epochs': 2, 'patience': 1, 'batch_size': 3, 'pos_weight_cap': 2}}
    summary = train(config, model)
    run = tmp_path / 'runs' / summary['run_id']
    history = json.loads((run / 'train_history.json').read_text())
    for epochs in history.values():
        assert 1 <= len(epochs) <= 2
        assert all(np.isfinite(e['train_loss']) for e in epochs)
    preds = pd.read_parquet(run / 'test_predictions.parquet')
    assert summary['calibration']['fitted']
    assert np.isfinite(preds['score'].iloc[1:]).all()
    if model in FUSION_MODELS:
        assert summary['insufficient_data_rows'] == 4
        assert summary['metrics']['test_n_excluded'] == 1
        assert np.isnan(preds['score'].iloc[0])


def test_track_a_sampled_training_and_invalid_fanout(tmp_path):
    from reviewring.pipelines.track_a import _train_graph
    rng = np.random.default_rng(5)
    x = rng.normal(size=(32, 6)).astype('float32')
    y = np.arange(32) % 2
    edges = {'r': np.array([np.arange(31), np.arange(1, 32)])}
    cfg = {'model': {'layers': 2, 'hidden_dim': 8, 'dropout': .1},
           'training': {'max_epochs': 2, 'patience': 1, 'batch_size': 4, 'fanout_per_relation': [2, 2]}}
    _, _, val, test = _train_graph(cfg, x, edges, np.arange(16), np.arange(16, 24),
                                  np.arange(24, 32), y, 42, tmp_path)
    assert np.isfinite(val).all() and np.isfinite(test).all()
    assert (tmp_path / 'graph_model.pt').exists()
    cfg['training']['fanout_per_relation'] = [2]
    with pytest.raises(ValueError, match='one positive fanout'):
        _train_graph(cfg, x, edges, np.arange(16), np.arange(16, 24),
                     np.arange(24, 32), y, 42, tmp_path)


def test_cli_loads_shared_training_defaults():
    from reviewring.cli import resolve_config
    config = resolve_config('configs/amazon_replay.yaml')
    assert config['training']['pos_weight_cap'] == 100
    assert config['training']['fanout_per_relation'] == [10, 5]
    assert config['training']['max_epochs'] == 80


def test_track_a_provenance_without_raw_download(tmp_path):
    from reviewring.pipelines.track_a import _source_provenance
    raw = tmp_path / 'missing.mat'
    config = {'paths': {'processed_dir': str(tmp_path), 'yelp_mat': str(raw)}}
    (tmp_path / 'manifest.json').write_text(json.dumps({'source': {'sha256': 'recorded-hash'}}))
    assert _source_provenance(config) == {'data_sha256': 'recorded-hash', 'data_hash_source': 'prepare_manifest'}
    raw.write_bytes(b'raw-data')
    assert _source_provenance(config)['data_hash_source'] == 'raw_file'
