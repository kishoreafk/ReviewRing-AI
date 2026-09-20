"""Experiment isolation and failure bookkeeping regression checks."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('experiment_runner', Path(__file__).resolve().parents[1] / 'scripts' / 'run_experiment.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_existing_prepared_data_is_copied_and_config_saved(tmp_path, monkeypatch):
    original = tmp_path / 'original'
    original.mkdir()
    (original / 'data.txt').write_text('original')
    cfg = {'paths': {'processed_dir': str(original), 'artifacts_dir': 'old'}, 'training': {'max_epochs': 80}}
    monkeypatch.setattr(runner, 'resolve_config', lambda path: cfg)
    experiment = tmp_path / 'new'
    result = runner.make_config('amazon', experiment, epochs=2, prepared_data=True)
    copied = Path(result['paths']['processed_dir']) / 'data.txt'
    copied.write_text('changed')
    assert (original / 'data.txt').read_text() == 'original'
    assert result['paths']['artifacts_dir'] == str(experiment / 'artifacts')
    assert result['training']['fusion_max_epochs'] == 2
    assert (experiment / 'configs' / 'amazon_replay.yaml').exists()


def test_failure_recorded_without_overwriting_existing_experiment(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'REPO', tmp_path)
    def fail(*args):
        raise RuntimeError('fixture data failure')
    monkeypatch.setattr(runner, 'make_config', fail)
    experiment = tmp_path / 'experiment'
    args = runner.parser().parse_args(['--track', 'amazon', '--smoke', '--experiment-dir', str(experiment)])
    with pytest.raises(RuntimeError, match='fixture data failure'):
        runner.run_experiment(args)
    state = json.loads((experiment / 'experiment.json').read_text())
    assert state['status'] == 'failed'
    assert state['seeds'] == [42]
    assert state['runs'] == []
    with pytest.raises(FileExistsError):
        runner.run_experiment(args)
    assert json.loads((experiment / 'experiment.json').read_text()) == state


def test_duplicate_seeds_rejected(tmp_path):
    args = runner.parser().parse_args(['--seeds', '42', '42', '--experiment-dir', str(tmp_path / 'out')])
    with pytest.raises(ValueError, match='distinct'):
        runner.run_experiment(args)
    assert not (tmp_path / 'out').exists()
