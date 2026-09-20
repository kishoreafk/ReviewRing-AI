# Running ReviewRing AI on Google Colab

Open `colab/ReviewRing_AI_Colab.ipynb`, or run the shell entry point below from an
updated checkout of https://github.com/kishoreafk/ReviewRing-AI.git.
The trainers use CPU; MiniLM text embedding can use a GPU if one is available.
A GPU is optional. Training time depends on the CPU assigned by Colab.

## Use the updated source bundle before the fixes are pushed

Upload `ReviewRing_AI_Colab_source.zip` to `/content`, then run:

```python
import zipfile
with zipfile.ZipFile('/content/ReviewRing_AI_Colab_source.zip') as bundle:
    bundle.extractall('/content/ReviewRing-AI')
%cd /content/ReviewRing-AI
```

The bundle contains project source and the notebook, not datasets or credentials.
Use a fresh folder if you have edited files in an existing Colab checkout.
The notebook detects existing source and does not automatically pull over it.

## Full experiment

```python
import subprocess
subprocess.run(['bash', '-o', 'pipefail', '-c',
                'bash colab/colab_run.sh 2>&1 | tee /content/train.log'], check=True)
```

This installs dependencies, runs tests, downloads datasets, and trains both tracks.
Failures stop the command and are visible in the cell. Kaggle credentials are optional;
canonical dataset sources work without them. If needed, upload `kaggle.json` to
`/content` before running the shell script.

Each experiment saves to a new `experiments/<UTC-id>/` directory:

- `configs/`: resolved YAML settings for each track.
- `processed/`: that experiment's data and features.
- `artifacts/<run_id>/`: checkpoints, predictions, metrics and reports.
- `RESULTS.md`: only this experiment's comparisons.
- `experiment.json`: seed list, code revision, completion/failure and completed runs.

`latest_experiment.json` points to the newest experiment. Historical artifacts shipped
with the repository are never added to the new table or selected for evaluation.
All seven Amazon variants run on seeds 17, 42 and 73, including global fusion and the
no-context ablation. The fixed review budget stays unchanged; tune it on validation
in a separate experiment if desired.

## Quick runtime check and individual tracks

```python
# Setup + download + two-epoch check, including reports and replay:
subprocess.run(['bash', 'colab/colab_run.sh', '--smoke'], check=True)

# After setup, run one track or use an explicit epoch ceiling:
subprocess.run(['python', 'scripts/run_experiment.py', '--track', 'yelp', '--epochs', '250'], check=True)
# subprocess.run(['python', 'scripts/run_experiment.py', '--track', 'amazon'], check=True)
```

`--smoke` uses seed 42 and two epochs for Yelp graph and Amazon adaptive fusion.
It is a runtime check, not a model-quality result. `--epochs N` overrides expert and
fusion epoch limits; patience still comes from the configuration. Without an override,
the runner uses the track YAML settings. Optional `fusion_max_epochs` and
`fusion_patience` configure fusion separately. `--prepared-data` copies existing
`data/processed` inputs into the new experiment for checks without rebuilding features.

For individual CLI stages, pass the experiment's saved config:

```python
# Example, replacing EXPERIMENT and RUN with values printed by the runner:
# !python -m reviewring.cli --config EXPERIMENT/configs/amazon_replay.yaml report --run RUN
```

## Collect and download the current results

```python
import json, shutil
from pathlib import Path
from google.colab import files
experiment = Path(json.loads(Path('latest_experiment.json').read_text())['experiment_dir'])
subprocess.run(['python', 'scripts/collect_results.py',
                '--artifacts-dir', str(experiment / 'artifacts'),
                '--output', str(experiment / 'RESULTS.md')], check=True)
archive = shutil.make_archive('/content/' + experiment.name, 'zip',
                              root_dir=experiment.parent, base_dir=experiment.name)
files.download(archive)
```

Save the experiment folder or zip to Drive before the runtime ends. The zip includes
processed data so later reports use the same inputs. Failed runs retain their finished
artifacts and record the error in `experiment.json`; rerunning creates a fresh experiment.
For legacy mixed artifacts, the collector keeps only the latest run per model and seed;
use isolated experiments when comparing different settings or encoder versions.
