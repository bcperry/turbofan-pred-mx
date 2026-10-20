# Nasa-MX demo

A notebook-first learning project for predicting remaining useful life (RUL) from
NASA C-MAPSS simulated turbofan measurements, then serving a small regression model
through Azure Machine Learning in Azure Government.

The goal is to understand the complete path: acquire data, validate it, create labels,
avoid engine leakage, train and evaluate a model, register the artifact, deploy it,
and inspect a real authenticated HTTP request and prediction response. This is not
a production maintenance system or certified flight-safety tool.

## What works today

- Anonymous download and extraction of the public C-MAPSS archive.
- Raw data registration and preparation of versioned AML data assets.
- Local FD001 training with a mean-prediction baseline and a small gradient-boosted model.
- Separate Python 3.10/scikit-learn 1.5.2 retraining for the selected Azure environment.
- Model registration and managed online deployment through AML Studio.
- A live Entra-authenticated online request returning HTTP 200 and predictions matching
  the saved local examples. That smoke test succeeded on 2026-10-16.
- Local tests of the shared online/batch scoring script. A remote batch scoring job
  has not been validated here.

Local notebooks support learning and experimentation. Reproducible AML v2 training
and evaluation jobs remain a future milestone; local training plus an online deployment
does not establish a completed remote training pipeline.

## Setup

Install [uv](https://docs.astral.sh/uv/) and use VS Code with Python/Jupyter support.
Run from the repository root:

```powershell
uv sync --locked
```

Select the root `.venv` Python 3.12 kernel for local exploration, default local training,
and endpoint-calling notebooks. Dependencies are declared in [pyproject.toml](pyproject.toml)
and locked in [uv.lock](uv.lock). The configured package feed must be reachable.
Use `uv add` or `uv add --dev` when changing dependencies; restore with `uv sync --locked`.

For Azure Government, install Azure CLI and sign in when needed:

```powershell
az cloud set --name AzureUSGovernment
az login
```

Data-registration notebooks require an AML workspace configuration discoverable by
`MLClient.from_config`, and permissions to use the workspace and its storage. They were
developed for the AML Python SDK v2 notebook kernel. The online calling notebook needs
only its endpoint URL, deployment name, local example JSON, and a signed-in identity
with endpoint scoring permission. It does not need subscription or resource-group inputs.

## Notebook Guide

All notebooks live in [notebooks](notebooks). Run cells in order and check paths and
asset versions before running upload or registration cells. Cloud operations are explicit;
they can store data or start billable compute.

| Notebook | Where to run | What it does and why |
| --- | --- | --- |
| [download_cmapss.ipynb](notebooks/download_cmapss.ipynb) | AML notebook, or locally for the download | Downloads the public archive without AWS credentials, extracts nested files, and optionally registers the raw folder as `nasa-cmapss`. |
| [prepare_cmapss.ipynb](notebooks/prepare_cmapss.ipynb) | AML SDK v2 notebook kernel | Reads the registered raw asset through Blob storage, validates trajectories, names sensor channels, calculates RUL, and exports/registers train, validation, test, and full MLTable assets. |
| [train_local_cmapss.ipynb](notebooks/train_local_cmapss.ipynb) | Local root `.venv` by default | Reads local raw FD001 files, makes engine-disjoint splits, evaluates a baseline and regression pipeline, saves artifacts, and verifies save/reload prediction parity. |
| [call_online_endpoint.ipynb](notebooks/call_online_endpoint.ipynb) | Local root `.venv` | Sends local example records directly as JSON to the real-time endpoint, using an Entra token from Azure CLI. Displays the request with token redacted, HTTP status, and response. |
| [call_batch_endpoint.ipynb](notebooks/call_batch_endpoint.ipynb) | Local root `.venv` | Shows the alternative asynchronous API: sends a URI for an already uploaded CSV and displays the job-submission response. It does not upload local files or return predictions synchronously. |

### Data locations and preparation

The local training notebook expects raw files under `notebooks/data/cmapss/CMAPSSData`.
When running the download notebook locally, use `notebooks` as its working directory
to produce that layout. The download and preparation notebooks use paths relative to
their kernel's working directory; their files do not automatically transfer between
your machine and an AML compute instance.

The preparation notebook reads the registered raw asset rather than assuming that the
download notebook's local files are available. It stages outputs under
`data/prepared/FD001` relative to its working directory and registers:

- `cmapss-fd001-train`: selected training engines, features, and target.
- `cmapss-fd001-validation`: different engines, with the same feature schema.
- `cmapss-fd001-test`: the last observed cycle of each separate NASA test engine.
- `cmapss-fd001-full`: all observed train/test cycles, all original inputs, RUL,
  `engine_uid`, and `source_split`, for making your own engine-level partitions.

Use a new asset version when changing preparation. Use `engine_uid` to group the full
table: NASA train and test fleets reuse numeric engine IDs. Never randomly split rows
from the same engine across training and validation. Exclude IDs, split markers, and
the target from model inputs. Fit preprocessing only on training engines.

The ready-made prepared splits remove constant features. They are not interchangeable
with this saved model's input schema: the local model expects all 25 named inputs,
including constants, and removes constants inside its fitted pipeline.

### Modeling choices

We start with FD001, a single operating condition and HPC degradation, rather than
mixing the four subsets. Descriptive sensor names preserve documented compressor and
turbine locations; the preparation notebook records the source mapping and unit caveats.

The target is uncapped RUL in cycles. Training labels are each engine's final cycle
minus its current cycle. Truncated test engines use NASA's supplied final-observation
RUL. Seed 42 reserves 20 of the 100 training engines for validation. The 100 NASA test
engines are evaluated separately at their last observed observation.

The model is `VarianceThreshold` followed by `HistGradientBoostingRegressor`, with
100 iterations and at most 15 leaves. Internal random-row early stopping is disabled
to avoid mixing observations from the same engine. This was a fixed small-model exercise,
not a hyperparameter search or an official benchmark claim.

For the compatible scikit-learn 1.5.2 model, saved test results are:

| Model | RMSE (cycles) | MAE (cycles) | Summed NASA score |
| --- | ---: | ---: | ---: |
| Mean baseline | 52.976 | 41.119 | 285074.190 |
| Gradient boosting | 26.136 | 18.857 | 6411.285 |

Lower is better for all three metrics. The NASA score sums an asymmetric exponential
penalty over observations: with error `prediction - actual`, the per-row penalty is
`exp(-error / 13) - 1` for negative errors and `exp(error / 10) - 1` otherwise.
Overestimating remaining life is penalized more heavily. Metrics are FD001-only;
results from different subsets must be reported separately.

## Why Two Python Environments?

The root project uses Python 3.12 and scikit-learn 1.9.1. Running the training notebook
normally writes the original artifact to `models/cmapss-fd001`.

The chosen Azure serving environment uses Python 3.10 and scikit-learn 1.5.2. Loading
the original artifact there failed during deserialization. The same training notebook
was rerun under the separate [deployment/sklearn15](deployment/sklearn15) project with
`MODEL_DIR` redirected to `models/cmapss-fd001-sklearn15`. That retrained artifact is the
one registered for the successful online test. This folder is not regenerated by an
ordinary run of the root-environment training notebook.

Keep the separate lockfile so that the compatible local environment can be restored:

```powershell
uv sync --locked --project deployment/sklearn15 --python 3.10
```

Do not treat scikit-learn pickle/joblib artifacts as portable across library versions.
The local compatibility environment is not an exact copy of the Azure image; recorded
dependency versions and actual load/prediction checks matter. Do not load untrusted
joblib artifacts, because deserialization can execute code.

## Artifacts and Deployment

Generated model folders contain `model.joblib`, `schema.json`, metrics, provenance and
dependency metadata, test predictions, and example JSON inputs and predictions. Raw
data, model binaries, virtual environments, and credentials stay outside Git. A fresh
clone must acquire the data and recreate or retrieve the appropriate model artifacts.

Register the trained model folder, not `deployment/sklearn15` or its virtual environment.
The scoring script needs `model.joblib` and the adjacent `schema.json`.

[deployment/README.md](deployment/README.md) explains model registration, environment
compatibility, online versus batch inference, authentication, troubleshooting, and cost.
[deployment/score.py](deployment/score.py) supports both online JSON records and batch
CSV file paths. The repository does not automatically provision Azure resources.

The online notebook currently targets a named deployment with the
`azureml-model-deployment` header. This allows testing even with 0% routed traffic;
ordinary requests without that header need a configured traffic allocation. The endpoint
URL and deployment name are example configuration, not reusable resources for other users.

## Tests and Safety

```powershell
uv run --locked python -W ignore -m unittest discover -s tests -v
```

Tests cover preparation, RUL labels, engine splits, feature mapping, local training,
artifact serialization, scoring contracts, and offline batch-request behavior. They
do not provision cloud resources or replace a live Azure smoke test.

Do not commit access tokens, keys, data, model binaries, or notebook outputs containing
secrets. Review outputs before committing. Obtain authorization before provisioning or
submitting paid jobs. Real-time deployments cost money while idle; batch clusters need
idle scale-down, quota, and storage permissions. Remove unused cloud deployments when
finished; deleting a local directory does not delete Azure resources.

## Data Attribution and License

Dataset: A. Saxena and K. Goebel (2008), *Turbofan Engine Degradation Simulation Data Set*,
NASA Ames Research Center, NASA Prognostics Data Repository. The notebook retains the
supplied readme and damage-propagation paper alongside the raw files. Archive source:
<https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip>.

Original project code and documentation are licensed under the [MIT License](LICENSE).
The license does not relicense NASA datasets, supplied papers, or third-party packages.
Those materials retain their own terms and attribution requirements; public download
availability alone does not establish redistribution permission.