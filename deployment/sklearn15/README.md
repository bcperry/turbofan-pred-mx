# Python 3.10 / scikit-learn 1.5 compatibility

The original Python 3.12 / scikit-learn 1.9.1 artifact was tested first under this
isolated environment. Loading failed with a NumPy BitGenerator deserialization error,
after a scikit-learn inconsistent-version warning. No original files were replaced.

The existing training notebook's four code cells were then executed under Python
3.10.17 with MODEL_DIR redirected to `models/cmapss-fd001-sklearn15`. The algorithm,
seed, engine split, features, and evaluation protocol were unchanged. The new folder
contains the model, schema, metadata, metrics, predictions and example JSON inputs.

Test results on 100 separate NASA test engines: RMSE 26.136 cycles, MAE 18.857 cycles.
The artifact was reloaded and its predictions checked under the older environment.
This is a dependency-compatibility retraining, not a hyperparameter search; the
test set must not be used to tune subsequent models.

## Dependencies

- Python 3.10.17
- scikit-learn 1.5.2
- NumPy 1.26.4
- SciPy 1.13.1
- pandas 2.2.3
- joblib 1.4.2
- threadpoolctl 3.5.0

Restore the environment from the repository root:

```powershell
uv sync --locked --project deployment/sklearn15 --python 3.10
```

## Azure registration

Upload `models/cmapss-fd001-sklearn15` as a new custom-model version, and enter
framework version `1.5.2`. Continue to use the [shared scoring script](../score.py).
Do not upload this directory or its `.venv` as the registered model.

The compatible artifact was registered and deployed in Azure Government. A live
Entra-authenticated online call returned HTTP 200 with the same two example predictions
as local scoring. The observed Azure environment has scikit-learn 1.5.2 and NumPy 1.26.4,
but differs from this local lockfile in other dependencies, including SciPy and joblib.
This project is not an exact replica of that image or a complete inference-server
environment. Validate compatibility rather than assuming cross-version serialization.

The default root-environment training notebook still writes the original Python 3.12
artifact. Reproducing this older-runtime model requires running its training cells under
this environment with `MODEL_DIR` redirected after the setup cell; record this lockfile
and the actual package versions with the resulting artifact. Do not overwrite the
original model as a shortcut or claim that running `uv sync` retrains a model.

The obsolete Python 3.12 deployment image definition was removed. Restore generated
local environments with uv when needed. See the [deployment guide](../README.md) for
registration, serving requirements, authentication, and the batch alternative.