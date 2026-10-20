# Serve the C-MAPSS model

This folder keeps the shared [scoring script](score.py) and the
[Python 3.10/scikit-learn 1.5 reproduction project](sklearn15). It does not automatically
register models, build images, or provision resources. See the [project guide](../README.md)
for the notebook workflow. Predictions concern NASA simulations, not certified maintenance.

## Use the Compatible Model

The working artifact is generated locally at `models/cmapss-fd001-sklearn15`.
Register that folder through **Models > Register > From local files**, with custom/
unspecified model type. It contains `model.joblib`, adjacent `schema.json`, and supporting
JSON/CSV files. Do not upload this dependency project or a `.venv` folder as a model.

The original `models/cmapss-fd001` artifact uses scikit-learn 1.9.1 and failed to load
in the selected 1.5.2 environment. That produced the NumPy `PCG64` error during `init()`;
the subsequent 502 health probes were a consequence, not an authentication problem.
Register the retrained artifact as a new version and select that version when deploying.
An extra folder level in the registered artifact is supported. Only load trusted models.

## Online Deployment in Studio

1. Select the compatible registered model version and deploy to a managed real-time endpoint.
2. Upload [score.py](score.py) as the scoring script.
3. Choose an inference environment compatible with Python 3.10/scikit-learn 1.5.2 and
   containing `azureml-inference-server-http`. Check the actual full dependency versions;
   the local compatibility project is not an inference-server image definition.
4. Select an authorized VM size and instance count with sufficient regional/family quota.
   The successful exercise used one `Standard_DS3_v2` instance.
5. Wait for the deployment, not just the endpoint, to report success.
6. Allocate traffic for ordinary endpoint calls, or target the deployment explicitly
   with the `azureml-model-deployment` request header. Targeted tests work at 0% traffic.

The successful exercise used model `cmapss-fd001-local-regressor:2`, deployment
`cmapss-fd001-local-regressor-2`, and custom environment `turbine-local-env:1`.
These are workspace-specific examples. The removed Python 3.12 custom image files
are not part of the current serving path. The selected Azure environment was configured
through Studio; this repository does not yet contain its exact reproducible image build.

## Call the Online API

Use [the online notebook](../notebooks/call_online_endpoint.ipynb) with the root Python
3.12 kernel. It reads local example input JSON and sends it directly to `/score`;
there is no storage upload and no local model deserialization.

The endpoint uses Microsoft Entra authentication. Sign in to Azure Government using
Azure CLI. `AzureCliCredential` obtains a token for `https://ml.azure.us/.default`;
no endpoint key is needed. Your identity needs
`Microsoft.MachineLearningServices/workspaces/onlineEndpoints/score/action`.
The notebook displays the request and response with the token redacted.

Online input is a JSON list of 1 to 1000 records, or an object with a `data` list.
Every record must contain exactly the 25 named numeric features in `schema.json`.
Order does not matter. Extra columns, missing inputs, nulls, booleans, strings,
NaN, and infinity are rejected. IDs and target labels are not model inputs.

The live two-record smoke test returned HTTP 200 on 2026-10-16:

```json
{"predictions": [183.7632235282114, 144.3027393871352], "unit": "cycles"}
```

Invalid input raises a scoring error, not a custom HTTP 400 response. Predictions
are returned without clipping, so future inputs can produce negative values.
If a deployment fails, inspect its container logs. If invocation fails with 401/403,
check token audience, login, and scoring permissions. If an ordinary call has no route,
check traffic allocation or send the deployment header.

## Batch Alternative

The same script accepts a list of CSV paths supplied by the batch runtime. Each file
must have a header and all required model features. Additional columns are excluded
from prediction inputs. The returned DataFrame has `source_file`, `row_number` (1-based
data row), and `predicted_rul_cycles`. With **Append row**, Azure writes rows without a header.

[The batch notebook](../notebooks/call_batch_endpoint.ipynb) previews and submits an
asynchronous job using an already uploaded CSV's datastore/storage URI. Batch endpoints
do not accept local file bytes as their scoring payload. The immediate response identifies
a job; predictions are read from job outputs later. Mini-batch size counts files, not rows.

Batch requires an AML compute cluster and a model-compatible batch environment containing
`azureml-core` and `azureml-dataset-runtime[fuse]`. The local compatibility project alone
is not a ready-to-deploy batch image. The batch scoring code was tested locally;
no successful remote batch job is claimed here.

## Local Checks and Cost

Run from the repository root:

```powershell
uv sync --locked --project deployment/sklearn15 --python 3.10
uv run --locked --project deployment/sklearn15 python -W ignore -m unittest discover -s tests -p test_scoring.py -v
```

Generated virtual environments and caches can be deleted and restored without changing
the lockfiles. Online instances remain billable at 0% traffic. Delete unused Azure
deployments when finished; removing local files does not remove cloud resources.
Batch clusters need idle scale-down, and deleting a batch endpoint does not delete its cluster.