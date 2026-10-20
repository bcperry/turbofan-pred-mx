import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


model = None
features = None


def init():
    global model, features
    model_root = Path(os.environ["AZUREML_MODEL_DIR"])
    candidates = list(model_root.rglob("model.joblib"))
    if len(candidates) != 1:
        raise ValueError("Expected exactly one model.joblib in AZUREML_MODEL_DIR")
    model_path = candidates[0]
    schema = json.loads((model_path.parent / "schema.json").read_text(encoding="utf-8"))
    features = schema["features"]
    if not features or not all(isinstance(name, str) for name in features) or len(set(features)) != len(features):
        raise ValueError("Invalid feature names in schema.json")
    model = joblib.load(model_path)
    if list(model.feature_names_in_) != features:
        raise ValueError("Model input names do not match schema.json")


def run(raw_data):
    if model is None:
        raise RuntimeError("Model has not been initialized")
    if isinstance(raw_data, (list, tuple)):
        return _run_batch(raw_data)
    records = json.loads(raw_data)
    if isinstance(records, dict):
        records = records.get("data")
    if not isinstance(records, list) or not 1 <= len(records) <= 1000:
        raise ValueError("Supply 1 to 1000 records as a JSON list or an object with a data list")
    required = set(features)
    for record in records:
        if not isinstance(record, dict) or set(record) != required:
            raise ValueError("Each record must contain exactly the feature names in schema.json")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in record.values()):
            raise ValueError("All feature values must be numeric, not null, strings, or booleans")
    frame = pd.DataFrame(records, columns=features, dtype=float)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("All feature values must be finite")
    with threadpool_limits(limits=2):
        predictions = model.predict(frame)
    if not np.isfinite(predictions).all():
        raise RuntimeError("Model returned nonfinite predictions")
    return {"predictions": predictions.tolist(), "unit": "cycles"}


def _run_batch(mini_batch):
    if not mini_batch:
        raise ValueError("Supply at least one CSV file path")
    results = []
    for file_path in mini_batch:
        if not isinstance(file_path, (str, os.PathLike)):
            raise ValueError("Batch inputs must be CSV file paths")
        path = Path(file_path)
        if path.suffix.lower() != ".csv":
            raise ValueError("Batch inputs must be CSV files")
        frame = pd.read_csv(path)
        missing = set(features) - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name}: missing features: {sorted(missing)}")
        frame = frame.loc[:, features]
        if frame.empty:
            raise ValueError(f"{path.name}: input CSV has no rows")
        if any(
            not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype)
            for dtype in frame.dtypes
        ):
            raise ValueError(f"{path.name}: all feature values must be numeric, not booleans")
        frame = frame.astype(float)
        if not np.isfinite(frame.to_numpy()).all():
            raise ValueError(f"{path.name}: all feature values must be finite")
        with threadpool_limits(limits=2):
            predictions = model.predict(frame)
        if not np.isfinite(predictions).all():
            raise RuntimeError("Model returned nonfinite predictions")
        results.append(pd.DataFrame({
            "source_file": str(path),
            "row_number": np.arange(1, len(frame) + 1),
            "predicted_rul_cycles": predictions,
        }))
    return pd.concat(results, ignore_index=True)