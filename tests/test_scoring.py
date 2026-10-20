import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class ScoringTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "deployment/score.py"
        spec = importlib.util.spec_from_file_location("score", path)
        self.score = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.score)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        folder = Path(self.directory.name) / "uploaded-model"
        folder.mkdir()
        training = pd.DataFrame({"cycle": [1, 2, 3], "sensor": [4, 5, 6]})
        model = LinearRegression().fit(training, [9, 8, 7])
        joblib.dump(model, folder / "model.joblib")
        (folder / "schema.json").write_text(json.dumps({"features": ["cycle", "sensor"]}))
        with patch.dict(os.environ, {"AZUREML_MODEL_DIR": self.directory.name}):
            self.score.init()

    def test_records_reordered_and_wrapper_supported(self):
        records = [{"sensor": 5, "cycle": 2}]
        result = self.score.run(json.dumps(records))
        np.testing.assert_allclose(result["predictions"], [8])
        self.assertEqual(result["unit"], "cycles")
        self.assertEqual(result, self.score.run(json.dumps({"data": records})))

    def test_invalid_requests_rejected(self):
        for records in [
            [], {}, [1], [{"cycle": 2}],
            [{"cycle": 2, "sensor": 5, "engine_id": 1}],
            [{"cycle": True, "sensor": 5}],
            [{"cycle": "2", "sensor": 5}],
            [{"cycle": None, "sensor": 5}],
            [{"cycle": float("nan"), "sensor": 5}],
            [{"cycle": float("inf"), "sensor": 5}],
            [{"cycle": 2, "sensor": 5}] * 1001,
        ]:
            with self.subTest(records=str(records)[:80]), self.assertRaises(ValueError):
                self.score.run(json.dumps(records))

    def test_missing_model_rejected(self):
        with tempfile.TemporaryDirectory() as empty, patch.dict(os.environ, {"AZUREML_MODEL_DIR": empty}):
            with self.assertRaisesRegex(ValueError, "exactly one"):
                self.score.init()

    def test_batch_csv_files_reordered_and_identified(self):
        first = Path(self.directory.name) / "first.csv"
        second = Path(self.directory.name) / "second.csv"
        pd.DataFrame({"sensor": [5, 6], "cycle": [2, 3], "engine_id": [1, 1]}).to_csv(first, index=False)
        pd.DataFrame({"cycle": [1], "sensor": [4]}).to_csv(second, index=False)
        result = self.score.run([str(first), str(second)])
        self.assertEqual(list(result.columns), ["source_file", "row_number", "predicted_rul_cycles"])
        self.assertEqual(result["source_file"].tolist(), [str(first), str(first), str(second)])
        self.assertEqual(result["row_number"].tolist(), [1, 2, 1])
        np.testing.assert_allclose(result["predicted_rul_cycles"], [8, 7, 9])

    def test_batch_has_no_online_row_limit(self):
        path = Path(self.directory.name) / "large.csv"
        pd.DataFrame({"cycle": [2] * 1001, "sensor": [5] * 1001}).to_csv(path, index=False)
        result = self.score.run([path])
        self.assertEqual(len(result), 1001)
        np.testing.assert_allclose(result["predicted_rul_cycles"], 8)

    def test_invalid_batch_features_rejected(self):
        path = Path(self.directory.name) / "invalid.csv"
        for values in [
            {"cycle": [2]},
            {"cycle": [], "sensor": []},
            {"cycle": [True], "sensor": [5]},
            {"cycle": ["invalid"], "sensor": [5]},
            {"cycle": [None], "sensor": [5]},
            {"cycle": [float("inf")], "sensor": [5]},
        ]:
            with self.subTest(values=values):
                pd.DataFrame(values).to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    self.score.run([str(path)])

    def test_invalid_batch_paths_rejected(self):
        for mini_batch in [[], [1], ["input.json"]]:
            with self.subTest(mini_batch=mini_batch), self.assertRaises(ValueError):
                self.score.run(mini_batch)