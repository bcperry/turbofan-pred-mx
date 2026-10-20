import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


class LocalTrainingNotebookTests(unittest.TestCase):
    def test_training_export_and_inference(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "notebooks/train_local_cmapss.ipynb").read_text(encoding="utf-8-sig"))
        cells = [
            compile("".join(cell["source"]), "train_local_cmapss.ipynb", "exec")
            for cell in notebook["cells"] if cell["cell_type"] == "code"
        ]
        scope = {}
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            exec(cells[0], scope)
            data_dir = Path(directory) / "data"
            data_dir.mkdir()
            scope["DATA_DIR"] = data_dir
            scope["MODEL_DIR"] = Path(directory) / "model"
            for kind, engine_count, cycles in [("train", 5, 20), ("test", 2, 10)]:
                frame = pd.DataFrame([
                    [engine, cycle, 0, 0, 100] + [cycle + engine / 100] * 21
                    for engine in range(1, engine_count + 1)
                    for cycle in range(1, cycles + 1)
                ])
                frame.to_csv(data_dir / f"{kind}_FD001.txt", sep=" ", index=False, header=False)
            pd.DataFrame([11, 22]).to_csv(data_dir / "RUL_FD001.txt", index=False, header=False)
            for cell in cells[1:]:
                exec(cell, scope)
            self.assertTrue(set(scope["train_engines"]).isdisjoint(scope["validation_engines"]))
            self.assertEqual(scope["test_rows"]["rul_cycles"].tolist(), [11, 22])
            self.assertEqual(scope["training"].groupby("engine_id")["rul_cycles"].apply(list).iloc[0], list(range(19, -1, -1)))
            self.assertFalse({"engine_id", "rul_cycles"}.intersection(scope["features"]))
            self.assertFalse(scope["model"][-1].early_stopping)
            self.assertIn("setting_3", scope["removed_constants"])
            self.assertEqual(scope["metrics"]["model_test"]["observations"], 2)
            folder = scope["MODEL_DIR"]
            model = joblib.load(folder / "model.joblib")
            schema = json.loads((folder / "schema.json").read_text())
            records = json.loads((folder / "example_input.json").read_text())
            expected = json.loads((folder / "example_predictions.json").read_text())
            with threadpool_limits(limits=2):
                np.testing.assert_allclose(model.predict(pd.DataFrame(records)[schema["features"]]), expected)
            self.assertTrue((folder / "test_predictions.csv").is_file())
            self.assertIn("scikit-learn", json.loads((folder / "metadata.json").read_text())["packages"])
            duplicate = pd.concat([pd.read_csv(data_dir / "train_FD001.txt", sep=r"\s+", header=None)] * 2)
            duplicate_path = data_dir / "duplicate.txt"
            duplicate.to_csv(duplicate_path, sep=" ", index=False, header=False)
            with self.assertRaisesRegex(AssertionError, "Duplicate"):
                scope["read_trajectories"](duplicate_path)