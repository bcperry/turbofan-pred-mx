import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd


class PreparationNotebookTests(unittest.TestCase):
    def setUp(self):
        notebook_path = Path(__file__).resolve().parents[1] / "notebooks/prepare_cmapss.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8-sig"))
        self.cells = [
            compile("".join(cell["source"]), str(notebook_path), "exec")
            for cell in notebook["cells"] if cell["cell_type"] == "code"
        ]
        self.scope = {}

    def test_registered_asset_preparation(self):
        def trajectory_bytes(engine_count, cycles):
            rows = [
                " ".join(map(str, [engine, cycle, 0, 0, 100] + [
                    cycle + sensor_number / 100 for sensor_number in range(1, 22)
                ]))
                for engine in range(1, engine_count + 1)
                for cycle in range(1, cycles + 1)
            ]
            return ("\n".join(rows) + "\n").encode()

        contents = {
            "train_FD001.txt": trajectory_bytes(5, 3),
            "test_FD001.txt": trajectory_bytes(2, 2),
            "RUL_FD001.txt": b"11\n22\n",
        }
        client = Mock()
        client.data.get.return_value = SimpleNamespace(
            id="azureml:nasa-cmapss:1",
            path="azureml://subscriptions/sub/resourcegroups/rg/workspaces/ws/"
                 "datastores/workspaceblobstore/paths/LocalUpload/engine%20data/",
        )
        client.datastores.get.return_value = SimpleNamespace(
            type="AzureBlob", account_name="example", container_name="data",
            endpoint="core.usgovcloudapi.net", credentials=SimpleNamespace(),
        )
        storage = Mock()

        def read_blob(container, blob_name):
            self.assertEqual(container, "data")
            self.assertTrue(blob_name.startswith("LocalUpload/engine data/"))
            blob = Mock()
            blob.download_blob.return_value.readall.return_value = contents[blob_name.rsplit("/", 1)[1]]
            return blob

        storage.get_blob_client.side_effect = read_blob
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            with patch("azure.ai.ml.MLClient.from_config", return_value=client) as connection, \
                 patch("azure.identity.DefaultAzureCredential") as identity, \
                 patch("azure.storage.blob.BlobServiceClient") as blob_type, \
                 patch.dict("sys.modules", {
                     "mltable": None, "azureml.fsspec": None,
                     "azureml.dataprep": None, "pkg_resources": None,
                 }):
                blob_type.return_value.__enter__.return_value = storage
                exec(self.cells[0], self.scope)
            client.data.get.assert_called_once_with(name="nasa-cmapss", version="1")
            self.assertEqual(connection.call_args.kwargs["cloud"], "AzureUSGovernment")
            client.datastores.get.assert_called_once_with("workspaceblobstore", include_secrets=True)
            blob_type.assert_called_once_with(
                account_url="https://example.blob.core.usgovcloudapi.net",
                credential=identity.return_value,
            )
            self.assertEqual(storage.get_blob_client.call_count, 3)
            self.scope["OUTPUT_DIR"] = Path(directory)
            for cell in self.cells[1:]:
                exec(cell, self.scope)
            self.assertTrue(set(self.scope["train_engines"]).isdisjoint(self.scope["validation_engines"]))
            self.assertEqual(self.scope["training"].groupby("engine_id")["rul_cycles"].apply(list).iloc[0], [2, 1, 0])
            self.assertEqual(self.scope["tables"]["test"]["rul_cycles"].tolist(), [11, 22])
            self.assertNotIn("engine_id", self.scope["features"])
            self.assertNotIn("setting_3", self.scope["features"])
            self.assertEqual(len(self.scope["sensor_column_mapping"]), 21)
            self.assertEqual(len(set(self.scope["columns"])), 26)
            expected_values = {
                "t2_fan_inlet_total_temperature_rankine": 1.01,
                "t24_lpc_outlet_total_temperature_rankine": 1.02,
                "t30_hpc_outlet_total_temperature_rankine": 1.03,
                "t50_lpt_outlet_total_temperature_rankine": 1.04,
                "p30_hpc_outlet_total_pressure_psia": 1.07,
                "ps30_hpc_outlet_static_pressure_psia": 1.11,
                "nf_physical_fan_speed_rpm": 1.08,
                "nrf_corrected_fan_speed_rpm": 1.13,
                "w31_hpt_coolant_bleed_lbm_per_s": 1.20,
                "w32_lpt_coolant_bleed_lbm_per_s": 1.21,
            }
            for column, expected_value in expected_values.items():
                self.assertAlmostEqual(self.scope["training"].iloc[0][column], expected_value)
            self.assertFalse(any(name.startswith("sensor_") for name in self.scope["features"]))
            self.assertEqual(
                self.scope["preparation"]["sensor_column_mapping"], self.scope["sensor_column_mapping"]
            )
            self.assertEqual(self.scope["preparation"]["source_asset_id"], "azureml:nasa-cmapss:1")
            self.assertTrue((Path(directory) / "train/MLTable").is_file())
            full = self.scope["tables"]["full"]
            self.assertEqual(len(full), 19)
            self.assertEqual(full["engine_uid"].nunique(), 7)
            self.assertFalse(full.duplicated(["engine_uid", "cycle"]).any())
            self.assertTrue(set(self.scope["columns"]).issubset(full.columns))
            self.assertEqual(full.groupby("source_split").size().to_dict(), {
                "train": 12, "validation": 3, "test": 4,
            })
            full_test = full[full["source_split"] == "test"]
            self.assertEqual(full_test["rul_cycles"].tolist(), [12, 11, 23, 22])
            self.assertEqual(
                full_test.groupby("engine_id").tail(1)["rul_cycles"].tolist(), [11, 22]
            )
            for split in ("train", "validation"):
                expected_engines = set(self.scope[f"{split}_engines"])
                rows = full[full["source_split"] == split]
                self.assertEqual(set(rows["engine_id"]), expected_engines)
                self.assertEqual(rows.groupby("engine_uid")["rul_cycles"].apply(list).iloc[0], [2, 1, 0])
            exported_full = pd.read_csv(Path(directory) / "full/data.csv")
            pd.testing.assert_frame_equal(exported_full, full)
            self.assertTrue((Path(directory) / "full/MLTable").is_file())
            self.assertEqual(client.data.create_or_update.call_count, 4)
            for split, call in zip(("train", "validation", "test", "full"), client.data.create_or_update.call_args_list):
                asset = call.args[0]
                self.assertEqual(asset.name, f"cmapss-fd001-{split}")
                self.assertEqual(asset.type, "mltable")
                self.assertEqual(asset.version, self.scope["PREPARED_VERSION"])
                self.assertEqual(Path(asset.path), Path(directory) / split)
                self.assertEqual(asset.tags["source_asset_id"], "azureml:nasa-cmapss:1")
                self.assertEqual(
                    json.loads((Path(directory) / split / "preparation.json").read_text()),
                    self.scope["preparation"],
                )
            duplicate = pd.concat([self.scope["source_tables"]["train"]] * 2)
            with self.assertRaisesRegex(AssertionError, "Duplicate"):
                self.scope["read_trajectories"](duplicate)