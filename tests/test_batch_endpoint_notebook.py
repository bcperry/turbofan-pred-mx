from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import requests


class BatchEndpointNotebookTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "notebooks/call_batch_endpoint.ipynb"
        self.notebook = json.loads(path.read_text(encoding="utf-8"))
        self.code = ["".join(cell["source"]) for cell in self.notebook["cells"] if cell["cell_type"] == "code"]
        self.input_uri = "azureml://datastores/workspaceblobstore/paths/test.csv"
        self.credential = Mock()
        self.credential.get_token.return_value.token = "test-secret-token"
        self.response = requests.Response()
        self.response.status_code = 202
        self.response._content = b'{"name":"test-job","properties":{"status":"NotStarted"}}'
        self.response.request = requests.Request("POST", "https://example.inference.ml.azure.us/jobs").prepare()
        self.output = io.StringIO()
        self.scope = {}

    def execute(self, code):
        with redirect_stdout(self.output):
            exec(code, self.scope)

    def configure(self):
        with patch("azure.identity.AzureCliCredential", return_value=self.credential):
            self.execute(self.code[0])
        self.scope["INPUT_URI"] = self.input_uri
        self.execute(self.code[1])

    def test_direct_request_and_response_without_workspace_configuration(self):
        for cell in self.notebook["cells"]:
            self.assertEqual(cell["metadata"]["language"], "python" if cell["cell_type"] == "code" else "markdown")
            self.assertTrue(cell["metadata"]["id"])
        self.configure()
        self.assertNotIn("client", self.scope)
        self.assertNotIn("SUBSCRIPTION_ID", self.scope)
        self.assertNotIn("RESOURCE_GROUP", self.scope)
        self.assertEqual(len(self.code), 4)
        with patch("requests.request", return_value=self.response) as send:
            self.execute(self.code[2])
        self.assertEqual(send.call_count, 1)
        arguments = send.call_args.kwargs
        self.assertEqual(arguments["method"], "POST")
        self.assertEqual(arguments["url"], "https://aml-demo-turbine-local.usgovvirginia.inference.ml.azure.us/jobs")
        self.assertEqual(arguments["json"]["properties"]["InputData"]["UriFile"]["Uri"], self.input_uri)
        self.assertNotIn("azureml-model-deployment", arguments["headers"])
        self.credential.get_token.assert_called_once_with("https://ml.azure.us/.default")
        self.execute(self.code[3])
        self.assertEqual(self.scope["job_name"], "test-job")
        self.assertIn("HTTP 202", self.output.getvalue())
        self.assertNotIn("test-secret-token", self.output.getvalue())
        self.assertEqual(self.response.request.headers["Authorization"], "Bearer <redacted>")

    def test_missing_and_local_input_uris_rejected_before_submission(self):
        self.configure()
        for input_uri in ["", "batch-test.csv", "C:/data/batch-test.csv", "http://storage/test.csv"]:
            with self.subTest(input_uri=input_uri):
                self.scope["INPUT_URI"] = input_uri
                with self.assertRaises(ValueError):
                    self.execute(self.code[1])

    def test_error_body_displayed_before_raising(self):
        self.configure()
        self.response.status_code = 403
        self.response._content = b"Permission denied"
        with patch("requests.request", return_value=self.response):
            self.execute(self.code[2])
        with self.assertRaises(requests.HTTPError):
            self.execute(self.code[3])
        self.assertIn("HTTP 403", self.output.getvalue())
        self.assertIn("Permission denied", self.output.getvalue())

    def test_timeout_does_not_retry_or_keep_stale_job(self):
        self.configure()
        self.scope["job_name"] = "old-job"
        with patch("requests.request", side_effect=requests.Timeout) as send:
            with self.assertRaisesRegex(RuntimeError, "before resubmitting"):
                self.execute(self.code[2])
        self.assertEqual(send.call_count, 1)
        self.assertIsNone(self.scope["job_name"])