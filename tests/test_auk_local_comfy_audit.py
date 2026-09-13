from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PACKAGE = Path(__file__).resolve().parents[1]
NODE_PATH = PACKAGE / "comfyui" / "ComfyUI-AuK-Local" / "nodes.py"


class ComfyNodeAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use the actual Comfy API; no inference model is imported or loaded.
        comfy_root = os.environ.get("AUK_TEST_COMFY_ROOT")
        if comfy_root:
            sys.path.insert(0, comfy_root)
        if importlib.util.find_spec("comfy_api") is None:
            raise unittest.SkipTest("Set AUK_TEST_COMFY_ROOT to test against an installed ComfyUI API")
        spec = importlib.util.spec_from_file_location("auk_nodes_audit", NODE_PATH)
        cls.nodes = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.nodes)

    def connection(self):
        return {
            "service_url": "http://127.0.0.1:7860", "token_file": "unused",
            "model": "flash", "cpu_offload": True, "keep_loaded": False,
        }

    def test_optional_connection_inputs_can_be_omitted(self):
        nodes = self.nodes
        with (
            patch.object(nodes, "resolve_token_file", return_value=Path("unused")) as resolve,
            patch.object(nodes.Client, "health", return_value={"protocol_version": "1.0"}),
        ):
            result = nodes.AuKLocalConnection.execute(service_url="http://127.0.0.1:7860", model="flash")
        resolve.assert_called_once_with("")
        self.assertEqual(result.result[0]["model"], "flash")

    def test_examples_match_real_schema_widget_order(self):
        nodes = self.nodes
        classes = {
            "AuKLocalConnection": nodes.AuKLocalConnection,
            "AuKLocalGenerateEdit": nodes.AuKLocalGenerateEdit,
        }
        workflows = list((PACKAGE / "comfyui" / "workflows").glob("AuK-*.json"))
        self.assertEqual(len(workflows), 3)
        for workflow in workflows:
            for node in json.loads(workflow.read_text(encoding="utf-8"))["nodes"]:
                if node["type"] not in classes:
                    continue
                with self.subTest(workflow=workflow.name, node=node["type"]):
                    # Comfy frontend orders required inputs before optional inputs.
                    specs = classes[node["type"]].INPUT_TYPES()
                    widgets = []
                    for group in ("required", "optional"):
                        for name, (kind, options) in specs.get(group, {}).items():
                            if kind not in {"STRING", "COMBO", "INT", "FLOAT", "BOOLEAN"}:
                                continue
                            widgets.append((name, options))
                            if options.get("control_after_generate"):
                                widgets.append((f"{name}:control", {"options": ["fixed", "increment", "decrement", "randomize"]}))
                    self.assertEqual(len(node["widgets_values"]), len(widgets))
                    values = dict(zip((name for name, _ in widgets), node["widgets_values"]))
                    for name, options in widgets:
                        value = values[name]
                        if "options" in options:
                            self.assertIn(value, options["options"])
                        if "min" in options:
                            self.assertGreaterEqual(value, options["min"])
                        if "max" in options:
                            self.assertLessEqual(value, options["max"])
                    if node["type"] == "AuKLocalConnection":
                        self.assertEqual(values["model"], "flash")
                        self.assertEqual(values["token_file"], "")
                    else:
                        self.assertEqual(values["seed:control"], "fixed")
                        self.assertEqual(values["refresh:control"], "increment")
                        self.assertEqual(values["refresh"], 0)
                        self.assertEqual(values["nfe_steps"], 32)
                        self.assertEqual(values["cfg_strength"], 2.0)
                        self.assertEqual(values["sway_sampling_coef"], -1.0)

    def execute_with_fake_client(self, task, audio):
        nodes = self.nodes
        payloads = []

        class FakeClient:
            def __init__(self, *_args):
                pass

            def health(self):
                return {"protocol_version": "1.0"}

            def json_request(self, method, path, payload=None, timeout=None):
                if method == "POST":
                    payloads.append(payload)
                    return {"state": "succeeded"}
                return {"instruction": "test"}

            def download(self, _path):
                return b"fake audio"

        with (
            patch.object(nodes, "Client", FakeClient),
            patch.object(nodes.torchaudio, "load", return_value=(nodes.torch.zeros(1, 24), 24000)),
        ):
            result = nodes.AuKLocalGenerateEdit.execute(self.connection(), task, "test", "", 1.0, 42, audio)
        return payloads[0], result

    def test_text_generation_ignores_connected_reference_audio(self):
        nodes = self.nodes
        # A valid 30 s reference must not consume the budget after switching to TTS.
        audio = {"waveform": nodes.torch.zeros(1, 1, 30 * 24000), "sample_rate": 24000}
        payload, _ = self.execute_with_fake_client(nodes.TASK_OPTIONS[0], audio)
        self.assertIsNone(payload["audio"])

    def test_reference_task_keeps_and_downmixes_stereo_audio(self):
        nodes = self.nodes
        audio = {"waveform": nodes.torch.zeros(1, 2, 2400), "sample_rate": 24000}
        payload, result = self.execute_with_fake_client(nodes.TASK_OPTIONS[1], audio)
        self.assertEqual(payload["audio"]["channels"], 1)
        self.assertEqual(payload["audio"]["frames"], 2400)
        self.assertEqual(result.result[0]["waveform"].shape, (1, 1, 24))

    def test_uncertain_submit_retries_same_request(self):
        nodes = self.nodes
        payload = {"request_id": "test-id"}
        calls = []

        class FakeClient:
            def json_request(self, method, path, payload=None, timeout=None):
                calls.append((method, path, payload))
                if len(calls) == 1:
                    raise TimeoutError("response lost")
                if method == "GET":
                    raise nodes.HttpResponseError(404, "not submitted")
                return {"request_id": payload["request_id"], "state": "queued"}

        with patch.object(nodes.time, "sleep", return_value=None):
            result = nodes.submit_with_recovery(FakeClient(), "test-id", payload)
        self.assertEqual(result["request_id"], "test-id")
        self.assertIs(calls[0][2], calls[2][2])


@unittest.skipUnless(os.name == "nt", "Windows installer")
class ComfyInstallerAuditTests(unittest.TestCase):
    def test_install_and_move_with_spaces_unicode_and_brackets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "整合包 [test]"
            comfy = root / "Comfy UI [test]"
            (comfy / "custom_nodes").mkdir(parents=True)
            (package / "scripts").mkdir(parents=True)
            (package / "data").mkdir()
            shutil.copytree(PACKAGE / "comfyui", package / "comfyui")
            script = package / "scripts" / "Install-ComfyUI-Node.ps1"
            shutil.copy2(PACKAGE / "scripts" / script.name, script)
            token = "temporary-test-token-not-a-real-credential"
            (package / "data" / "session-token").write_text(token, encoding="utf-8")
            completed = subprocess.run(
                ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), "-ComfyUIRoot", str(comfy)],
                check=False, capture_output=True, timeout=20,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr.decode("utf-8", errors="replace"))
            moved = root / "已移动 Comfy"
            comfy.rename(moved)
            installed = moved / "custom_nodes" / "ComfyUI-AuK-Local"
            config = json.loads((installed / "auk-local-config.json").read_text(encoding="utf-8"))
            self.assertFalse(Path(config["token_file"]).is_absolute())
            self.assertEqual((installed / config["token_file"]).read_text(encoding="utf-8"), token)
            self.assertEqual(len(list((installed / "example_workflows").glob("AuK-*.json"))), 3)


if __name__ == "__main__":
    unittest.main()
