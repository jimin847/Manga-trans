import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import main
from antigravity_client import AntigravityCliClient
from ocr.vlm_ocr import VlmOcr
from translation.translator import Translator


def test_antigravity_client_stages_images_and_uses_safe_print_mode(tmp_path, monkeypatch):
    binary = tmp_path / "agy"
    binary.write_text("fake")
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        staged = list(Path(kwargs["cwd"]).glob("*.png"))
        assert len(staged) == 1
        assert staged[0].is_file()
        return SimpleNamespace(returncode=0, stdout="RESULT\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    client = AntigravityCliClient(model="gemini-test", timeout=30, binary=str(binary))
    result = client.generate("Do the task", images={"b001": Image.new("RGB", (8, 8), "white")})

    assert result == "RESULT"
    command = captured["command"]
    assert command[1] == "-p"
    assert command[2].startswith("Do the task")
    assert "IMAGE FILES" in command[2]
    assert "--agent" in command and "gemini-test" in command
    assert "--sandbox" in command
    assert "--add-dir" in command
    assert "--dangerously-skip-permissions" not in command
    assert captured["kwargs"]["stdin"] is subprocess.DEVNULL


def test_antigravity_client_fails_closed_on_empty_output(tmp_path, monkeypatch):
    binary = tmp_path / "agy"
    binary.write_text("fake")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    client = AntigravityCliClient(model="gemini-test", binary=str(binary))

    assert client.generate("prompt") is None
    assert "no output" in client.last_error.lower()


def test_vlm_ocr_parses_antigravity_batch_and_rejects_unknown_ids(monkeypatch):
    ocr = VlmOcr(provider="antigravity-cli", model="gemini-test")

    class FakeClient:
        available = True

        def generate(self, prompt, images=None):
            assert set(images) == {"r0001", "r0002"}
            return """```json
            [{"id":"r0001","text":"ごめーん！"},
             {"id":"r0002","text":"[NO TEXT]"},
             {"id":"invented","text":"偽物"}]
            ```"""

    monkeypatch.setattr(ocr, "_get_cli_client", lambda model: FakeClient())
    crop = Image.new("RGB", (8, 8), "white")

    result = ocr.read_batch([
        ("r0001", crop, "vertical"),
        ("r0002", crop, "horizontal"),
    ])

    assert result == {"r0001": "ごめーん！", "r0002": None}


def test_run_ocr_batches_antigravity_crops(monkeypatch, tmp_path):
    calls = []

    class FakeOcr:
        def check_auth(self, api_key):
            return True

        def read_batch(self, crops):
            calls.append(crops)
            return {item_id: "それに人間となんて付き合えないわ" for item_id, _, _ in crops}

    class FakeCache:
        def hash_image(self, payload):
            return str(len(payload))

        def get(self, key):
            return None

        def set(self, key, value):
            pass

    monkeypatch.setattr(main, "get_ocr_client", lambda config: FakeOcr())
    detection = {
        "page_id": "010",
        "bubbles": [{"id": "b001", "bbox": [0, 0, 30, 50]}],
        "floating_texts": [],
        "text_mask": None,
    }
    config = {
        "ocr": {
            "provider": "antigravity-cli",
            "model": "gemini-test",
            "crop_upscale": 1,
            "multi_view": False,
            "batch_max_items": 12,
        },
        "output": {"save_crops": False},
    }

    main.run_ocr(
        detection,
        Image.new("RGB", (40, 60), "white"),
        config,
        api_key="",
        tmp_dir=tmp_path,
        cache=FakeCache(),
    )

    assert len(calls) == 1
    assert detection["bubbles"][0]["ocr_text"] == "それに人間となんて付き合えないわ"
    assert detection["bubbles"][0]["ocr_status"] == "accepted"


def test_translator_uses_independent_antigravity_review_model(monkeypatch):
    translator = Translator(
        provider="antigravity-cli",
        model="gemini-draft",
        review_model="claude-review",
    )
    used_models = []

    class FakeClient:
        available = True
        last_error = ""

        def __init__(self, model):
            self.model = model

        def generate(self, prompt, images=None):
            used_models.append(self.model)
            return '[{"id":"b001","translation":"미안해!","type":"dialogue"}]'

    monkeypatch.setattr(translator, "_get_cli_client", lambda model: FakeClient(model))
    source = [{"id": "b001", "text": "ごめん！"}]
    draft = translator.translate_batch(source)
    reviewed = translator.review_batch(source, draft)

    assert draft[0]["translation"] == "미안해!"
    assert reviewed[0]["translation"] == "미안해!"
    assert used_models == ["gemini-draft", "claude-review"]


def test_antigravity_provider_needs_cli_not_api_key(monkeypatch):
    monkeypatch.setattr(main.shutil, "which", lambda name: "/usr/local/bin/agy")

    assert main._provider_has_credentials("antigravity-cli", "") is True
