import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from web_gui import build_pipeline_command, discover_result_items, load_gui_defaults


def test_build_pipeline_command_for_single_page():
    payload = {
        "mode": "single",
        "image_path": "/tmp/page.png",
        "config_path": "config.yaml",
        "output_dir": "/tmp/out",
        "api_key": "secret",
    }

    command, output_dir = build_pipeline_command(payload, REPO_ROOT)

    assert command[:2] == [sys.executable, str(REPO_ROOT / "main.py")]
    assert "--api-key" in command
    assert command[-1] == "/tmp/page.png"
    assert output_dir == "/tmp/out"


def test_build_pipeline_command_for_batch_run():
    payload = {
        "mode": "batch",
        "patterns": "chapter/*.png\nchapter/*.jpg",
        "config_path": "config.yaml",
        "output_dir": "/tmp/out",
        "max_pages": 3,
        "api_key": "",
    }

    command, output_dir = build_pipeline_command(payload, REPO_ROOT)

    assert command[:2] == [sys.executable, str(REPO_ROOT / "run_batch.py")]
    assert "--max-pages" in command
    assert command[-2:] == ["chapter/*.png", "chapter/*.jpg"]
    assert output_dir == "/tmp/out"
    assert any(arg.endswith("batch_report.json") for arg in command)


def test_discover_result_items_reads_saved_result_json(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    final_image = out_dir / "page_ko.png"
    final_image.write_bytes(b"png")
    result_path = out_dir / "page_result.json"
    result_path.write_text(
        json.dumps(
            {
                "page_id": "page",
                "final_image": str(final_image),
                "detection": {
                    "bubbles": [
                        {"id": "b001", "translation": "번역1"},
                        {"id": "b002", "translation": None},
                    ]
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    items = discover_result_items(out_dir)

    assert items[0]["page_id"] == "page"
    assert items[0]["translated_count"] == 1
    assert items[0]["final_image"] == str(final_image)


def test_load_gui_defaults_reads_repo_config():
    defaults = load_gui_defaults(REPO_ROOT)

    assert defaults["config_path"] == "config.yaml"
    assert defaults["mode"] == "single"
    assert "output_dir" in defaults
