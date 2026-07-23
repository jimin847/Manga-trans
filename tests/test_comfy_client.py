"""
comfy_client 단위 테스트
"""
import json
import sys
import time
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from comfy_client import ComfyClient, flat_fill


# ─── flat_fill (no mocking needed — pure function) ──────────────────────────

def _mask_and_image():
    """100x100 image with a 30x30 white-masked region at center.
    Mask covers only the center; pixels INSIDE the mask are text (dark gray = 100)."""
    img_arr = np.ones((100, 100, 3), dtype=np.uint8) * 255  # white bg
    mask_arr = np.zeros((100, 100), dtype=np.uint8)
    mask_arr[35:65, 35:65] = 255  # mask: center 30x30
    mask = Image.fromarray(mask_arr, mode="L")

    # Draw dark text in the mask region
    img_arr[40:60, 40:60] = (100, 100, 100)
    img = Image.fromarray(img_arr)
    return img, mask


class TestFlatFill:
    def test_flat_fill_white_bubble(self):
        """White-background bubble should be filled white."""
        img, mask = _mask_and_image()

        # bbox must extend beyond mask to have background pixels for sampling
        bubbles = [{"bbox": [30, 30, 70, 70]}]  # larger than mask (35:65)
        result, remaining = flat_fill(img, mask, bubbles=bubbles)

        # Mask area (center) should be filled white (bg is 255, text was 100)
        result_arr = np.array(result)
        center_pixel = result_arr[50, 50].tolist()
        # bg sampled from white area outside mask → mean > 220 → flat fill to white
        assert center_pixel == [255, 255, 255]

        # Remaining should be None (entirely handled by flat fill)
        assert remaining is None or not np.any(np.array(remaining) > 128)

    def test_flat_fill_dark_bubble(self):
        """Dark-background bubble should remain for LaMa."""
        # Dark image with a dark bubble area
        img_arr = np.ones((100, 100, 3), dtype=np.uint8) * 50
        mask_arr = np.zeros((100, 100), dtype=np.uint8)
        mask_arr[35:65, 35:65] = 255
        mask = Image.fromarray(mask_arr)

        # bbox larger than mask to sample background
        bubbles = [{"bbox": [30, 30, 70, 70]}]
        result, remaining = flat_fill(Image.fromarray(img_arr), mask, bubbles=bubbles)

        result_arr = np.array(result)
        assert result_arr[50, 50].tolist() == [50, 50, 50]  # unchanged (dark bg < 220)

        assert remaining is not None
        assert np.any(np.array(remaining) > 128)  # still needs LaMa

    def test_flat_fill_no_bubbles_legacy(self):
        """Without bubbles, entire mask should be filled."""
        img, mask = _mask_and_image()
        result, remaining = flat_fill(img, mask)

        result_arr = np.array(result)
        assert result_arr[50, 50].tolist() == [255, 255, 255]

        assert remaining is None

    def test_flat_fill_tiny_bubble(self):
        """Tiny bubble (<4px) should be skipped from processing."""
        img, mask = _mask_and_image()
        mask_arr = np.array(mask)
        mask_arr[5:9, 5:9] = 255  # make it part of mask (4px)
        mask = Image.fromarray(mask_arr)

        bubbles = [{"bbox": [5, 5, 9, 9]}]  # 4px exactly — bx2-bx1=4 → not skipped, but tiny
        result, remaining = flat_fill(img, mask, bubbles=bubbles)

        result_arr = np.array(result)
        # Tiny region: bg_pixels < 10 → removed from remaining but not filled
        assert remaining is not None
        remaining_arr = np.array(remaining)
        assert not (remaining_arr[5:9, 5:9] > 128).any()

    def test_flat_fill_edge_clamp(self):
        """Bbox should clamp to image boundaries."""
        img, mask = _mask_and_image()
        bubbles = [{"bbox": [-5, -5, 105, 105]}]
        result, remaining = flat_fill(img, mask, bubbles=bubbles)

        result_arr = np.array(result)
        # Should not crash, fill works within bounds
        assert result_arr.shape == (100, 100, 3)


# ─── ComfyClient (mocked API) ───────────────────────────────────────────────

@pytest.fixture
def client():
    return ComfyClient(base_url="http://test:8188")


class TestComfyClient:
    def test_init(self):
        c = ComfyClient("http://localhost:8188")
        assert c.base_url == "http://localhost:8188"

    def test_init_trailing_slash_stripped(self):
        c = ComfyClient("http://localhost:8188/")
        assert c.base_url == "http://localhost:8188"

    @patch("comfy_client.requests.post")
    def test_upload_image(self, mock_post):
        mock_post.return_value = Mock(status_code=200)
        mock_post.return_value.json.return_value = {"name": "inp_img_123.png"}

        c = ComfyClient("http://test:8188")
        result = c._upload_image(b"fake-image-bytes", "test.png")

        assert result == "inp_img_123.png"
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert kwargs["files"]["image"][0] == "test.png"
        assert kwargs["timeout"] == 30

    @patch("comfy_client.requests.post")
    def test_upload_image_failure(self, mock_post):
        mock_post.side_effect = Exception("Connection refused")

        c = ComfyClient("http://test:8188")
        with pytest.raises(Exception, match="Connection refused"):
            c._upload_image(b"data", "test.png")

    @patch("comfy_client.requests.post")
    def test_queue_prompt(self, mock_post):
        mock_post.return_value = Mock(status_code=200)
        mock_post.return_value.json.return_value = {"prompt_id": "abc-123"}

        c = ComfyClient("http://test:8188")
        result = c._queue_prompt({"dummy": "workflow"})

        assert result == "abc-123"
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert "prompt" in kwargs["json"]

    @patch("comfy_client.requests.get")
    def test_wait_for_result_history(self, mock_get):
        """History returns the result on first check."""
        mock_get.return_value = Mock(status_code=200)
        mock_get.return_value.json.return_value = {"prompt-42": {"status": "done"}}

        c = ComfyClient("http://test:8188")
        result = c._wait_for_result("prompt-42", timeout=10)

        assert result == {"status": "done"}

    @patch("comfy_client.requests.get")
    def test_wait_for_result_timeout(self, mock_get):
        """Never finishes → TimeoutError."""
        mock_get.return_value = Mock(status_code=404)

        c = ComfyClient("http://test:8188")
        start = time.time()
        with pytest.raises(TimeoutError):
            c._wait_for_result("prompt-42", timeout=3)
        # Should have taken ~3 seconds
        elapsed = time.time() - start
        assert abs(elapsed - 3) < 2

    @patch("comfy_client.requests.get")
    def test_download_image(self, mock_get):
        """Download image from history outputs."""
        img = Image.new("RGB", (10, 10), color=(255, 0, 0))
        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        mock_get.return_value = Mock(status_code=200)
        mock_get.return_value.content = buf.read()

        history = {
            "outputs": {
                "4": {
                    "images": [
                        {"filename": "manga_clean_v2_00001.png", "subfolder": "", "type": "output"}
                    ]
                }
            }
        }

        c = ComfyClient("http://test:8188")
        result = c._download_image(history)

        assert result is not None
        assert result.size == (10, 10)

    def test_download_image_empty_history(self):
        """No images in history → return None."""
        c = ComfyClient("http://test:8188")
        result = c._download_image({"outputs": {}})
        assert result is None

    @patch("comfy_client.ComfyClient._upload_image")
    @patch("comfy_client.ComfyClient._queue_prompt")
    @patch("comfy_client.ComfyClient._wait_for_result")
    @patch("comfy_client.ComfyClient._download_image")
    def test_run_inpaint_lama_full(self, mock_dl, mock_wait, mock_queue, mock_upload):
        """Full end-to-end: upload → queue → wait → download."""
        mock_upload.side_effect = ["img_123.png", "mask_123.png"]
        mock_queue.return_value = "prompt-42"
        mock_wait.return_value = {"status": "done"}
        mock_dl.return_value = Image.new("RGB", (10, 10), color=(0, 255, 0))

        c = ComfyClient("http://test:8188")
        img = Image.new("RGB", (20, 20), color=(255, 255, 255))
        mask = Image.new("L", (20, 20), color=255)

        result = c.run_inpaint_lama(img, mask)

        assert result is not None
        assert result.size == (10, 10)

        assert mock_upload.call_count == 2
        mock_queue.assert_called_once()
        mock_wait.assert_called_once_with("prompt-42", timeout=180)
        mock_dl.assert_called_once()

    @patch("comfy_client.ComfyClient._upload_image")
    @patch("comfy_client.ComfyClient._queue_prompt")
    def test_run_inpaint_lama_failure_returns_none(self, mock_queue, mock_upload):
        """Exception in run_inpaint_lama returns None (not crash)."""
        mock_upload.side_effect = Exception("Server unavailable")

        c = ComfyClient("http://test:8188")
        img = Image.new("RGB", (20, 20), color=(255, 255, 255))
        mask = Image.new("L", (20, 20), color=255)

        result = c.run_inpaint_lama(img, mask)
        assert result is None
