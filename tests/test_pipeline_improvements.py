import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import main
from comfy_client import flat_fill, opencv_inpaint
from ocr.vlm_ocr import VlmOcr
from translation.translator import Translator
from cache_manager import CacheManager
from scripts.pp_ocrv6_match import _load_vlm_json, match_pp_ocr_to_bubbles


def test_translator_applies_authoritative_glossary_and_name_policy():
    translator = Translator(
        glossary={"御幸": "미유키"},
        preserve_terms=["田中", "みゆき"],
    )
    prompt = translator._build_system_prompt()

    assert "Transliterate Japanese proper names into Hangul" in prompt
    assert "御幸" in prompt and "미유키" in prompt
    assert "田中" in prompt and "never preserve its Japanese glyphs" in prompt
    assert "Convert to Korean alphabet" not in prompt
    assert "NOTHING else" in prompt
    assert "no reasoning" in prompt
    assert "旦那様→남편님" in prompt


def test_vlm_ocr_auth_error_skips_fallback(monkeypatch):
    ocr = VlmOcr(provider="openrouter", model="primary", fallback_model="fallback")
    calls = []

    class FakeSession:
        def get(self, *args, **kwargs):
            calls.append("auth")
            return SimpleNamespace(status_code=401, text='{"error":{"message":"User not found."}}')

        def post(self, *args, **kwargs):
            calls.append(kwargs["json"]["model"])
            return SimpleNamespace(status_code=401, text='{"error":{"message":"User not found."}}')

    monkeypatch.setattr(ocr, "_get_session", lambda: FakeSession())
    crop = Image.new("RGB", (8, 8), "white")

    assert ocr.check_auth("bad") is False
    assert ocr.auth_failed is True
    assert ocr.read_crop(crop, api_key="bad") is None
    assert calls == ["auth"]


def test_translator_auth_error_returns_without_retry(monkeypatch):
    translator = Translator(model="primary")
    calls = []

    class FakeSession:
        def get(self, *args, **kwargs):
            calls.append("auth")
            return SimpleNamespace(status_code=401, text='{"error":{"message":"User not found."}}')

        def post(self, *args, **kwargs):
            calls.append(kwargs["json"]["model"])
            return SimpleNamespace(status_code=401, text='{"error":{"message":"User not found."}}')

    monkeypatch.setattr(translator, "_get_session", lambda: FakeSession())

    assert translator.check_auth("bad") is False
    assert translator.auth_failed is True
    assert translator.translate_batch([{"id": "b001", "text": "こんにちは"}], api_key="bad") is None
    assert calls == ["auth"]


def test_plan_korean_lines_wraps_long_text_by_bbox():
    config = {"typesetting": {"target_font_size": 40, "avg_char_width": 0.85, "line_height": 50, "padding": 20}}
    planned = main._plan_korean_lines("이건좀긴대사라서여러줄로나눠야함", [10, 10, 260, 180], config)
    assert "\n" in planned
    assert len(planned.split("\n")) <= 3

    planned = main._plan_korean_lines("이건 좀 긴 대사라서 여러 줄로 나눠야 함", [10, 10, 320, 180], config)
    assert "이건 좀" in planned or "이건" in planned
    assert "긴 대사" in planned or "긴" in planned
    assert len(planned.split("\n")) <= 3

    config["typesetting"]["max_chars_per_line"] = 6
    planned = main._plan_korean_lines("무한히 잘해주고 그럼에도 대가도 요구하지 않아", [10, 10, 280, 320], config)
    assert all(len(line) <= 6 for line in planned.split("\n"))
    assert "무한히" in planned
    assert "잘해주고" in planned


def test_plan_korean_lines_preserves_explicit_breaks():
    config = {"typesetting": {}}
    planned = main._plan_korean_lines("첫줄\n둘째줄", [10, 10, 300, 220], config)
    assert planned == "첫줄\n둘째줄"


def test_plan_korean_lines_does_not_orphan_punctuation():
    config = {"typesetting": {"max_chars_per_line": 4}}

    planned = main._plan_korean_lines("효율이 안 좋은걸!!", [0, 0, 260, 390], config)

    assert planned.splitlines()[-1] == "좋은걸!!"
    assert "\n!" not in planned

    planned = main._plan_korean_lines("비효율적인걸!!", [0, 0, 260, 390], config)
    assert planned.splitlines()[-1] == "인걸!!"
    assert "\n!" not in planned


def test_plan_vertical_text_splits_into_columns():
    config = {
        "typesetting": {
            "target_font_size": 42,
            "avg_char_width": 0.85,
            "line_height": 52,
            "padding": 20,
        }
    }
    planned = main._plan_vertical_text("청춘은변했어모험소녀와미답의대지", [0, 0, 120, 600], config)
    assert "\n" in planned
    assert all(len(col) <= 11 for col in planned.split("\n"))


def test_add_left_margin_vertical_text_mask_adds_narrow_left_columns():
    image = Image.new("L", (100, 180), 220)
    arr = np.asarray(image).copy()
    arr[20:140, 10:16] = 40
    result = {
        "text_mask": np.zeros((180, 100), dtype=np.uint8),
        "floating_text_mask": np.zeros((180, 100), dtype=np.uint8),
    }
    main._add_left_margin_vertical_text_mask(Image.fromarray(arr), result, {"inpainting": {}})
    assert not np.any(result["text_mask"] > 128)
    assert np.any(result["floating_text_mask"][20:140, 10:16] > 128)


def test_add_threshold_text_mask_adds_dark_pixels_inside_bubbles():
    image = Image.new("L", (20, 20), 255)
    arr = np.asarray(image).copy()
    arr[5:15, 8:10] = 100
    result = {
        "text_mask": np.zeros((20, 20), dtype=np.uint8),
        "bubbles": [{"bbox": [0, 0, 20, 20]}],
    }
    main._add_threshold_text_mask(Image.fromarray(arr), result, {"inpainting": {}})
    assert np.any(result["text_mask"][5:15, 8:10] > 128)


def test_fill_threshold_text_components_removes_tiny_text_remnants():
    base = Image.new("L", (40, 40), 180)
    arr = np.asarray(base).copy()
    arr[15:25, 15:17] = 40
    original = Image.fromarray(arr, mode="L")
    result = Image.new("RGB", (40, 40), (180, 180, 180))
    detection = {
        "bubbles": [{"bbox": [10, 10, 30, 30]}],
        "texts": [],
    }
    main._fill_threshold_text_components(result, original, detection, {"inpainting": {}})
    sample = np.asarray(result)[20, 16]
    assert 160 <= sample[0] <= 200


def test_add_threshold_text_mask_prefers_precise_text_bboxes():
    image = Image.new("L", (40, 40), 255)
    arr = np.asarray(image).copy()
    arr[5:15, 8:10] = 40
    arr[20:30, 20:22] = 40
    result = {
        "text_mask": np.zeros((40, 40), dtype=np.uint8),
        "bubbles": [{"bbox": [0, 0, 40, 40]}],
        "texts": [{"bbox": [5, 5, 15, 15]}],
    }
    main._add_threshold_text_mask(Image.fromarray(arr), result, {"inpainting": {"threshold_global_mask": False}})
    assert np.any(result["text_mask"][5:15, 8:10] > 128)
    assert not np.any(result["text_mask"][20:30, 20:22] > 128)


def test_add_threshold_text_mask_does_not_add_floating_text_to_bubble_mask():
    image = Image.new("L", (40, 40), 255)
    arr = np.asarray(image).copy()
    arr[20:30, 20:22] = 40
    result = {
        "text_mask": np.zeros((40, 40), dtype=np.uint8),
        "bubbles": [{"bbox": [0, 0, 10, 10]}],
        "texts": [{"bbox": [18, 18, 25, 32], "in_bubble": False}],
    }

    main._add_threshold_text_mask(
        Image.fromarray(arr), result, {"inpainting": {"threshold_global_mask": False}}
    )

    assert not np.any(result["text_mask"][18:32, 18:25] > 128)


def test_clip_text_mask_to_text_bboxes_protects_background():
    mask = Image.new("L", (40, 40), 0)
    mask.paste(255, (0, 0, 10, 10))
    mask.paste(255, (15, 15, 25, 25))
    clipped = main._clip_text_mask_to_text_bboxes(mask, {"texts": [{"bbox": [13, 13, 27, 27]}]}, {"inpainting": {"keep_external_text_components": False}})
    arr = np.asarray(clipped)
    assert arr[5, 5] == 0
    assert arr[20, 20] == 255


def test_clip_text_mask_to_text_bboxes_keeps_external_text_components():
    mask = Image.new("L", (40, 40), 0)
    mask.paste(255, (15, 15, 25, 25))
    clipped = main._clip_text_mask_to_text_bboxes(mask, {"texts": [{"bbox": [5, 5, 10, 10]}]}, {"inpainting": {}})
    arr = np.asarray(clipped)
    assert arr[20, 20] == 255


def test_clip_text_mask_to_bubbles_protects_background():
    original = Image.new("L", (40, 40), 255)
    mask = Image.new("L", (40, 40), 0)
    mask.paste(255, (0, 0, 10, 10))
    mask.paste(255, (15, 15, 25, 25))
    clipped = main._clip_text_mask_to_bubbles(mask, [{"bbox": [10, 10, 30, 30]}], original)
    arr = np.asarray(clipped)
    assert arr[5, 5] == 0
    assert arr[20, 20] == 255


def test_plan_vertical_text_preserves_space_separated_words():
    planned = main._plan_vertical_text("인류의 생존 경쟁에", [0, 0, 180, 420], {"typesetting": {}})
    assert "".join(planned.split("\n")) == "인류의생존경쟁에"


def test_vlm_ocr_clean_rejects_placeholder_only():
    assert VlmOcr._clean("〇〇〇…") == ""
    assert VlmOcr._clean("第1話 青春は変わった。") != ""


def test_vlm_ocr_clean_preserves_intentional_reaction_punctuation():
    assert VlmOcr._clean("…") == "…"
    assert VlmOcr._clean("!?") == "!?"


def test_run_inpainting_opencv_fallback_uses_remaining_mask(monkeypatch, tmp_path):
    detection = {
        "page_id": "page",
        "text_mask": np.zeros((10, 10), dtype=np.uint8),
        "bubbles": [],
    }
    detection["text_mask"][3:7, 3:7] = 255
    original = Image.new("RGB", (10, 10), "white")
    config = {
        "inpainting": {
            "flat_fill": False,
            "lama_fallback": False,
            "opencv_fallback": True,
            "opencv_radius": 3,
            "opencv_dilate": 2,
        },
        "comfyui": {"base_url": "http://127.0.0.1:8188"},
    }
    seen = {}

    def fake_opencv_inpaint(image, mask, radius=3, dilate_iterations=2):
        seen["mask"] = np.array(mask)
        seen["radius"] = radius
        seen["dilate_iterations"] = dilate_iterations
        return image.copy()

    monkeypatch.setattr("comfy_client.opencv_inpaint", fake_opencv_inpaint)

    out = main.run_inpainting("unused.jpg", detection, config, tmp_dir=tmp_path, original=original)

    assert out.exists()
    assert np.any(seen["mask"] > 128)
    assert seen["radius"] == 3
    assert seen["dilate_iterations"] == 2


def test_run_ocr_caches_no_text_result(monkeypatch, tmp_path):
    main._OCR_CLIENT_CACHE.clear()
    detection = {"page_id": "page", "bubbles": [{"id": "b001", "bbox": [0, 0, 4, 4]}]}
    original = Image.new("RGB", (8, 8), "white")
    config = {
        "ocr": {"provider": "openrouter", "model": "demo-model", "crop_upscale": 1, "max_workers": 1},
        "output": {"save_crops": False},
    }
    saved = {}

    class FakeCache:
        def hash_image(self, image_bytes):
            return "hash1"

        def get(self, key):
            return None

        def set(self, key, value):
            saved[key] = value

    class FakeOcr:
        def __init__(self, provider, model, fallback_model=None):
            pass

        def read_crop(self, crop, api_key="", orientation=None):
            return None

    monkeypatch.setattr("ocr.vlm_ocr.VlmOcr", FakeOcr)

    main.run_ocr(detection, original, config, api_key="key", tmp_dir=tmp_path, cache=FakeCache())

    assert list(saved.values()) == ["[NO TEXT]"]
    assert next(iter(saved)).startswith("ocr_v6_")
    assert detection["bubbles"][0]["ocr_text"] is None


def test_run_ocr_uses_cached_no_text_without_api_call(monkeypatch, tmp_path):
    main._OCR_CLIENT_CACHE.clear()
    detection = {"page_id": "page", "bubbles": [{"id": "b001", "bbox": [0, 0, 4, 4]}]}
    original = Image.new("RGB", (8, 8), "white")
    config = {
        "ocr": {"provider": "openrouter", "model": "demo-model", "crop_upscale": 1, "max_workers": 1},
        "output": {"save_crops": False},
    }

    class FakeCache:
        def hash_image(self, image_bytes):
            return "hash1"

        def get(self, key):
            return "[NO TEXT]"

        def set(self, key, value):
            raise AssertionError("cached no-text path should not rewrite cache")

    class FakeOcr:
        def __init__(self, provider, model, fallback_model=None):
            pass

        def read_crop(self, crop, api_key="", orientation=None):
            raise AssertionError("cached no-text path should not call OCR API")

    monkeypatch.setattr("ocr.vlm_ocr.VlmOcr", FakeOcr)

    main.run_ocr(detection, original, config, api_key="key", tmp_dir=tmp_path, cache=FakeCache())

    assert detection["bubbles"][0]["ocr_text"] is None


def test_run_ocr_deduplicates_identical_crops_within_page(monkeypatch, tmp_path):
    main._OCR_CLIENT_CACHE.clear()
    detection = {
        "page_id": "page",
        "bubbles": [
            {"id": "b001", "bbox": [0, 0, 4, 4]},
            {"id": "b002", "bbox": [0, 0, 4, 4]},
        ],
    }
    original = Image.new("RGB", (8, 8), "white")
    config = {
        "ocr": {"provider": "openrouter", "model": "demo-model", "crop_upscale": 1, "max_workers": 2},
        "output": {"save_crops": False},
    }
    calls = {"count": 0}

    class FakeCache:
        def hash_image(self, image_bytes):
            return "samehash"

        def get(self, key):
            return None

        def set(self, key, value):
            pass

    class FakeOcr:
        def __init__(self, provider, model, fallback_model=None):
            pass

        def read_crop(self, crop, api_key="", orientation=None):
            calls["count"] += 1
            return "텍스트"

    monkeypatch.setattr("ocr.vlm_ocr.VlmOcr", FakeOcr)

    main.run_ocr(detection, original, config, api_key="key", tmp_dir=tmp_path, cache=FakeCache())

    assert calls["count"] == 1
    assert detection["bubbles"][0]["ocr_text"] == "텍스트"
    assert detection["bubbles"][1]["ocr_text"] == "텍스트"


def test_run_ocr_crops_text_mask_inside_bubble(monkeypatch, tmp_path):
    main._OCR_CLIENT_CACHE.clear()
    detection = {
        "page_id": "page",
        "bubbles": [{"id": "b001", "bbox": [0, 0, 20, 20]}],
        "text_mask": np.zeros((20, 20), dtype=np.uint8),
    }
    detection["text_mask"][5:10, 5:10] = 255
    original = Image.new("RGB", (20, 20), "white")
    config = {
        "ocr": {"provider": "openrouter", "model": "demo-model", "crop_upscale": 1, "text_mask_margin": 4, "max_workers": 1},
        "output": {"save_crops": False},
    }
    seen_crops = []

    class FakeCache:
        def hash_image(self, image_bytes):
            return "hash1"

        def get(self, key):
            return None

        def set(self, key, value):
            pass

    class FakeOcr:
        def __init__(self, provider, model, fallback_model=None):
            pass

        def read_crop(self, crop, api_key="", orientation=None):
            seen_crops.append((crop.size, orientation))
            return "あ"

    monkeypatch.setattr("ocr.vlm_ocr.VlmOcr", FakeOcr)

    main.run_ocr(detection, original, config, api_key="key", tmp_dir=tmp_path, cache=FakeCache())

    assert seen_crops == [((13, 13), "mixed")]
    assert detection["bubbles"][0]["ocr_text"] == "あ"


def test_run_ocr_applies_configured_upscale_once(monkeypatch, tmp_path):
    main._OCR_CLIENT_CACHE.clear()
    detection = {"page_id": "page", "bubbles": [{"id": "b001", "bbox": [0, 0, 4, 4]}]}
    original = Image.new("RGB", (8, 8), "white")
    config = {
        "ocr": {"provider": "openrouter", "model": "demo-model", "crop_upscale": 2, "max_workers": 1},
        "output": {"save_crops": False},
    }
    seen_sizes = []

    class FakeCache:
        def hash_image(self, image_bytes):
            return "hash1"

        def get(self, key):
            return None

        def set(self, key, value):
            pass

    class FakeOcr:
        def __init__(self, provider, model, fallback_model=None):
            pass

        def read_crop(self, crop, api_key="", orientation=None):
            seen_sizes.append(crop.size)
            return "あ"

    monkeypatch.setattr("ocr.vlm_ocr.VlmOcr", FakeOcr)

    main.run_ocr(detection, original, config, api_key="key", tmp_dir=tmp_path, cache=FakeCache())

    assert seen_sizes == [(8, 8)]


def test_run_ocr_accepts_only_agreeing_tight_and_context_views(monkeypatch, tmp_path):
    main._OCR_CLIENT_CACHE.clear()
    text_mask = np.zeros((20, 20), dtype=np.uint8)
    text_mask[5:10, 5:10] = 255
    detection = {
        "page_id": "page",
        "bubbles": [{"id": "b001", "bbox": [0, 0, 20, 20]}],
        "text_mask": text_mask,
    }
    config = {
        "ocr": {
            "provider": "openrouter",
            "model": "demo-model",
            "crop_upscale": 1,
            "text_mask_margin": 4,
            "multi_view": True,
            "max_workers": 2,
        },
        "output": {"save_crops": False},
    }
    seen_sizes = []

    class FakeCache:
        def hash_image(self, image_bytes):
            return hashlib.md5(image_bytes).hexdigest()

        def get(self, key):
            return None

        def set(self, key, value):
            pass

    class FakeOcr:
        def __init__(self, provider, model, fallback_model=None):
            pass

        def read_crop(self, crop, api_key="", orientation=None):
            seen_sizes.append(crop.size)
            return "ちゃんと同意を取ってよ!!"

    monkeypatch.setattr("ocr.vlm_ocr.VlmOcr", FakeOcr)

    main.run_ocr(
        detection,
        Image.new("RGB", (20, 20), "white"),
        config,
        api_key="key",
        tmp_dir=tmp_path,
        cache=FakeCache(),
    )

    assert sorted(seen_sizes) == [(13, 13), (20, 20)]
    assert detection["bubbles"][0]["ocr_text"] == "ちゃんと同意を取ってよ!!"
    assert detection["bubbles"][0]["ocr_status"] == "accepted"


def test_run_translation_uses_previous_context_in_cache_key_and_request(monkeypatch):
    main._TRANSLATOR_CACHE.clear()
    detection = {"bubbles": [{"id": "b001", "ocr_text": "日本語"}]}
    config = {"translation": {"provider": "openrouter", "model": "demo-model"}}
    previous_context = ["이전 대사"]

    texts_to_translate = [{"id": "b001", "text": "日本語"}]
    text_only_hash = hashlib.md5(
        json.dumps(texts_to_translate, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    class FakeCache:
        def __init__(self):
            stale = json.dumps(
                [{"id": "b001", "translation": "오래된 번역", "type": "dialogue"}],
                ensure_ascii=False,
            )
            self.data = {f"trans_{text_only_hash}": stale}

        def hash_text(self, text):
            return hashlib.md5(text.encode("utf-8")).hexdigest()

        def get(self, key):
            return self.data.get(key)

        def set(self, key, value):
            self.data[key] = value

    class FakeTranslator:
        called_with_context = None

        def __init__(self, provider, model):
            self.provider = provider
            self.model = model

        def translate_batch(self, texts, api_key="", previous_context=None):
            FakeTranslator.called_with_context = previous_context
            return [{"id": "b001", "translation": "새 번역", "type": "thought"}]

    monkeypatch.setattr("translation.translator.Translator", FakeTranslator)

    main.run_translation(
        detection,
        config,
        api_key="key",
        cache=FakeCache(),
        previous_context=previous_context,
    )

    assert FakeTranslator.called_with_context == previous_context
    assert detection["bubbles"][0]["translation"] == "새 번역"
    assert detection["bubbles"][0]["text_type"] == "thought"


def test_run_translation_uses_versioned_cache_key(monkeypatch):
    main._TRANSLATOR_CACHE.clear()
    detection = {"bubbles": [{"id": "b001", "ocr_text": "日本語"}]}
    config = {"translation": {"provider": "openrouter", "model": "demo-model"}}
    seen_keys = []

    class FakeCache:
        def hash_text(self, text):
            return hashlib.md5(text.encode("utf-8")).hexdigest()

        def get(self, key):
            seen_keys.append(key)
            return None

        def set(self, key, value):
            seen_keys.append(key)

    class FakeTranslator:
        def __init__(self, provider, model):
            self.provider = provider
            self.model = model

        def translate_batch(self, texts, api_key="", previous_context=None):
            return [{"id": "b001", "translation": "번역", "type": "dialogue"}]

    monkeypatch.setattr("translation.translator.Translator", FakeTranslator)

    main.run_translation(detection, config, api_key="key", cache=FakeCache())

    assert all(key.startswith("trans_v15_") for key in seen_keys)


def test_run_ocr_uses_versioned_cache_key(monkeypatch, tmp_path):
    main._OCR_CLIENT_CACHE.clear()
    detection = {"page_id": "page", "bubbles": [{"id": "b001", "bbox": [0, 0, 4, 4]}]}
    original = Image.new("RGB", (8, 8), "white")
    config = {
        "ocr": {"provider": "openrouter", "model": "demo-model", "crop_upscale": 1, "max_workers": 1},
        "output": {"save_crops": False},
    }
    seen_keys = []

    class FakeCache:
        def hash_image(self, image_bytes):
            return "hash1"

        def get(self, key):
            seen_keys.append(key)
            return None

        def set(self, key, value):
            seen_keys.append(key)

    class FakeOcr:
        def __init__(self, provider, model, fallback_model=None):
            pass

        def read_crop(self, crop, api_key="", orientation=None):
            return None

    monkeypatch.setattr("ocr.vlm_ocr.VlmOcr", FakeOcr)

    main.run_ocr(detection, original, config, api_key="key", tmp_dir=tmp_path, cache=FakeCache())

    assert all(key.startswith("ocr_v6_") for key in seen_keys)


def test_run_translation_sanitizes_cached_translations(monkeypatch):
    main._TRANSLATOR_CACHE.clear()
    detection = {"bubbles": [{"id": "b001", "ocr_text": "日本語"}]}
    config = {"translation": {"provider": "openrouter", "model": "demo-model"}}

    class FakeCache:
        def hash_text(self, text):
            return hashlib.md5(text.encode("utf-8")).hexdigest()

        def get(self, key):
            return json.dumps(
                [{
                    "id": "b001",
                    "translation": "어?!\n\nContext (주변 대화):\n이전 대사",
                    "type": "dialogue",
                }],
                ensure_ascii=False,
            )

        def set(self, key, value):
            raise AssertionError("cached path should not rewrite")

    class FakeTranslator:
        _sanitize_batch_result = staticmethod(Translator._sanitize_batch_result)

        def __init__(self, provider, model):
            raise AssertionError("cached path should not call translator")

    monkeypatch.setattr("translation.translator.Translator", FakeTranslator)

    main.run_translation(detection, config, api_key="key", cache=FakeCache())

    assert detection["bubbles"][0]["translation"] == "어?!"


def test_vlm_ocr_clean_drops_no_text_markers_from_mixed_output():
    cleaned = VlmOcr._clean("は!?\n[NO TEXT]\n.....")
    assert cleaned == "は!?"


def test_vlm_ocr_clean_drops_english_prompt_leak():
    cleaned = VlmOcr._clean(
        "The image shows a browser tab and there is no visible text.\n[NO TEXT]"
    )
    assert cleaned == ""


def test_translator_sanitize_translation_strips_context_leak():
    item = {
        "id": "b001",
        "translation": "어?!\n\nContext (주변 대화):\n이전 대사",
        "type": "dialogue",
    }

    sanitized = Translator._sanitize_translation_item(item)

    assert sanitized["translation"] == "어?!"


def test_translation_prompt_has_one_authoritative_korean_name_policy():
    from translation.translator import Translator

    prompt = Translator()._build_system_prompt()

    assert "みゆき stays みゆき" not in prompt
    assert "Transliterate Japanese proper names into Hangul" in prompt
    assert "sentence-final particles by their function" in prompt


def test_run_translation_uses_editor_review_when_enabled(monkeypatch):
    main._TRANSLATOR_CACHE.clear()
    detection = {"bubbles": [{"id": "b001", "ocr_text": "もう付き合ってる人がいるの"}]}
    config = {
        "translation": {
            "provider": "openrouter",
            "model": "demo-model",
            "quality_review": True,
        }
    }

    class FakeCache:
        def hash_text(self, text):
            return hashlib.md5(text.encode("utf-8")).hexdigest()

        def get(self, key):
            return None

        def set(self, key, value):
            pass

    class FakeTranslator:
        def __init__(self, provider, model):
            pass

        def translate_batch(self, texts, api_key="", previous_context=None):
            return [{"id": "b001", "translation": "이미 교제하는 사람이 있어", "type": "dialogue"}]

        def review_batch(self, texts, drafts, api_key="", previous_context=None):
            assert drafts[0]["translation"] == "이미 교제하는 사람이 있어"
            return [{"id": "b001", "translation": "나 이미 사귀는 사람 있어", "type": "dialogue"}]

    monkeypatch.setattr("translation.translator.Translator", FakeTranslator)

    main.run_translation(detection, config, api_key="key", cache=FakeCache())

    assert detection["bubbles"][0]["translation"] == "나 이미 사귀는 사람 있어"


def test_run_translation_retries_editor_output_that_loses_emphatic_ending(monkeypatch):
    main._TRANSLATOR_CACHE.clear()
    source = "これだから腐れ縁はイヤなんでござる！拙者の恥ずかしい過去何つでも知ってる！！"
    detection = {"bubbles": [{"id": "b001", "ocr_text": source}]}
    config = {
        "translation": {
            "provider": "openrouter",
            "model": "demo-model",
            "quality_review": True,
        }
    }
    review_calls = []

    class FakeCache:
        def hash_text(self, text):
            return hashlib.md5(text.encode("utf-8")).hexdigest()

        def get(self, key):
            return None

        def set(self, key, value):
            pass

    class FakeTranslator:
        def __init__(self, provider, model):
            pass

        def translate_batch(self, texts, api_key="", previous_context=None):
            return [{
                "id": "b001",
                "translation": "이래서 지긋지긋한 인연은 싫다니까! 내 부끄러운 과거를 뭐든 알고 있어!!",
                "type": "dialogue",
            }]

        def review_batch(self, texts, drafts, api_key="", previous_context=None):
            review_calls.append(len(texts))
            if len(review_calls) == 1:
                return [{
                    "id": "b001",
                    "translation": "이래서 지긋지긋한 인연은 싫은 것이오! 소생의 부끄러운 과거 따윈 얼마든",
                    "type": "dialogue",
                }]
            return [{
                "id": "b001",
                "translation": "이래서 지긋지긋한 인연은 싫은 것이오! 소생의 부끄러운 과거는 뭐든 알고 있소!!",
                "type": "dialogue",
            }]

    monkeypatch.setattr("translation.translator.Translator", FakeTranslator)

    main.run_translation(detection, config, api_key="key", cache=FakeCache())

    bubble = detection["bubbles"][0]
    assert review_calls == [1, 1]
    assert bubble["translation"].endswith("알고 있소!!")
    assert bubble["translation_status"] == "accepted"


def test_run_translation_retries_malformed_pseudo_archaic_editor_ending(monkeypatch):
    main._TRANSLATOR_CACHE.clear()
    source = "これだから腐れ縁はイヤなんでござる！！"
    detection = {"bubbles": [{"id": "b001", "ocr_text": source}]}
    config = {
        "translation": {
            "provider": "openrouter",
            "model": "demo-model",
            "quality_review": True,
        }
    }
    review_calls = []

    class FakeCache:
        def hash_text(self, text):
            return hashlib.md5(text.encode("utf-8")).hexdigest()

        def get(self, key):
            return None

        def set(self, key, value):
            pass

    class FakeTranslator:
        def __init__(self, provider, model):
            pass

        def translate_batch(self, texts, api_key="", previous_context=None):
            return [{"id": "b001", "translation": "이래서 악연은 싫다오!!", "type": "dialogue"}]

        def review_batch(self, texts, drafts, api_key="", previous_context=None):
            review_calls.append(len(texts))
            ending = "싫소이다!!" if len(review_calls) == 1 else "싫은 것이오!!"
            return [{"id": "b001", "translation": f"이래서 악연은 {ending}", "type": "dialogue"}]

    monkeypatch.setattr("translation.translator.Translator", FakeTranslator)

    main.run_translation(detection, config, api_key="key", cache=FakeCache())

    bubble = detection["bubbles"][0]
    assert review_calls == [1, 1]
    assert bubble["translation"] == "이래서 악연은 싫은 것이오!!"
    assert bubble["translation_status"] == "accepted"


def test_run_translation_preserves_punctuation_only_region_without_model_call(monkeypatch):
    main._TRANSLATOR_CACHE.clear()
    detection = {"bubbles": [{"id": "b001", "ocr_text": "…"}]}
    config = {"translation": {"provider": "openrouter", "model": "demo-model"}}

    class FakeCache:
        def hash_text(self, text):
            raise AssertionError("punctuation-only regions must not be cached for translation")

    class FakeTranslator:
        def __init__(self, provider, model):
            raise AssertionError("punctuation-only regions must not call a model")

    monkeypatch.setattr("translation.translator.Translator", FakeTranslator)

    main.run_translation(detection, config, api_key="key", cache=FakeCache())

    bubble = detection["bubbles"][0]
    assert bubble["translation"] is None
    assert bubble["translation_status"] == "preserved"
    assert bubble["skip_inpaint"] is True


def test_cache_manager_can_batch_writes_until_flush(tmp_path):
    cache_file = tmp_path / "cache.json"
    cache = CacheManager(cache_file, autosave=False)

    cache.set("a", "1")
    cache.set("b", "2")

    assert not cache_file.exists()

    cache.flush()

    assert json.loads(cache_file.read_text(encoding="utf-8")) == {"a": "1", "b": "2"}


def test_translator_reuses_session_instance():
    translator = Translator()
    assert translator._get_session() is translator._get_session()


def test_vlm_ocr_reuses_session_instance_per_thread():
    ocr = VlmOcr()
    assert ocr._get_session() is ocr._get_session()


def test_run_translation_deduplicates_identical_ocr_texts(monkeypatch):
    main._TRANSLATOR_CACHE.clear()
    detection = {
        "bubbles": [
            {"id": "b001", "ocr_text": "同じ台詞"},
            {"id": "b002", "ocr_text": "同じ台詞"},
            {"id": "b003", "ocr_text": "別の台詞"},
        ]
    }
    config = {"translation": {"provider": "openrouter", "model": "demo-model"}}

    class FakeCache:
        def hash_text(self, text):
            return hashlib.md5(text.encode("utf-8")).hexdigest()

        def get(self, key):
            return None

        def set(self, key, value):
            pass

    class FakeTranslator:
        payloads = []

        def __init__(self, provider, model):
            self.provider = provider
            self.model = model

        def translate_batch(self, texts, api_key="", previous_context=None):
            FakeTranslator.payloads.append(texts)
            return [
                {"id": "b001", "translation": "같은 대사", "type": "dialogue"},
                {"id": "b003", "translation": "다른 대사", "type": "dialogue"},
            ]

    monkeypatch.setattr("translation.translator.Translator", FakeTranslator)

    main.run_translation(detection, config, api_key="key", cache=FakeCache())

    assert FakeTranslator.payloads == [[
        {"id": "b001", "text": "同じ台詞"},
        {"id": "b003", "text": "別の台詞"},
    ]]
    assert detection["bubbles"][0]["translation"] == "같은 대사"
    assert detection["bubbles"][1]["translation"] == "같은 대사"
    assert detection["bubbles"][2]["translation"] == "다른 대사"


def test_run_translation_uses_manga_reading_order_in_one_page_batch(monkeypatch):
    main._TRANSLATOR_CACHE.clear()
    detection = {
        "bubbles": [
            {"id": "left", "bbox": [10, 10, 30, 40], "ocr_text": "左の台詞"},
            {"id": "bottom", "bbox": [70, 80, 90, 110], "ocr_text": "下の台詞"},
            {"id": "right", "bbox": [70, 10, 90, 40], "ocr_text": "右の台詞"},
        ]
    }
    config = {
        "translation": {
            "provider": "openrouter",
            "model": "demo-model",
            "batch_max_items": 12,
        }
    }

    class FakeCache:
        def hash_text(self, text):
            return hashlib.md5(text.encode("utf-8")).hexdigest()

        def get(self, key):
            return None

        def set(self, key, value):
            pass

    class FakeTranslator:
        payloads = []

        def __init__(self, provider, model):
            pass

        def translate_batch(self, texts, api_key="", previous_context=None):
            FakeTranslator.payloads.append(texts)
            return [
                {"id": item["id"], "translation": f"번역 {item['id']}", "type": "dialogue"}
                for item in texts
            ]

    monkeypatch.setattr("translation.translator.Translator", FakeTranslator)

    main.run_translation(detection, config, api_key="key", cache=FakeCache())

    assert [[item["id"] for item in batch] for batch in FakeTranslator.payloads] == [
        ["right", "left", "bottom"]
    ]


def test_run_translation_deduplicates_texts_with_whitespace_variants(monkeypatch):
    main._TRANSLATOR_CACHE.clear()
    detection = {
        "bubbles": [
            {"id": "b001", "ocr_text": "同じ 台詞"},
            {"id": "b002", "ocr_text": "同じ\n台詞"},
        ]
    }
    config = {"translation": {"provider": "openrouter", "model": "demo-model"}}

    class FakeCache:
        def hash_text(self, text):
            return hashlib.md5(text.encode("utf-8")).hexdigest()

        def get(self, key):
            return None

        def set(self, key, value):
            pass

    class FakeTranslator:
        payloads = []

        def __init__(self, provider, model):
            self.provider = provider
            self.model = model

        def translate_batch(self, texts, api_key="", previous_context=None):
            FakeTranslator.payloads.append(texts)
            return [{"id": "b001", "translation": "같은 대사", "type": "dialogue"}]

    monkeypatch.setattr("translation.translator.Translator", FakeTranslator)

    main.run_translation(detection, config, api_key="key", cache=FakeCache())

    assert len(FakeTranslator.payloads[0]) == 1
    assert detection["bubbles"][0]["translation"] == "같은 대사"
    assert detection["bubbles"][1]["translation"] == "같은 대사"


def test_run_translation_trims_previous_context_before_request(monkeypatch):
    main._TRANSLATOR_CACHE.clear()
    detection = {"bubbles": [{"id": "b001", "ocr_text": "現在の台詞"}]}
    config = {
        "translation": {
            "provider": "openrouter",
            "model": "demo-model",
            "context_max_items": 2,
            "context_max_chars": 12,
        }
    }
    previous_context = ["첫 번째 긴 대사입니다", "둘째", "셋째 대사도 깁니다"]

    class FakeCache:
        def hash_text(self, text):
            return hashlib.md5(text.encode("utf-8")).hexdigest()

        def get(self, key):
            return None

        def set(self, key, value):
            pass

    class FakeTranslator:
        called_with_context = None

        def __init__(self, provider, model):
            self.provider = provider
            self.model = model

        def translate_batch(self, texts, api_key="", previous_context=None):
            FakeTranslator.called_with_context = previous_context
            return [{"id": "b001", "translation": "현재 대사", "type": "dialogue"}]

    monkeypatch.setattr("translation.translator.Translator", FakeTranslator)

    main.run_translation(
        detection,
        config,
        api_key="key",
        cache=FakeCache(),
        previous_context=previous_context,
    )

    assert FakeTranslator.called_with_context == ["둘째", "셋째 대사도 깁니다"]


def test_run_translation_deduplicates_normalized_previous_context(monkeypatch):
    main._TRANSLATOR_CACHE.clear()
    detection = {"bubbles": [{"id": "b001", "ocr_text": "現在の台詞"}]}
    config = {
        "translation": {
            "provider": "openrouter",
            "model": "demo-model",
            "context_max_items": 5,
            "context_max_chars": 50,
        }
    }
    previous_context = ["같은 대사", "같은   대사", "다른 대사", "같은\n대사"]

    class FakeCache:
        def hash_text(self, text):
            return hashlib.md5(text.encode("utf-8")).hexdigest()

        def get(self, key):
            return None

        def set(self, key, value):
            pass

    class FakeTranslator:
        called_with_context = None

        def __init__(self, provider, model):
            self.provider = provider
            self.model = model

        def translate_batch(self, texts, api_key="", previous_context=None):
            FakeTranslator.called_with_context = previous_context
            return [{"id": "b001", "translation": "현재 대사", "type": "dialogue"}]

    monkeypatch.setattr("translation.translator.Translator", FakeTranslator)

    main.run_translation(
        detection,
        config,
        api_key="key",
        cache=FakeCache(),
        previous_context=previous_context,
    )

    assert FakeTranslator.called_with_context == ["같은 대사", "다른 대사"]


def test_process_page_persists_final_ko_png_in_output_dir(tmp_path, monkeypatch):
    image_path = tmp_path / "page.png"
    Image.new("RGB", (8, 8), "white").save(image_path)

    config = {
        "output": {"base_dir": str(tmp_path / "out")},
        "inpainting": {},
    }

    monkeypatch.setattr(
        main,
        "run_detection",
        lambda image_path, config, tmp_dir: (
            {
                "page_id": "page",
                "width": 8,
                "height": 8,
                "bubbles": [{"id": "b001", "bbox": [0, 0, 4, 4], "translation": "번역", "text_type": "dialogue"}],
                "texts": [],
                "text_mask": None,
            },
            Image.open(image_path).convert("RGB"),
        ),
    )
    monkeypatch.setattr(main, "run_inpainting", lambda image_path, detection, config, tmp_dir, original=None: image_path)
    monkeypatch.setattr(main, "run_typesetting", lambda cleaned_path, detection, config, tmp_dir, page_id: Path(cleaned_path))

    result = main.process_page(str(image_path), config, api_key="")

    # v2: no result.json written — only _ko.png in output dir, result returned as dict
    assert result["page_id"] == "page"
    assert result["final_image"].endswith("page_ko.png")
    assert Path(result["final_image"]).exists()


def test_run_detection_opens_source_image_once(tmp_path, monkeypatch):
    image_path = tmp_path / "page.png"
    Image.new("RGB", (8, 8), "white").save(image_path)

    opened = {"count": 0}
    real_open = main.Image.open

    def counting_open(path, *args, **kwargs):
        if str(path) == str(image_path):
            opened["count"] += 1
        return real_open(path, *args, **kwargs)

    class FakeDetector:
        def __init__(self, **kwargs):
            pass

        def detect(self, image, page_id="000"):
            return {
                "page_id": page_id,
                "width": image.width,
                "height": image.height,
                "bubbles": [],
                "texts": [],
                "text_mask": None,
            }

    monkeypatch.setattr("main.Image.open", counting_open)
    monkeypatch.setattr("detection.yolo_detector.YoloDetector", FakeDetector)

    result, original = main.run_detection(
        str(image_path),
        {
            "output": {"debug_overlay": False, "save_masks": False},
            "models": {"text_segmenter": "seg.pt", "bubble_detector": "bubble.pt"},
            "yolo": {"conf_threshold": 0.25, "iou_threshold": 0.45, "device": "cpu"},
        },
        tmp_path,
    )

    assert opened["count"] == 1
    assert result["page_id"] == "page"
    assert original.size == (8, 8)


def test_yolo_detector_uses_configured_checkpoint_resolution():
    from detection.yolo_detector import YoloDetector

    calls = []

    class EmptyResult:
        boxes = None
        masks = None

    class FakeModel:
        names = {0: "text"}

        def __call__(self, image, **kwargs):
            calls.append(kwargs)
            return [EmptyResult()]

    detector = YoloDetector.__new__(YoloDetector)
    detector.text_model = FakeModel()
    detector.bubble_model = FakeModel()
    detector.conf = 0.2
    detector.iou = 0.55
    detector.device = "cpu"
    detector.image_size = 1280
    detector.text_classes = None

    detector.detect(Image.new("RGB", (1125, 1600), "white"), page_id="page")

    assert [call["imgsz"] for call in calls] == [1280, 1280]


def test_run_detection_reuses_detector_for_same_config(tmp_path, monkeypatch):
    image_path = tmp_path / "page.png"
    Image.new("RGB", (8, 8), "white").save(image_path)
    main._DETECTOR_CACHE.clear()

    created = {"count": 0}

    class FakeDetector:
        def __init__(self, **kwargs):
            created["count"] += 1

        def detect(self, image, page_id="000"):
            return {
                "page_id": page_id,
                "width": image.width,
                "height": image.height,
                "bubbles": [],
                "texts": [],
                "text_mask": None,
            }

    monkeypatch.setattr("detection.yolo_detector.YoloDetector", FakeDetector)

    config = {
        "output": {"debug_overlay": False, "save_masks": False},
        "models": {"text_segmenter": "seg-cache.pt", "bubble_detector": "bubble-cache.pt"},
        "yolo": {"conf_threshold": 0.25, "iou_threshold": 0.45, "device": "cpu"},
    }

    main.run_detection(str(image_path), config, tmp_path)
    main.run_detection(str(image_path), config, tmp_path)

    assert created["count"] == 1


def test_run_ocr_reuses_client_for_same_config(monkeypatch, tmp_path):
    detection = {"page_id": "page", "bubbles": [{"id": "b001", "bbox": [0, 0, 4, 4]}]}
    original = Image.new("RGB", (8, 8), "white")
    config = {
        "ocr": {"provider": "openrouter", "model": "demo-model", "fallback_model": "fallback", "crop_upscale": 1, "max_workers": 1},
        "output": {"save_crops": False},
    }
    main._OCR_CLIENT_CACHE.clear()
    created = {"count": 0}

    class FakeCache:
        def hash_image(self, image_bytes):
            return "hash1"

        def get(self, key):
            return "[NO TEXT]"

        def set(self, key, value):
            pass

    class FakeOcr:
        def __init__(self, provider, model, fallback_model=None):
            created["count"] += 1

        def read_crop(self, crop, api_key="", orientation=None):
            return None

    monkeypatch.setattr("ocr.vlm_ocr.VlmOcr", FakeOcr)

    main.run_ocr(detection, original, config, api_key="key", tmp_dir=tmp_path, cache=FakeCache())
    main.run_ocr(detection, original, config, api_key="key", tmp_dir=tmp_path, cache=FakeCache())

    assert created["count"] == 1


def test_run_translation_reuses_client_for_same_config(monkeypatch):
    detection = {"bubbles": [{"id": "b001", "ocr_text": "日本語"}]}
    config = {"translation": {"provider": "openrouter", "model": "demo-model"}}
    main._TRANSLATOR_CACHE.clear()
    created = {"count": 0}

    class FakeCache:
        def hash_text(self, text):
            return hashlib.md5(text.encode("utf-8")).hexdigest()

        def get(self, key):
            return None

        def set(self, key, value):
            pass

    class FakeTranslator:
        def __init__(self, provider, model):
            created["count"] += 1

        def translate_batch(self, texts, api_key="", previous_context=None):
            return [{"id": "b001", "translation": "번역", "type": "dialogue"}]

    monkeypatch.setattr("translation.translator.Translator", FakeTranslator)

    main.run_translation(detection, config, api_key="key", cache=FakeCache())
    main.run_translation(detection, config, api_key="key", cache=FakeCache())

    assert created["count"] == 1


def test_load_vlm_json_list(tmp_path):
    path = tmp_path / "vlm.json"
    path.write_text('[{"id":"b001","text":"テスト"}]', encoding="utf-8")

    assert _load_vlm_json(path) == {"b001": "テスト"}


def test_pp_ocrv6_match_assigns_text_inside_bubble():
    detection = {
        "page_id": "page",
        "width": 100,
        "height": 100,
        "bubbles": [{"id": "b001", "bbox": [10, 10, 40, 50], "ocr_text": "テスト"}],
    }
    pp_rows = [
        {"text": "あ", "score": 0.9, "poly": [[15, 15], [25, 15], [25, 22], [15, 22]]},
    ]

    result = match_pp_ocr_to_bubbles(detection, pp_rows, score_threshold=0.5)

    assert result["pp_rows_total"] == 1
    assert result["matches"]["b001"]["pp"] == "あ"
    assert result["matches"]["b001"]["vlm"] == "テスト"
    assert result["pp_rows_filtered"][0]["bubble_id"] == "b001"


def test_pp_ocrv6_match_filters_low_match_score_rows():
    detection = {
        "page_id": "page",
        "width": 100,
        "height": 100,
        "bubbles": [{"id": "b001", "bbox": [10, 10, 40, 50]}],
    }
    pp_rows = [
        {"text": "あ", "score": 0.9, "poly": [[15, 15], [25, 15], [25, 22], [15, 22]]},
    ]

    result = match_pp_ocr_to_bubbles(detection, pp_rows, score_threshold=0.5, match_score_threshold=3.0)

    assert result["pp_rows_filtered"] == []
    assert result["matches"]["b001"]["pp"] == ""


def test_pp_ocrv6_match_filters_low_score_rows():
    detection = {
        "page_id": "page",
        "width": 100,
        "height": 100,
        "bubbles": [{"id": "b001", "bbox": [10, 10, 40, 50]}],
    }
    pp_rows = [
        {"text": "あ", "score": 0.4, "poly": [[15, 15], [25, 15], [25, 22], [15, 22]]},
    ]

    result = match_pp_ocr_to_bubbles(detection, pp_rows, score_threshold=0.5)

    assert result["pp_rows_total"] == 1
    assert result["pp_rows_filtered"] == []
    assert result["matches"]["b001"]["pp"] == ""


def test_opencv_inpaint_changes_masked_region():
    image = Image.new("RGB", (32, 32), (250, 250, 250))
    mask = Image.new("L", (32, 32), 0)
    for y in range(10, 22):
        for x in range(10, 22):
            mask.putpixel((x, y), 255)

    result = opencv_inpaint(image, mask, radius=3)

    assert result.size == (32, 32)
    assert result.getpixel((16, 16)) != (250, 250, 250)


def test_flat_fill_white_bubble_removes_remaining_mask():
    image = Image.new("RGB", (40, 40), (255, 255, 255))
    mask = Image.new("L", (40, 40), 0)
    for y in range(10, 30):
        for x in range(10, 30):
            mask.putpixel((x, y), 255)
    bubbles = [{"bbox": [0, 0, 40, 40]}]

    result, remaining = flat_fill(image, mask, bubbles=bubbles)

    assert result.getpixel((15, 15)) == (255, 255, 255)
    assert remaining is None


def test_flat_fill_preserves_pixels_outside_glyph_mask_inside_text_bbox():
    image_arr = np.full((40, 40, 3), 245, dtype=np.uint8)
    image_arr[12:28, 12:28] = 230
    image = Image.fromarray(image_arr)
    mask = Image.new("L", (40, 40), 0)
    mask.paste(255, (18, 18, 22, 22))
    bubbles = [{"bbox": [0, 0, 40, 40]}]
    texts = [{"bbox": [12, 12, 28, 28]}]

    result, _ = flat_fill(image, mask, bubbles=bubbles, texts=texts)

    assert result.getpixel((13, 13)) == image.getpixel((13, 13))


def test_flat_fill_routes_high_variance_screentone_to_inpainting():
    image_arr = np.full((40, 40, 3), 245, dtype=np.uint8)
    image_arr[::3, ::3] = 20
    image = Image.fromarray(image_arr)
    mask = Image.new("L", (40, 40), 0)
    mask.paste(255, (18, 18, 22, 22))

    result, remaining = flat_fill(
        image,
        mask,
        bubbles=[{"bbox": [0, 0, 40, 40]}],
        texts=[{"bbox": [10, 10, 30, 30]}],
    )

    assert remaining is not None
    assert np.array_equal(np.array(result), np.array(image))


def test_flat_background_classifier_separates_white_and_screentone():
    white = Image.new("L", (100, 100), 255)
    tone_arr = np.full((100, 100), 186, dtype=np.uint8)
    tone_arr[::3, ::3] = 40
    mask = Image.new("L", (100, 100), 0)
    mask.paste(255, (45, 30, 55, 70))
    bubble = {"bbox": [0, 0, 100, 100]}
    config = {"inpainting": {}}

    assert main._is_flat_bubble_background(white, mask, bubble, config)
    assert not main._is_flat_bubble_background(
        Image.fromarray(tone_arr), mask, bubble, config
    )


def test_genai_replace_text_region_fails_closed_without_a_safe_glyph_mask():
    from inpainting.genai_inpainter import GenAIEditInpainter
    genai = GenAIEditInpainter({"genai_provider": "local_comfyui"})
    image = Image.new("RGB", (100, 100), (200, 200, 200))
    item = {
        "id": "f001",
        "bbox": [20, 20, 80, 80],
        "type": "floating_text",
        "translation": "테스트 효과음"
    }
    result = genai.replace_text_region(image, item)
    assert result.size == (100, 100)
    assert np.array_equal(np.array(result), np.array(image))


def test_floating_text_erasure_preserves_large_art_components(monkeypatch):
    from inpainting.genai_inpainter import GenAIEditInpainter

    class FakeInpainter:
        def __init__(self, config):
            pass

        def inpaint(self, image, mask):
            out = np.array(image.convert("RGB"), copy=True)
            out[np.array(mask) > 0] = (255, 0, 0)
            return Image.fromarray(out)

    monkeypatch.setattr("inpainting.lama_inpainter.LaMaONNXInpainter", FakeInpainter)
    image_arr = np.full((100, 100, 3), 255, dtype=np.uint8)
    image_arr[10:55, 10:55] = 0  # artwork: too large to be a glyph
    image_arr[75:80, 75:80] = 0  # glyph-sized component
    image = Image.fromarray(image_arr)
    mask = Image.new("L", image.size, 255)
    genai = GenAIEditInpainter({
        "genai_provider": "lama_onnx",
        "floating_text_max_component": 200,
        "floating_text_composite_dilate": 0,
    })

    result = genai.replace_text_region(
        image,
        {"id": "f001", "bbox": [0, 0, 100, 100], "translation": "대사"},
        mask=mask,
        render_translation=False,
    )

    assert result.getpixel((20, 20)) == (0, 0, 0)
    assert result.getpixel((77, 77)) == (255, 0, 0)


def test_typesetting_uses_horizontal_korean_and_includes_floating_text(tmp_path, monkeypatch):
    cleaned = tmp_path / "cleaned.png"
    Image.new("RGB", (240, 480), "white").save(cleaned)
    captured = {}

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command, **kwargs):
        plan_path = command[command.index("--plan") + 1]
        captured.update(json.loads(Path(plan_path).read_text(encoding="utf-8")))
        return Completed()

    monkeypatch.setattr(main.subprocess, "run", fake_run)
    detection = {
        "bubbles": [{
            "id": "b001",
            "bbox": [120, 10, 220, 300],
            "ocr_text": "数年後",
            "translation": "몇 년 후",
            "translation_status": "accepted",
        }],
        "floating_texts": [{
            "id": "f001",
            "type": "floating_text",
            "bbox": [10, 10, 100, 350],
            "ocr_text": "ちゃんと同意を取ってな!!",
            "translation": "제대로 동의를 구해!!",
            "translation_status": "accepted",
        }],
    }

    main.run_typesetting(cleaned, detection, {"typesetting": {}}, tmp_path, "page")

    plans = {plan["id"]: plan for plan in captured["typeset_plans"]}
    assert set(plans) == {"b001", "f001"}
    assert plans["b001"]["vertical"] is False
    assert plans["f001"]["vertical"] is True
    assert "몇 년" in plans["b001"]["text"]


def test_local_diffusers_inpainter():
    from unittest.mock import MagicMock
    from inpainting.diffusers_inpainter import LocalDiffusersInpainter
    from inpainting import get_inpainter

    inpainter = get_inpainter({"inpainting": {"backend": "local_diffusers"}})
    assert isinstance(inpainter, LocalDiffusersInpainter)

    # Mock diffusers pipeline for unit testing speed
    mock_pipe = MagicMock()
    mock_pipe.return_value.images = [Image.new("RGB", (64, 64), (100, 100, 100))]
    inpainter._pipeline = mock_pipe

    image = Image.new("RGB", (64, 64), (200, 200, 200))
    mask = Image.new("L", (64, 64), 0)
    out = inpainter.inpaint(image, mask)
    assert out.size == (64, 64)
    assert mock_pipe.called

    # Test fallback behavior when pipeline fails or is not installed
    inpainter._pipeline = False
    out_fb = inpainter.inpaint(image, mask)
    assert out_fb.size == (64, 64)


def test_inpainting_preserves_source_when_replacement_is_not_ready(tmp_path):
    image = Image.new("RGB", (40, 40), "white")
    pixels = np.array(image)
    pixels[10:30, 15:25] = 0
    image = Image.fromarray(pixels)

    text_mask = np.zeros((40, 40), dtype=np.uint8)
    text_mask[10:30, 15:25] = 255
    detection = {
        "page_id": "unresolved",
        "bubbles": [
            {
                "id": "b001",
                "bbox": [5, 5, 35, 35],
                "ocr_text": None,
                "translation": None,
            }
        ],
        "texts": [{"id": "t001", "bbox": [15, 10, 25, 30]}],
        "text_mask": text_mask,
    }
    config = {
        "inpainting": {
            "backend": "opencv",
            "flat_fill": True,
            "preserve_bubble_shape": True,
        },
        "comfyui": {"base_url": "", "timeout": 1},
    }

    output = main.run_inpainting(
        "unused.png",
        detection,
        config,
        tmp_dir=tmp_path,
        original=image,
    )

    assert np.array_equal(np.array(Image.open(output).convert("RGB")), np.array(image))


def test_dialogue_quality_report_does_not_count_unresolved_or_japanese_as_translated():
    from qa_module import build_dialogue_quality_report

    report = build_dialogue_quality_report(
        page_id="page",
        detection={
            "bubbles": [
                {"id": "b001", "ocr_text": "こんにちは", "translation": "안녕", "text_type": "dialogue"},
                {"id": "b002", "ocr_text": None, "translation": None, "text_type": "dialogue"},
                {"id": "b003", "ocr_text": "守る", "translation": "守る", "text_type": "dialogue"},
            ]
        },
        final_image="page_ko.png",
    )

    assert report["translated_dialogue"] == 1
    assert report["total_dialogue"] == 3
    assert report["dialogue_coverage"] == 33.33
    assert report["bubbles"][1]["flags"] == ["ocr_unresolved"]
    assert report["bubbles"][2]["flags"] == ["japanese_remaining"]


def test_dialogue_quality_report_fails_closed_for_manual_redraw():
    from qa_module import build_dialogue_quality_report

    report = build_dialogue_quality_report(
        page_id="page",
        detection={
            "bubbles": [],
            "floating_texts": [{
                "id": "f001",
                "ocr_text": "日本語",
                "translation": "한국어",
                "translation_status": "accepted",
                "render_status": "needs_review",
                "text_type": "dialogue",
            }],
        },
        final_image="page_ko.png",
    )

    assert report["dialogue_coverage"] == 0.0
    assert report["bubbles"][0]["flags"] == ["render_needs_review"]
    assert report["issues"] == ["f001: manual redraw review required"]


def test_dialogue_quality_report_counts_preserved_punctuation_as_handled():
    from qa_module import build_dialogue_quality_report

    report = build_dialogue_quality_report(
        page_id="page",
        detection={
            "bubbles": [{
                "id": "b001",
                "ocr_text": "…",
                "translation": None,
                "translation_status": "preserved",
                "skip_inpaint": True,
                "text_type": "dialogue",
            }],
        },
        final_image="page_ko.png",
    )

    assert report["dialogue_coverage"] == 100.0
    assert report["bubbles"][0]["flags"] == ["preserved"]
    assert report["issues"] == []


def test_replacement_gate_rejects_unreviewed_translation():
    from qa_module import is_replacement_ready

    assert not is_replacement_ready({
        "ocr_text": "日本語",
        "ocr_status": "accepted",
        "translation": "한국어 초벌",
        "translation_status": "needs_review",
        "text_type": "dialogue",
    })


def test_typesetting_skips_region_that_is_not_replacement_ready(tmp_path, monkeypatch):
    cleaned = tmp_path / "cleaned.png"
    Image.new("RGB", (40, 40), "white").save(cleaned)
    detection = {
        "bubbles": [{
            "id": "b001",
            "bbox": [5, 5, 35, 35],
            "ocr_text": "日本語",
            "ocr_status": "accepted",
            "translation": "한국어 초벌",
            "translation_status": "needs_review",
            "text_type": "dialogue",
        }]
    }
    render_calls = []
    monkeypatch.setattr(main.subprocess, "run", lambda *args, **kwargs: render_calls.append(args))

    result = main.run_typesetting(cleaned, detection, {"typesetting": {}}, tmp_path, "page")

    assert render_calls == []
    assert np.array_equal(np.array(Image.open(result)), np.array(Image.open(cleaned)))


def test_quality_benchmark_fails_when_a_known_region_is_missing():
    from scripts.quality_benchmark import evaluate_page

    result = evaluate_page(
        {
            "page_id": "015",
            "minimum_regions": 5,
            "required_source_texts": ["数年後"],
        },
        {
            "dialogue_coverage": 100.0,
            "issues": [],
            "bubbles": [
                {"ocr_text": "数年後", "translation": "몇 년 후", "flags": []},
                {"ocr_text": "台詞", "translation": "대사", "flags": []},
            ],
        },
    )

    assert result["passed"] is False
    assert result["region_count"] == 2
    assert result["failures"] == ["regions: 2 < required 5"]


def test_quality_benchmark_accepts_semantically_equivalent_translation_terms():
    from scripts.quality_benchmark import evaluate_page

    result = evaluate_page(
        {
            "page_id": "010",
            "minimum_regions": 1,
            "required_translation_any": [["비효율", "효율이 안 좋"]],
        },
        {
            "dialogue_coverage": 100.0,
            "issues": [],
            "bubbles": [{"ocr_text": "効率が悪いもの!!", "translation": "효율이 안 좋은걸!!"}],
        },
    )

    assert result["passed"] is True
    assert result["failures"] == []


def test_typesetter_measures_wrapped_korean_instead_of_forcing_minimum_font(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from scripts.render_text import STYLE_PRESETS, calculate_font_size

    size = calculate_font_size(
        260,
        180,
        "놀랄 정도로 자연스러운 한국어 대사",
        STYLE_PRESETS["normal"],
    )

    assert size > STYLE_PRESETS["normal"]["min_font_size"]


def test_translation_postprocess_does_not_force_banmal(monkeypatch):
    main._TRANSLATOR_CACHE.clear()
    detection = {"bubbles": [{"id": "b001", "ocr_text": "よろしくお願いします"}]}
    config = {"translation": {"provider": "openrouter", "model": "demo-model"}}

    class FakeCache:
        def hash_text(self, text):
            return hashlib.md5(text.encode("utf-8")).hexdigest()

        def get(self, key):
            return None

        def set(self, key, value):
            pass

    class FakeTranslator:
        def __init__(self, provider, model):
            pass

        def translate_batch(self, texts, api_key="", previous_context=None):
            return [{"id": "b001", "translation": "잘 부탁해요", "type": "dialogue"}]

    monkeypatch.setattr("translation.translator.Translator", FakeTranslator)

    main.run_translation(detection, config, api_key="key", cache=FakeCache())

    assert detection["bubbles"][0]["translation"] == "잘 부탁해요"


def test_google_translator_auth_uses_google_key_without_openrouter(monkeypatch):
    translator = Translator(provider="google-ai-studio", model="google/gemini-2.5-pro")
    monkeypatch.setenv("GOOGLE_API_KEY", "local-test-key")
    monkeypatch.setattr(
        translator,
        "_get_session",
        lambda: (_ for _ in ()).throw(AssertionError("must not call OpenRouter auth")),
    )

    assert translator.check_auth("") is True
