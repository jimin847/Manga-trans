import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import main
from ocr.vlm_ocr import VlmOcr
from cache_manager import CacheManager
from translation.translator import Translator


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

    assert saved == {"ocr_v2_hash1": "[NO TEXT]"}
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

    assert all(key.startswith("trans_v2_") for key in seen_keys)


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

    assert all(key.startswith("ocr_v2_") for key in seen_keys)


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
