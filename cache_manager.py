import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class CacheManager:
    def __init__(self, cache_file: str | Path, autosave: bool = True):
        self.cache_file = Path(cache_file)
        self.autosave = autosave
        self.cache_data = {}
        self._dirty = False
        self._load()

    def _load(self):
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.cache_data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache from {self.cache_file}: {e}")
                self.cache_data = {}

    def _save(self):
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache_data, f, indent=2, ensure_ascii=False)
            self._dirty = False
        except Exception as e:
            logger.warning(f"Failed to save cache to {self.cache_file}: {e}")

    def flush(self):
        if self._dirty:
            self._save()

    def get(self, key: str) -> Optional[str]:
        return self.cache_data.get(key)

    def set(self, key: str, value: str):
        self.cache_data[key] = value
        self._dirty = True
        if self.autosave:
            self._save()

    @staticmethod
    def hash_image(image_bytes: bytes) -> str:
        return hashlib.md5(image_bytes).hexdigest()

    @staticmethod
    def hash_text(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()
