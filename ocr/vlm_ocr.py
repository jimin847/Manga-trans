"""VLM OCR — crop 단위로 텍스트 읽기 (전체 페이지 X)
Supports OpenRouter and Google AI Studio providers.
"""
import base64
import json
import logging
import os
import threading
from io import BytesIO
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)

OCR_SYSTEM_PROMPT = """You are a Japanese manga OCR scanner. You have NO translation capability. You output only raw Japanese characters as they appear in the image.
- Copy the EXACT visible characters, no more, no less
- NEVER translate to Korean, English, or any other language
- NEVER add explanations, notes, or commentary
- Preserve kanji, hiragana, katakana, numbers, and punctuation exactly as shown
- For katakana words, copy katakana exactly. Do NOT replace them with Korean, English, or similar-looking characters
- For vertical text: read top-to-bottom, right-to-left
- For horizontal text: read left-to-right, top-to-bottom
- Include sound effects (SFX) if any
- Ignore furigana (ruby annotations) — extract primary text only
- Ignore repeating dots/commas from background screentones
- If no text is clearly visible, output exactly: [NO TEXT]"""


class VlmOcr:
    def __init__(
        self,
        provider: str = "openrouter",
        model: str = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        fallback_model: Optional[str] = None,
    ):
        self.provider = provider
        self.model = model
        self.fallback_model = fallback_model
        self.api_base = "https://openrouter.ai/api/v1"
        self._thread_local = threading.local()
        self._api_lock = threading.Lock()
        self.auth_failed = False

    @staticmethod
    def _is_auth_error(resp) -> bool:
        body = getattr(resp, "text", "").lower()
        return resp.status_code in (401, 403) or (
            resp.status_code == 400 and ("user not found" in body or "unauthorized" in body)
        )

    def _get_session(self):
        import requests

        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            self._thread_local.session = session
        return session

    def check_auth(self, api_key: str) -> bool:
        """Return False for OpenRouter auth/key failures so OCR can skip the page."""
        if self.auth_failed or self.provider == "google-ai-studio":
            return not self.auth_failed

        session = self._get_session()
        try:
            resp = session.get(
                f"{self.api_base}/auth/key",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10,
            )
        except Exception as e:
            logger.warning(f"OCR auth check failed to reach OpenRouter, continuing to OCR: {e}")
            return True

        if self._is_auth_error(resp):
            logger.error(f"OCR API auth check failed: {resp.status_code} {resp.text[:300]}")
            self.auth_failed = True
            self._thread_local.auth_failed = True
            return False
        return True

    def _call_api(self, crop: Image.Image, api_key: str, model: str, timeout: int = 30, orientation: Optional[str] = None) -> Optional[str]:
        buf = BytesIO()
        crop.save(buf, format="JPEG", quality=95)
        b64 = base64.b64encode(buf.getvalue()).decode()

        user_text = "Transcribe the EXACT characters in this image character-by-character in Japanese. Do NOT translate. If the text says '15万' in the image, write '15万' — not Korean, not English, not any other language. For katakana words, preserve katakana exactly. Output the raw Japanese text exactly as written."
        if orientation == "vertical":
            user_text += " Hint: vertical text — read top-to-bottom, right-to-left."
        elif orientation == "horizontal":
            user_text += " Hint: horizontal text — read left-to-right, top-to-bottom."

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": OCR_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                },
            ],
            "temperature": 0.0,
            "max_tokens": 512,
            "top_p": 1,
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning": {"exclude": True},
        }

        import time
        session = self._get_session()

        max_retries = 6
        for attempt in range(1, max_retries + 1):
            try:
                with self._api_lock:
                    if self.auth_failed:
                        return None
                    resp = session.post(
                        f"{self.api_base}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                        timeout=timeout,
                    )

                if resp.status_code == 200:
                    break  # success

                if self._is_auth_error(resp):
                    logger.error(f"OCR API auth error ({model}): {resp.status_code} {resp.text[:300]}")
                    self.auth_failed = True
                    self._thread_local.auth_failed = True
                    return None

                logger.error(f"OCR API error ({model}) {resp.status_code} (attempt {attempt}/{max_retries})")
                try:
                    logger.error(f"  Response: {resp.text[:300]}")
                except:
                    pass
                if attempt < max_retries and resp.status_code in (502, 503, 504, 429):
                    # For rate limits, wait longer
                    base_wait = 4 if resp.status_code == 429 else 2
                    wait = base_wait ** attempt
                    wait = min(wait, 60) # cap at 60 seconds
                    logger.info(f"  Retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                return None

            except Exception as e:
                requests = __import__("requests")
                if not isinstance(e, (requests.ConnectionError, requests.Timeout)):
                    raise
                logger.error(f"OCR API connection error (attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    wait = min(2 ** attempt, 60)
                    time.sleep(wait)
                    continue
                return None

        try:
            data = resp.json()
            msg = data["choices"][0]["message"]
            text = (msg.get("content") or "").strip()
            text = self._clean(text)
            if not text or text == "[NO TEXT]":
                return None
            return text
        except (KeyError, IndexError, ValueError) as e:
            logger.error(f"OCR parse error ({model}): {e}")
            return None

    def _call_google_api(self, crop: Image.Image, api_key: str, model: str, timeout: int = 30,
                         orientation: Optional[str] = None) -> Optional[str]:
        # Resolve Google API key: check env var, then .env file
        google_key = os.environ.get("GOOGLE_API_KEY", "")
        if not google_key:
            # Try the skill-level .env
            env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
            if os.path.exists(env_path):
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("GOOGLE_API_KEY="):
                            google_key = line.split("=", 1)[1].strip().strip("\"'")
                            break
        if not google_key:
            logger.error("No Google API key found — set GOOGLE_API_KEY in .env")
            return None
        buf = BytesIO()
        crop.save(buf, format="JPEG", quality=95)
        b64 = base64.b64encode(buf.getvalue()).decode()

        # Google AI Studio: no system_instruction, all in user prompt for better OCR behavior
        user_text = """You are a character-copy machine for Japanese manga. Your ONLY function is to TRANSCRIBE the exact visible characters. You have NO language ability, NO translation module — you just copy characters.

RULES:
- Copy the characters exactly as they appear
- If the image shows "15万" → output: 15万
- If the image shows "30歳の誕生日に" → output: 30歳の誕生日に
- NEVER translate. NEVER output Korean or English.
- Output ONLY the raw text, no explanations"""

        if orientation == "vertical":
            user_text += "\nThis text is vertical — read top-to-bottom, right-to-left."
        elif orientation == "horizontal":
            user_text += "\nThis text is horizontal — read left-to-right, top-to-bottom."

        payload = {
            "contents": [{
                "parts": [
                    {"text": user_text},
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                ]
            }],
            "generation_config": {"temperature": 0.0, "max_output_tokens": 512, "top_p": 1},
        }

        import time
        session = self._get_session()

        max_retries = 6
        for attempt in range(1, max_retries + 1):
            try:
                with self._api_lock:
                    if self.auth_failed:
                        return None
                    resp = session.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={google_key}",
                        headers={"Content-Type": "application/json"},
                        json=payload,
                        timeout=timeout,
                    )

                if resp.status_code == 200:
                    break  # success

                if self._is_auth_error(resp):
                    logger.error(f"Google OCR API auth error ({model}): {resp.status_code} {resp.text[:300]}")
                    self.auth_failed = True
                    self._thread_local.auth_failed = True
                    return None

                logger.error(f"Google OCR API error ({model}) {resp.status_code} (attempt {attempt}/{max_retries})")
                body = resp.text[:500]
                if body:
                    logger.error(f"  Response body: {body}")
                if attempt < max_retries and resp.status_code in (502, 503, 504):
                    base_wait = min(2 ** attempt, 60)
                    logger.info(f"  Retrying in {base_wait}s...")
                    time.sleep(base_wait)
                    continue
                if resp.status_code == 429:
                    # Free tier: don't retry 429 — each retry resets the sliding window
                    # With 3s throttle, next bubble request will be within 20 req/min
                    logger.error(f"  Google OCR 429 — skipping (3s throttle between bubbles usually avoids this)")
                return None

            except Exception as e:
                import requests
                if not isinstance(e, (requests.ConnectionError, requests.Timeout)):
                    raise
                logger.error(f"Google OCR connection error (attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    wait = min(2 ** attempt, 60)
                    time.sleep(wait)
                    continue
                return None

        try:
            data = resp.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            text = self._clean(text)
            if not text or text == "[NO TEXT]":
                return None
            return text
        except (KeyError, IndexError, ValueError) as e:
            logger.error(f"Google OCR parse error ({model}): {e}")
            return None

    @staticmethod
    def _is_placeholder_text(text: str) -> bool:
        import re
        compact = re.sub(r"\s+", "", text)
        if not compact:
            return False
        placeholder_chars = set("〇○◯●・。、,.!?！？…◇◆□■▢△▲▽▼○●")
        kana_kanji = re.search(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]", compact)
        if kana_kanji and any(ch not in placeholder_chars and not ch.isdigit() and ch not in "第話歳年ヶ月日" for ch in compact):
            return False
        return all(ch in placeholder_chars or ch.isdigit() or ch in "第話歳年ヶ月日" for ch in compact)

    @staticmethod
    def _clean(text: str) -> str:
        import re
        # Strip thinking and translation tags
        text = re.sub(r"<think>[\s\S]*?</think>\s*", "", text, flags=re.I)
        text = re.sub(r"^(Translation|Korean):\s*", "", text, flags=re.I)
        text = re.sub(r"^\s*\[NO TEXT\]\s*$", "", text, flags=re.I | re.M)
        
        # Noise reduction: cap repeating punctuation to max 3 (e.g. ....... -> ...)
        text = re.sub(r'([.、。・…!?,~～/|]){4,}', r'\1\1\1', text)
        text = re.sub(r"[〇○◯●◇◆□■▢△▲▽▼]{2,}", " ", text)
        
        # If line contains only symbols/punctuation, it's likely background noise
        lines = text.split("\n")
        cleaned_lines = []
        leak_markers = (
            "the image",
            "visible text",
            "output only",
            "required by the instructions",
            "browser tab",
            "developer tools",
            "i should output",
            "there is no visible text",
        )
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if "[NO TEXT]" in line.upper():
                continue
            if any(marker in line.lower() for marker in leak_markers):
                continue
            # Check if line has any actual word characters (kanji/kana/alpha)
            if re.search(r'[a-zA-Z0-9\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', line):
                cleaned_lines.append(line)
                
        cleaned_text = "\n".join(cleaned_lines).strip()
        if VlmOcr._is_placeholder_text(cleaned_text):
            return ""
        return cleaned_text

    def read_crop(self, crop: Image.Image, api_key: str, orientation: Optional[str] = None) -> Optional[str]:
        """Read text from a bubble crop image. Retry once if reasoning leak detected. Fallback if failed."""
        if self.auth_failed:
            return None
        self._thread_local.auth_failed = False
        if self.provider == "google-ai-studio":
            text = self._call_google_api(crop, api_key, self.model, timeout=30, orientation=orientation)
            if self.auth_failed or getattr(self._thread_local, "auth_failed", False):
                return None

            if not text and self.fallback_model:
                logger.warning(f"  Primary OCR failed. Retrying with fallback: {self.fallback_model}")
                text = self._call_google_api(crop, api_key, self.fallback_model, timeout=30, orientation=orientation)

            return text

        # OpenRouter provider (default)
        text = self._call_api(crop, api_key, self.model, timeout=30, orientation=orientation)
        if self.auth_failed or getattr(self._thread_local, "auth_failed", False):
            return None

        # Fallback if primary model completely failed
        if not text and self.fallback_model:
            logger.warning(f"  Primary OCR failed. Retrying with fallback: {self.fallback_model}")
            text = self._call_api(crop, api_key, self.fallback_model, timeout=30, orientation=orientation)

        # If response looks like reasoning leak (all-ASCII text, no Japanese), retry once
        if text and len(text) > 10 and all(ord(c) < 128 for c in text[:20]):
            logger.info(f"  Reasoning leak detected, retrying...")
            model_to_use = self.fallback_model if self.fallback_model else self.model
            text2 = self._call_api(crop, api_key, model_to_use, timeout=30, orientation=orientation)
            if text2:
                text = text2

        return text
