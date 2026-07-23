"""
LLM translation module
"""
import json
import logging
import os
import re
from typing import Optional

import requests

logger = logging.getLogger(__name__)

TRANSLATE_SYSTEM_PROMPT = """You are a Japanese→Korean manga translation engine.

Task: Translate JSON-array of Japanese manga text into natural Korean. Classify each text's "type".

Input format:
[{"id": "b001", "text": "日本語"}, {"id": "b002", "text": "ドカーン"}]

Output format MUST be a valid JSON array and NOTHING else:
[{"id": "b001", "translation": "한국어", "type": "dialogue"}, {"id": "b002", "translation": "쾅", "type": "sfx"}]

Type Classification:
- "dialogue": Speech bubbles, conversations
- "thought": Internal monologue, narration
- "sfx": Onomatopoeia, sound effects, background text

CRITICAL RULES — follow exactly:

1. TRANSLATE ALL JAPANESE into Korean. Every word, every particle. No Japanese characters in output.

2. CHARACTER NAMES: Preserve character/person names exactly as they appear in the Japanese source unless an explicit glossary mapping is provided. Do NOT transliterate names to Hangul and do NOT translate names into Korean.
   - みゆき stays みゆき
   - 御幸 stays 御幸
   - 田中 stays 田中
   If a glossary mapping exists, use that exact translation for the mapped term only.

3. KOREAN SPACING: Every word boundary needs a space.
   WRONG: "미유키가만들었대" → RIGHT: "미유키가 만들었대"

4. Natural informal Korean speech (반말) for dialogue. No 요/습니다.

5. Sound effects: translate to natural Korean onomatopoeia. Never leave SFX in Japanese.

6. First char must be `[`, last must be `]`. No markdown, no explanations, no reasoning, no line-by-line analysis.

7. Preserve line breaks (\n) in output.

8. If uncertain, translate literally. Do NOT ask questions and do NOT output notes.

9. NEVER leave Japanese characters in translation. This includes Hiragana, Katakana, and Japanese Kanji.
   - Translate Japanese titles/honorifics: 旦那様→남편님, 様→님, さん/氏→씨, ちゃん→쨩, 先輩→선배, 先生→선생님.
   - Translate common literal traps naturally: 一生→평생, 最早→이제는, 不束者ですが→서툴지만/서툰 사람인데, 一片の憂い無し→한 점의 근심도 없어, いっしょうまも/一生守る→평생 지켜줄게.

10. Do NOT preserve Japanese names/titles unless they are explicit proper names or a glossary mapping is provided.
"""

class Translator:
    def __init__(
        self,
        provider: str = "openrouter",
        model: str = "openai/gpt-oss-120b:free",
        glossary: Optional[dict[str, str]] = None,
        preserve_terms: Optional[list[str]] = None,
    ):
        self.provider = provider
        self.model = model
        self.glossary = glossary or {}
        self.preserve_terms = preserve_terms or []
        self.api_base = "https://openrouter.ai/api/v1"
        self._session = None
        self.auth_failed = False

    def _get_session(self):
        if self._session is None:
            self._session = requests.Session()
        return self._session

    def check_auth(self, api_key: str) -> bool:
        """Return False for OpenRouter auth/key failures so translation can skip the page."""
        if self.auth_failed:
            return False

        session = self._get_session()
        try:
            resp = session.get(
                f"{self.api_base}/auth/key",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10,
            )
        except Exception as e:
            logger.warning(f"Translation auth check failed to reach OpenRouter, continuing to translation: {e}")
            return True

        if self._is_auth_error(resp):
            logger.error(f"Translation API auth check failed: {resp.status_code} {resp.text[:300]}")
            self.auth_failed = True
            return False
        return True

    @staticmethod
    def _is_auth_error(resp) -> bool:
        body = getattr(resp, "text", "").lower()
        return resp.status_code in (401, 403) or (
            resp.status_code == 400 and ("user not found" in body or "unauthorized" in body)
        )

    @staticmethod
    def _clean(text: str) -> str:
        """Clean markdown blocks and reasoning tags from JSON response."""
        if not text:
            return text

        # Strip thinking tags
        text = re.sub(r"<think>[\s\S]*?</think>\s*", "", text, flags=re.I)
        text = re.sub(r"^.*?thinking.*?response\s*", "", text, flags=re.I | re.S)

        # Strip markdown json blocks
        text = re.sub(r"```json\s*", "", text, flags=re.I)
        text = re.sub(r"```\s*", "", text, flags=re.I)
        text = re.sub(r"^\s*Here is .*?:\s*", "", text, flags=re.I)
        text = re.sub(r"^\s*JSON\s*:\s*", "", text, flags=re.I)

        # Try array first
        match = re.search(r"\[[\s\S]*\]", text)
        if match:
            return match.group(0).strip()

        # Try objects — detect comma-separated objects (not wrapped in array)
        obj_pattern = r"\{(?:[^{}]|(?!\})\{[^{}]*\})*\}"
        objs = re.findall(obj_pattern, text)
        if len(objs) >= 2:
            return "[" + ",".join(objs) + "]"

        # Single object
        obj_match = re.search(r"\{[\s\S]*\}", text)
        if obj_match:
            return obj_match.group(0).strip()
        return text.strip()

    def _build_system_prompt(self, extra_instruction: Optional[str] = None) -> str:
        prompt = TRANSLATE_SYSTEM_PROMPT
        rules = []
        for source, target in self.glossary.items():
            if isinstance(target, str) and target.strip():
                rules.append(f"- Preserve/translate exact term {source!r} as {target!r}.")
        for term in self.preserve_terms:
            if isinstance(term, str) and term.strip():
                rules.append(f"- Preserve exact source spelling for {term!r}; do not translate or transliterate it.")
        if rules:
            prompt += "\n\nGLOSSARY / PRESERVED TERMS:\n" + "\n".join(rules)
        if extra_instruction:
            prompt += "\n\nSTRICT EXTRA INSTRUCTION:\n" + extra_instruction
        return prompt

    @staticmethod
    def _sanitize_translation_item(item: dict) -> dict:
        translation = item.get("translation")
        if not isinstance(translation, str):
            return item

        # Convert literal \n (two chars) to actual newline — LLMs sometimes output both forms
        translation = translation.replace("\\n", "\n")

        translation = re.sub(r"\n{2,}Context\s*\(.*", "", translation, flags=re.I | re.S)
        translation = re.sub(r"\n{2,}주변 대화\s*:.*", "", translation, flags=re.I | re.S)
        translation = re.sub(r"\n{2,}Context\s*:.*", "", translation, flags=re.I | re.S)
        item["translation"] = translation.strip()
        return item

    @classmethod
    def _sanitize_batch_result(cls, items: list[dict]) -> list[dict]:
        return [cls._sanitize_translation_item(dict(item)) for item in items]

    def translate_batch(self, texts: list[dict], api_key: str = "", previous_context: Optional[list[str]] = None, extra_instruction: Optional[str] = None) -> Optional[list[dict]]:
        """Translate a batch of texts formatted as [{"id": "...", "text": "..."}]"""
        if self.auth_failed:
            return None
        if not texts:
            return []

        user_msg = json.dumps(texts, ensure_ascii=False, indent=2)
        system_prompt = self._build_system_prompt(extra_instruction=extra_instruction)
        
        if previous_context and len(previous_context) > 0:
            context_str = "\n".join([f"- {c}" for c in previous_context])
            system_prompt += f"\n\nContext (dialogue from the previous page for narrative continuity):\n{context_str}\n\nIMPORTANT: Do NOT translate the context, only translate the JSON input."

        messages = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": user_msg})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "top_p": 1,
            "max_tokens": 1500,
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning": {"exclude": True},
        }

        result = self._call_api(payload, api_key)
        if result:
            cleaned = self._clean(result)
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict) and "translations" in parsed:
                    return self._sanitize_batch_result(parsed["translations"])
                if isinstance(parsed, list):
                    return self._sanitize_batch_result(parsed)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse batch translation JSON: {e}\nRaw output: {cleaned}")
                # Fallback simple fix attempt
                try:
                    fixed = "[" + cleaned.split("[", 1)[1].rsplit("]", 1)[0] + "]"
                    return self._sanitize_batch_result(json.loads(fixed))
                except:
                    pass
        return None

    def _call_api(self, payload: dict, api_key: str) -> Optional[str]:
        """Make API call with retry logic. Routes google/ models to direct Google AI Studio API."""
        import time

        # ── Google AI Studio direct path (free) ──
        if self.model.startswith("google/"):
            return self._call_google_api(payload)

        # ── OpenRouter path ──
        session = self._get_session()

        max_retries = 6
        for attempt in range(1, max_retries + 1):
            try:
                resp = session.post(
                    f"{self.api_base}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=60,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"].strip()

                if self._is_auth_error(resp):
                    logger.error(f"Translation API auth error ({self.model}): {resp.status_code} {resp.text[:300]}")
                    self.auth_failed = True
                    return None

                logger.error(f"Translation API error ({self.model}) {resp.status_code} (attempt {attempt}/{max_retries})")
                if attempt < max_retries and resp.status_code in (502, 503, 504, 429):
                    base_wait = 4 if resp.status_code == 429 else 2
                    wait = min(base_wait ** attempt, 60)
                    logger.info(f"  Retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                return None

            except (requests.ConnectionError, requests.Timeout) as e:
                logger.error(f"Translation API connection error (attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    wait = min(2 ** attempt, 60)
                    time.sleep(wait)
                    continue
                return None
        return None

    def _call_google_api(self, payload: dict) -> Optional[str]:
        """Call Google AI Studio Gemini API directly (free tier, no OpenRouter proxy)."""
        import time

        # Resolve Google API key
        google_key = os.environ.get("GOOGLE_API_KEY", "")
        if not google_key:
            env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
            if os.path.exists(env_path):
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("GOOGLE_API_KEY="):
                            google_key = line.split("=", 1)[1].strip().strip("\"'")
                            break
        if not google_key:
            logger.error("No Google API key found — set GOOGLE_API_KEY env var or in .env")
            return None

        model_name = self.model.replace("google/", "", 1)

        # Convert OpenRouter-style messages to Google contents format
        system_prompt = ""
        user_texts = []
        for msg in payload.get("messages", []):
            if msg["role"] == "system":
                system_prompt = msg["content"]
            elif msg["role"] == "user":
                user_texts.append(msg["content"])

        user_content = "\n".join(user_texts)

        google_payload: dict = {
            "contents": [{"parts": [{"text": user_content}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 4096, "topP": 1},
        }
        if system_prompt:
            google_payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        session = self._get_session()
        max_retries = 6
        for attempt in range(1, max_retries + 1):
            try:
                resp = session.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent",
                    params={"key": google_key},
                    headers={"Content-Type": "application/json"},
                    json=google_payload,
                    timeout=60,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            return parts[0].get("text", "").strip()
                    logger.error(f"Google API: empty response — {json.dumps(data, ensure_ascii=False)[:500]}")
                    return None

                if self._is_auth_error(resp):
                    logger.error(f"Translation API auth error ({self.model}): {resp.status_code} {resp.text[:300]}")
                    self.auth_failed = True
                    return None

                logger.error(f"Translation API error ({self.model}) {resp.status_code} (attempt {attempt}/{max_retries})")
                if attempt < max_retries and resp.status_code in (429, 500, 502, 503, 504):
                    base_wait = 4 if resp.status_code == 429 else 2
                    wait = min(base_wait ** attempt, 60)
                    logger.info(f"  Retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                return None

            except (requests.ConnectionError, requests.Timeout) as e:
                logger.error(f"Google API connection error (attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    wait = min(2 ** attempt, 60)
                    time.sleep(wait)
                    continue
                return None
        return None
