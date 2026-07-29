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

TRANSLATE_SYSTEM_PROMPT = """You are a senior Japanese→Korean manga localizer.

Task: Translate JSON-array of Japanese manga text into natural Korean. Classify each text's "type".

Input format:
[{"id": "b001", "text": "日本語"}, {"id": "b002", "text": "ドカーン"}]

Output format MUST be a valid JSON array and NOTHING else:
[{"id": "b001", "translation": "한국어", "type": "dialogue"}, {"id": "b002", "translation": "쾅", "type": "sfx"}]

Type Classification:
- "dialogue": Speech bubbles, conversations
- "thought": Internal monologue, narration
- "sfx": Onomatopoeia, sound effects, background text

QUALITY RULES — follow exactly:

1. TRANSLATE ALL JAPANESE into Korean. Every word, every particle. No Japanese characters in output.

2. CHARACTER NAMES: The glossary is authoritative. Transliterate Japanese proper names into Hangul using their most likely Japanese reading when no glossary entry exists. Never leave Japanese glyphs. If a reading is genuinely ambiguous, choose the contextually strongest Korean reading and add "needs_review": true.

3. KOREAN SPACING: Every word boundary needs a space.
   WRONG: "미유키가만들었대" → RIGHT: "미유키가 만들었대"

4. Preserve each speaker's register, relationship, emotion, subtext, and verbal habits. Do not force every speaker into 반말. Prefer idiomatic spoken Korean over Japanese syntax or word-for-word calques.

5. Sound effects: classify them as "sfx". Translate only when requested by the caller; never mistake stylized SFX for dialogue.

6. First char must be `[`, last must be `]`. No markdown, no explanations, no reasoning, no line-by-line analysis.

7. Return semantic Korean text without copying Japanese visual line breaks. Lettering decides Korean line breaks later.

8. Silently compare a faithful, a character-voice, and a compact candidate. Select the most natural version that loses no meaning. Compactness never permits omission. Use "needs_review": true for unresolved ambiguity.

9. NEVER leave Japanese characters in translation. This includes Hiragana, Katakana, and Japanese Kanji.
   - Translate Japanese titles/honorifics: 旦那様→남편님, 様→님, さん/氏→씨, ちゃん→쨩, 先輩→선배, 先生→선생님.
   - Translate common literal traps naturally: 一生→평생, 最早→이제는, 不束者ですが→서툴지만/서툰 사람인데, 一片の憂い無し→한 점의 근심도 없어, いっしょうまも/一生守る→평생 지켜줄게.

10. Check negation, numbers, names, honorifics, omitted subjects, tense, and additions/omissions before returning the final JSON.

11. Render expressive Japanese vowel extension as natural Korean punctuation or cadence, never by misspelling Korean syllables. For example, ごめーん should become 미안~!/미안! according to tone, never 미아안/미안안.

12. Interpret sentence-final particles by their function in context. Do not mechanically render よ/ね/な as filler such as "말이지". Requests and commands ending in forms such as ～てよ/～てな should use a natural Korean request or command ending while preserving force.

13. Never leave a Korean connective form such as "-고서" dangling before sentence-final punctuation. Complete the request, command, or assertion naturally.

14. A katakana token immediately followed by a Japanese title or honorific (氏, さん, 君, くん, ちゃん, 様, etc.) is a person's name. Transliterate its Japanese sounds into Hangul; never reinterpret it as an English word or translate its meaning.

15. Preserve comic pseudo-archaic voices such as 拙者/ござる with fluent, internally consistent Korean characterization. Never create ungrammatical hybrid endings such as "-소이다".
"""

REVIEW_SYSTEM_PROMPT = """You are the independent senior Korean manga editor reviewing Japanese source text and draft Korean localization.

Return a JSON array only, using the same IDs and fields: id, translation, type, and optional needs_review.
For every item, compare source and draft and correct only real defects. Apply MQM-style checks for:
- mistranslation, omission, addition, negation, number, name, and honorific errors;
- speaker register, character voice, scene emotion, natural Korean idiom, and consistency;
- excessive length that can be shortened without losing meaning;
- any remaining Japanese glyphs or translationese.
Reject nonstandard Korean vowel-stretch spellings such as 미아안 or 그으래; express elongation with natural punctuation or cadence instead.
Interpret Japanese sentence-final particles by their dialogue function. Remove literal filler such as "말이지" when a natural Korean request, command, or assertion conveys the source more accurately.
Reject dangling connective endings such as "-고서!!"; make the Korean sentence pragmatically complete.
Verify that katakana names before titles/honorifics are phonetic Hangul transliterations, not English-word reinterpretations.
Keep pseudo-archaic character voice when present, but replace malformed hybrid Korean endings with a fluent, consistent register.
The glossary is authoritative. Never invent facts absent from the source or context. Never expose analysis.
"""

class Translator:
    def __init__(
        self,
        provider: str = "openrouter",
        model: str = "openai/gpt-oss-120b:free",
        glossary: Optional[dict[str, str]] = None,
        preserve_terms: Optional[list[str]] = None,
        review_model: Optional[str] = None,
    ):
        self.provider = provider
        self.model = model
        self.review_model = review_model or model
        self.glossary = glossary or {}
        self.preserve_terms = preserve_terms or []
        self.api_base = "https://openrouter.ai/api/v1"
        self._session = None
        self._cli_clients = {}
        self.auth_failed = False

    def _get_session(self):
        if self._session is None:
            self._session = requests.Session()
        return self._session

    @staticmethod
    def _google_api_key() -> str:
        google_key = os.environ.get("GOOGLE_API_KEY", "")
        if google_key:
            return google_key
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
        if os.path.exists(env_path):
            with open(env_path) as env_file:
                for line in env_file:
                    line = line.strip()
                    if line.startswith("GOOGLE_API_KEY="):
                        return line.split("=", 1)[1].strip().strip("\"'")
        return ""

    def check_auth(self, api_key: str) -> bool:
        """Return False for OpenRouter auth/key failures so translation can skip the page."""
        if self.auth_failed:
            return False
        if self.provider == "antigravity-cli":
            client = self._get_cli_client(self.model)
            if not client.available:
                logger.error("Antigravity CLI translation requires the 'agy' executable")
                self.auth_failed = True
                return False
            return True
        if self.provider == "google-ai-studio" or self.model.startswith("google/"):
            return bool(self._google_api_key())

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

    def _get_cli_client(self, model: str):
        from antigravity_client import AntigravityCliClient

        client = self._cli_clients.get(model)
        if client is None:
            client = AntigravityCliClient(model=model, timeout=180)
            self._cli_clients[model] = client
        return client

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
                if re.search(r"[\u3040-\u30ff\u3400-\u9fff]", term):
                    rules.append(
                        f"- Transliterate Japanese term {term!r} into Hangul consistently; "
                        "never preserve its Japanese glyphs."
                    )
                else:
                    rules.append(f"- Preserve exact source spelling for {term!r}.")
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
        return self._parse_batch_response(result)

    def review_batch(
        self,
        texts: list[dict],
        drafts: list[dict],
        api_key: str = "",
        previous_context: Optional[list[str]] = None,
    ) -> Optional[list[dict]]:
        """Independently edit a page-level draft while preserving stable IDs."""
        if self.auth_failed or not drafts:
            return None

        system_prompt = REVIEW_SYSTEM_PROMPT
        glossary_rules = [
            f"- {source!r} must be {target!r}."
            for source, target in self.glossary.items()
            if isinstance(target, str) and target.strip()
        ]
        if glossary_rules:
            system_prompt += "\n\nGLOSSARY:\n" + "\n".join(glossary_rules)
        if previous_context:
            context_str = "\n".join(f"- {item}" for item in previous_context)
            system_prompt += f"\n\nPrevious-page context (do not translate):\n{context_str}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {"sources": texts, "drafts": drafts},
                        ensure_ascii=False,
                        indent=2,
                    ),
                },
            ],
            "temperature": 0,
            "top_p": 1,
            "max_tokens": 2000,
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning": {"exclude": True},
        }
        return self._parse_batch_response(
            self._call_api(payload, api_key, model_override=self.review_model)
        )

    @classmethod
    def _parse_batch_response(cls, result: Optional[str]) -> Optional[list[dict]]:
        if result:
            cleaned = cls._clean(result)
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict) and "translations" in parsed:
                    return cls._sanitize_batch_result(parsed["translations"])
                if isinstance(parsed, list):
                    return cls._sanitize_batch_result(parsed)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse batch translation JSON: {e}\nRaw output: {cleaned}")
                # Fallback simple fix attempt
                try:
                    fixed = "[" + cleaned.split("[", 1)[1].rsplit("]", 1)[0] + "]"
                    return cls._sanitize_batch_result(json.loads(fixed))
                except:
                    pass
        return None

    def _call_api(
        self,
        payload: dict,
        api_key: str,
        model_override: Optional[str] = None,
    ) -> Optional[str]:
        """Make API call with retry logic. Routes google/ models to direct Google AI Studio API."""
        import time

        active_model = model_override or self.model

        if self.provider == "antigravity-cli":
            return self._call_antigravity(payload, active_model)

        # ── Google AI Studio direct path (free) ──
        if active_model.startswith("google/"):
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

    def _call_antigravity(self, payload: dict, model: str) -> Optional[str]:
        """Call an official Antigravity agent using the user's subscription session."""
        sections = []
        for message in payload.get("messages", []):
            role = str(message.get("role") or "user").upper()
            content = message.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            sections.append(f"{role} INSTRUCTIONS:\n{content}")
        prompt = (
            "Complete this localization task directly. The supplied SYSTEM INSTRUCTIONS are authoritative. "
            "Do not inspect the filesystem, run commands, or discuss the task. Return only the requested output.\n\n"
            + "\n\n".join(sections)
        )
        client = self._get_cli_client(model)
        result = client.generate(prompt)
        if result is None and "not logged" in client.last_error.lower():
            self.auth_failed = True
        return result

    def _call_google_api(self, payload: dict) -> Optional[str]:
        """Call Google AI Studio Gemini API directly (free tier, no OpenRouter proxy)."""
        import time

        # Resolve Google API key
        google_key = self._google_api_key()
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
                    headers={
                        "Content-Type": "application/json",
                        "x-goog-api-key": google_key,
                    },
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
