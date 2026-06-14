# Manga Pipeline v2 - Phase 5 Enhancements Spec

## 1. Overview
This specification details Phase 5 enhancements: Furigana & Noise Filtering (OCR Text Refinement). Japanese manga frequently uses furigana (ruby characters) to indicate kanji pronunciation. OCR models often read both the kanji and the furigana simultaneously, resulting in duplicated or interlaced gibberish. Furthermore, background screentones or line art are occasionally misinterpreted as repetitive punctuation (e.g., `....`, `,,,`, `///`). This enhancement cleans the OCR output before it reaches the translator.

## 2. Enhancements

### Step 1: Prompt Engineering for Furigana Suppression
**Concept:** The VLM performing OCR is intelligent enough to distinguish between main text and furigana if explicitly instructed.
**Implementation:**
- Update `OCR_SYSTEM_PROMPT` in `ocr/vlm_ocr.py`.
- Add explicit, strong directives:
  - "CRITICAL: IGNORE furigana (ruby characters) printed next to or above kanji. Only extract the primary text."
  - "Do NOT include repetitive punctuation artifacts caused by background screentones (e.g., do not output endless periods or commas)."
  - "Output clean, coherent Japanese text."

### Step 2: Post-Processing Regex Cleaner (Artifact Removal)
**Concept:** Even with prompt engineering, models occasionally hallucinate noise from complex backgrounds. We can apply regex filtering to sanitize the output.
**Implementation:**
- Add a text sanitization block inside `VlmOcr._clean`.
- Remove excessive repeating punctuation: `re.sub(r'([.、。・…]){3,}', r'\1\1\1', text)` (Cap repeating punctuation to max 3).
- Remove meaningless isolated symbols that often represent noise: e.g., lines containing only `|`, `/`, `\`, `_`.
- Strip leading/trailing whitespaces aggressively.

## 3. Implementation Order
1. Update `OCR_SYSTEM_PROMPT` in `ocr/vlm_ocr.py` to enforce furigana and noise suppression.
2. Update the `_clean` method in `ocr/vlm_ocr.py` to include regex-based artifact cleanup.
3. No changes needed in `main.py`; the cleaned text will naturally flow into the cache and batch translator.