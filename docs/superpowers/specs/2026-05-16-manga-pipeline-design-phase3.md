# Manga Pipeline v2 - Phase 3 Enhancements Spec

## 1. Overview
This specification details Phase 3 enhancements: Text Orientation Detection and SFX/Dialogue Separation. These improvements will significantly increase OCR accuracy for complex layouts and improve the aesthetic quality of the typeset output.

## 2. Enhancements

### Step 1: Text Orientation Detection
**Concept:** Japanese manga mixes horizontal and vertical text. OCR models perform better when they know the orientation.
**Implementation:**
- In `main.py` or the detection processing phase, calculate the aspect ratio of each bubble's bounding box: `width / height`.
- If `height > width * 1.5`, classify as `vertical`. If `width > height * 1.5`, classify as `horizontal`. Otherwise, `mixed/unknown`.
- Pass this `orientation` flag to the `ocr.read_crop()` method to dynamically adjust the system prompt (e.g., "Hint: This text is primarily vertical").

### Step 2: SFX vs Dialogue Classification via Batch Translation
**Concept:** The translation LLM already processes the whole page context. It is perfectly positioned to classify whether a piece of text is a standard dialogue or a Sound Effect (SFX) / typography.
**Implementation:**
- Update `translator.py`'s `TRANSLATE_SYSTEM_PROMPT` to require an additional JSON field: `"type"`.
- Instruct the LLM to classify each text as either `"dialogue"`, `"thought"`, or `"sfx"`.
  - `dialogue`: Standard speech bubbles.
  - `thought`: Internal monologues.
  - `sfx`: Onomatopoeia, sound effects, or background background text.

### Step 3: Typesetting Style Mapping
**Concept:** The typesetter (`manga-localization` Krita script) supports different styles (`style` parameter: normal, bold, italic, etc.).
**Implementation:**
- In `main.py` -> `run_typesetting`, map the LLM-classified `"type"` to a specific typesetting plan configuration.
- `dialogue` -> `"style": "normal"`
- `thought` -> `"style": "italic"`
- `sfx` -> `"style": "bold"`, `"font_policy": "sfx"` (or similar depending on Krita script capabilities).

## 3. Implementation Order
1. Update `main.py` `run_ocr` to calculate and pass orientation.
2. Update `ocr/vlm_ocr.py` to accept the orientation hint.
3. Update `translation/translator.py` prompt to output the `"type"` field.
4. Update `main.py` `run_typesetting` to utilize the new `"type"` field for styling.