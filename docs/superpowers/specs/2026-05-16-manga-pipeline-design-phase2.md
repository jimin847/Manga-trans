# Manga Pipeline v2 - Phase 2 Enhancements Spec

## 1. Overview
This specification details the architecture for the Phase 2 enhancements, which aim to resolve major performance bottlenecks, improve translation quality through context, and add robust fault-tolerance.

## 2. Enhancements

### Step 1: Parallel OCR Processing (Speed Optimization)
**Current:** `run_ocr` processes each bubble crop sequentially, taking O(N) time where N is the number of bubbles.
**Design:** 
- Utilize `concurrent.futures.ThreadPoolExecutor` within `run_ocr` to send API requests in parallel.
- Limit max workers to 5 (or configurable) to prevent instant rate-limiting from OpenRouter.
- Results are mapped back to their respective bubble IDs safely.

### Step 2: Batch Translation Architecture (Context & Cost)
**Current:** `run_translation` translates each bubble independently. N bubbles = N API calls. Context is limited to the previous 5 lines.
**Design:**
- Gather all detected `ocr_text` into a single JSON payload: `[{"id": "b001", "text": "..."}]`.
- Send a single API request to the Translation LLM with instructions to return a JSON array: `[{"id": "b001", "translation": "..."}]`.
- Parse the resulting JSON and inject the translations back into the detection object.
- **Fallback:** If the JSON response is malformed, fall back to parsing line-by-line or re-requesting.

### Step 3: Local Caching & Advanced Rate Limiting (Resiliency)
**Current:** If the pipeline fails at the inpainting stage, rerunning the script costs new API calls for detection, OCR, and Translation. OpenRouter `429 Too Many Requests` causes failure after 8 seconds.
**Design:**
- **Caching:** Implement a `Cache` dict saved to `output/cache.json`.
  - OCR Cache: Key is `MD5(cropped_image_bytes)`. Value is the extracted text.
  - Translation Cache: Key is `MD5(full_page_ocr_json)`. Value is the translated JSON payload.
- **Rate Limit Defense:** Expand the `_call_api` retry logic in both OCR and Translation modules to explicitly catch `429` status codes, waiting up to 60 seconds using exponential backoff (e.g., 2s, 4s, 8s, 16s, 32s).

## 3. Implementation Order
1. Implement the Cache manager and update `main.py` to support caching.
2. Refactor `ocr/vlm_ocr.py` and `main.py` for ThreadPool parallel execution.
3. Refactor `translation/translator.py` and `main.py` for Batch JSON translation.
4. Run tests to ensure end-to-end stability.