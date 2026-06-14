# Manga Pipeline v2 Enhancement Spec

## 1. Overview
This specification addresses structural issues causing intermittent errors in the Manga translation pipeline and enhances its overall stability and quality.

## 2. Issues & Solutions

### A. OCR Fallback Logic
**Issue:** `vlm_ocr.py` receives a `fallback_model` argument, but never uses it. If the primary API call fails or rate limits, OCR silently drops the text.
**Solution:** Implement the fallback logic in `read_crop`. If the primary model fails after retries, attempt the fallback model.

### B. ComfyUI Timeout Bottleneck
**Issue:** `comfy_client.py` uses `_wait_for_result` to poll `/history/{prompt_id}`. If the prompt fails (e.g. invalid node or out of memory), it is never added to history, causing a 120-second timeout block.
**Solution:** Modify `_wait_for_result` to check `/queue`. If `prompt_id` is not in history AND not in the queue (neither pending nor running), throw an error immediately instead of waiting for timeout.

### C. Translation Regex Parsing
**Issue:** `translator.py` attempts to strip reasoning tags, but the regex `thinking[\s\S]*? response\s*` is incorrect for standard `<think>` tags, leading to reasoning leaks in the output.
**Solution:** Update the regex to properly match `<think>[\s\S]*?</think>` and clean up any remaining markdown blocks.

### D. Model Configuration
**Issue:** The default OCR model `google/gemma-4-31b-it:free` in `config.yaml` is primarily text-based and may not support vision/images properly on OpenRouter, causing `[NO TEXT]` results.
**Solution:** Change the default OCR model in `config.yaml` to a reliable vision-capable model, such as `qwen/qwen-2-vl-7b-instruct:free` or `google/gemini-2.0-flash-exp:free`.

## 3. Implementation Plan
- Edit `ocr/vlm_ocr.py` to implement the `fallback_model` pipeline.
- Edit `translation/translator.py` to fix the parsing regex.
- Edit `comfy_client.py` to add the fail-fast `/queue` check.
- Edit `config.yaml` to update the default model to a VLM model.
