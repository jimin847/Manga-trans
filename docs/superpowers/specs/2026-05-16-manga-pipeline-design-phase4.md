# Manga Pipeline v2 - Phase 4 Enhancements Spec

## 1. Overview
This specification details Phase 4 enhancements: Cross-page Context Tracking. Currently, the translation module only understands the context within a single page. By passing context across pages, the LLM can maintain narrative consistency, character tone (e.g., formal vs. informal), and continuous dialogue flow.

## 2. Enhancements

### Cross-page Context Tracking
**Concept:** When processing a batch of images sequentially (e.g., a chapter), the translation of page N should be influenced by the dialogue on page N-1.
**Implementation:**
1. **Global Context Buffer:** In `run_batch.py`, initialize a `global_context` list (or queue) that stores the last N translated texts (e.g., max 10 texts).
2. **Parameter Passing:** Update `main.py` -> `process_page` and `run_translation` to accept a `previous_context` string or list.
3. **Prompt Injection:** In `translator.py` -> `translate_batch`, inject this `previous_context` into the system prompt. Example: "Here is the dialogue from the previous page for context: [...]".
4. **Context Updating:** After `process_page` finishes, extract the successfully translated texts from that page, append them to the `global_context`, and keep only the last N items to pass to the next page.
5. **Alphabetical Processing:** `run_batch.py` already uses `sorted(glob.glob(...))`, which correctly orders pages. The context will naturally flow from page 01 to page N.

## 3. Implementation Order
1. Update `translation/translator.py` `translate_batch` signature and prompt to accept `previous_context`.
2. Update `main.py` `process_page` and `run_translation` to pass `previous_context` down.
3. Update `main.py` `process_page` to return the translated texts so the caller can collect them.
4. Update `run_batch.py` to maintain a rolling buffer of previous dialogue and pass it to `process_page`.