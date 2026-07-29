# Professional-quality manga localization pipeline

Status: evidence-backed implementation proposal, 2026-07-24  
Scope: Japanese manga images to natural Korean, including detection, OCR, translation, cleaning, typesetting, and QA.

## Executive conclusion

“A human translator did it” quality will not come from replacing one model. The feasible leap is a **failure-aware production pipeline**:

1. find text independently of speech balloons;
2. require agreement or review between multiple OCR views;
3. translate with page/chapter/speaker memory, produce several candidates, then run an independent MQM-style review;
4. erase only text whose translation is accepted, using a background-specific cleaner;
5. optimize lettering inside the actual balloon shape and reject visual defects automatically;
6. measure every change on a frozen local benchmark.

The invariant is: **uncertainty must be visible, and an uncertain stage must never destroy the source.** No automatic score proves professional quality; human-calibrated gates and a small correction queue remain necessary.

## Local 178-page evaluation plan

Use the supplied 178 images as a private evaluation corpus. Do not copy the source pages into Git; store only a local manifest (relative filename, dimensions, SHA-256) and derived metrics.

- Run all 178 pages before and after every milestone. Preserve stage artifacts and a per-region state record.
- Build a 36-page `quality-gold` set, stratified across dense dialogue, vertical/ruby text, handwritten/stylized text, narration, SFX/scene text, screentones, and line art. Freeze 24 development pages and 12 unseen holdout pages.
- Gold fields per region: polygon, class, reading order, source transcription (base text and ruby separately where meaningful), speaker/unknown, Korean translation, acceptable variants, and intended lettering style.
- Have a Japanese/Korean reviewer inspect the 12 holdout pages with MQM-Core categories. Promote every production failure into the regression set.
- Report distributions and page-level failures, not only averages. One omitted balloon, destructive cleanup, or critical mistranslation fails that page.

Minimum scorecard:

| Stage | Primary measures | Hard release gate |
|---|---|---|
| Detection | region recall, pixel-mask recall, false positives by class | no missed dialogue/narration on holdout |
| OCR | character error rate (CER), exact match, ruby/base errors | no unreviewed low-consensus source text |
| Translation | major/critical MQM errors, name/number/register consistency | zero accepted critical errors |
| Cleaning | outside-mask pixel changes, edge continuity, SSIM/LPIPS on synthetic ground truth | zero out-of-scope changes; no source removal before accepted translation |
| Lettering | polygon spill, clipping, missing glyphs, overflow, residual Japanese | all zero |
| Page | detected/accepted/rendered/reviewed counts | denominator is every detected or audit-proposed text region |

For cleaning, create **synthetic ground truth** from genuine clean balloon/texture patches: overlay Japanese glyph masks, remove them with each backend, and compare against the known original. SSIM measures structural similarity, while LPIPS was trained against human perceptual judgments and is useful as a complementary metric; neither replaces edge/screentone-specific checks ([SSIM paper](https://www.cns.nyu.edu/~lcv/pubs/makeAbs.php?loc=Wang03), [LPIPS project/paper](https://richzhang.github.io/PerceptualSimilarity/)).

## 1. Detection: optimize recall first, classify second

### Implement now

- Match inference resolution to the checkpoint before changing architectures. `detection/yolo_detector.py` currently omits `imgsz`, so Ultralytics uses its documented default of 640, while the selected text-segmentation model card trains and demonstrates inference at 1280. Add `yolo.detection_size: 1280`, pass it explicitly, and benchmark 640/960/1280 plus tiled inference ([Ultralytics Predict docs](https://docs.ultralytics.com/modes/predict/), [checkpoint model card](https://huggingface.co/ShadowB/Manga109-panel-balloon-text-yolov26-segmentation)).
- Decouple `text candidate` from `speech balloon`. Form a union of:
  1. the current text detector;
  2. the current balloon detector;
  3. dark-component/pixel-segmentation proposals at full resolution and overlapping tiles;
  4. a full-page visual audit that proposes suspicious uncovered text.
- Merge overlapping proposals, but retain provenance and confidence. Classify the union into dialogue, thought, narration, SFX, scene/editorial, and ignore only afterward.
- Preserve every unlinked text proposal as a reviewable `floating_text`; replace the current `area >= 400` discard rule with confidence/review routing.
- Run at two scales and overlapping tiles; evaluate **recall** on the gold polygons before tuning precision. A false positive can be rejected later; a missed region cannot be translated.
- Keep polygon/mask geometry through the entire pipeline instead of reducing it to a rectangle.

Manga109 provides page-level text, frame, face, and body annotations and text contents; it is a valid external benchmark after following its access terms ([Manga109 annotations](https://manga109.github.io/manga109-project-website/en/annotations.html)). Pixel-level evaluation is important because box metrics do not measure whether all glyph strokes were captured; the manga-specific “Unconstrained Text Detection” work explicitly introduced pixel-level annotations and metrics for this reason ([paper](https://arxiv.org/abs/2009.04042)).

### Next, after the baseline

- Fine-tune a text **instance-segmentation** model on corrected local masks plus licensed Manga109-derived data, with hard-negative crops from hair, eyes, speed lines, and screentones.
- Add panel order and speaker association as context metadata. Magi demonstrates a unified graph of panels, text, characters, reading order, and speakers, and its chapter-wide successor adds tails, text categories, and persistent character identities ([CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Sachdeva_The_Manga_Whisperer_Automatically_Generating_Transcriptions_for_Comics_CVPR_2024_paper.pdf), [ACCV 2024 paper](https://openaccess.thecvf.com/content/ACCV2024/papers/Sachdeva_Tails_Tell_Tales_Chapter-wide_Manga_Transcriptions_with_Character_Names_ACCV_2024_paper.pdf)). Use the architecture idea, not its weights in a distributable product: the official repository states that its models/datasets are academic-research-only ([Magi repository](https://github.com/ragavsachdeva/magi)).

## 2. OCR: agreement, not blind trust

### Implement now

- For every candidate, run both a whole-region crop and orientation-aware subcrops (vertical columns or horizontal lines) at explicitly recorded scales. Remove accidental double upscaling.
- Add local Manga OCR as an independent recognizer beside the existing VLM. It is specifically designed for vertical/horizontal, furigana, overlaid, low-quality, and multi-line manga text ([official repository](https://github.com/kha-white/manga-ocr)).
- Normalize punctuation/width only for comparison; keep raw outputs. Accept automatically only when recognizers agree after normalization or a retry resolves the disagreement.
- Retry disagreements with crop expansion, thresholded and original-color views, and column splits. Otherwise mark `needs_review`; do not guess and do not clean.
- Preserve base text and ruby in separate fields when ruby changes reading/name meaning. Send both, plus the crop, to translation.
- Detect `no text` outside the recognizer. Manga OCR itself warns that it always attempts recognition and may hallucinate on empty images; it also recommends smaller crops when long text fails ([documented limitations](https://github.com/kha-white/manga-ocr#usage-tips)).

### Experimental

- Character-level ensemble alignment with a Japanese language model can propose a consensus, but language plausibility must never override visible glyph evidence.
- Handwritten SFX needs a separate recognizer or review queue; Manga OCR explicitly does not claim handwritten support.

## 3. Translation: chapter-aware generate, estimate, refine

### Implement now

Create a persistent `chapter_state` before translating individual regions:

- canonical names and readings;
- relationships, gender only when evidenced, honorific policy, and speech-level policy;
- per-character register, verbal habits, and pronouns;
- terminology/onomatopoeia memory;
- ordered dialogue from neighboring panels/pages and speaker confidence;
- visual mood/action notes from the page crop.

Then use a three-step translation unit:

1. **Generate** 2–3 candidates: faithful/literal, natural character voice, and compact-for-balloon. All must preserve meaning; compactness never authorizes omission.
2. **Estimate** with a separate model/provider where possible. Return structured MQM error spans and severities for accuracy, terminology, linguistic conventions, style, audience, and design. These are the official MQM dimensions ([MQM typology](https://themqm.org/error-types-2/typology/)).
3. **Refine/select** only from diagnosed errors, then re-check names, numbers, negation, speaker/register, omissions/additions, Japanese residue, and glossary consistency.

Manga-specific evidence supports this structure: prior-scene plus work/genre context improved manga translation and helped resolve omitted Japanese subjects ([LREC-COLING 2024](https://aclanthology.org/2024.lrec-main.1505/)); multimodal LLM experiments also found value in page-image context, translation-unit size, and context length ([COLING 2025](https://aclanthology.org/2025.coling-main.232/)). More generally, document-level metrics outperformed sentence-level counterparts in roughly 85% of tested conditions in one WMT study and improved discourse-phenomena accuracy ([WMT 2022 paper](https://aclanthology.org/2022.wmt-1.6/)). Candidate generation plus quality-aware reranking also improved both automatic and human assessments across the evaluated datasets ([NAACL 2022 paper](https://aclanthology.org/2022.naacl-main.100/)).

Use COMET/XCOMET or GEMBA-MQM only as **advisory signals calibrated on the Japanese→Korean gold set**. XCOMET can identify MQM error spans; COMET supports Japanese and Korean through its multilingual backbone, context scoring, and candidate ranking, but model licenses vary ([official COMET repository](https://github.com/Unbabel/COMET)). GEMBA-MQM reported strong system-ranking results but its authors explicitly caution against treating a proprietary black-box judge as proof of improvement ([paper](https://aclanthology.org/2023.wmt-1.64/)). Never optimize and certify with the same judge alone.

### Required prompt/data corrections

- Remove contradictory name rules. One glossary/style guide is authoritative.
- Do visual/text classification before translation; the translator may flag a conflict but should not silently reclassify.
- Translate an ordered page or scene in one request, with stable IDs, instead of unrelated two-item batches.
- Cache against source crop hash + OCR text + chapter-state version + prompt/model version. A corrected OCR or glossary must invalidate downstream cache.

## 4. Cleaning: route by background, preserve by construction

### Implement now

- Gate cleaning on `translation_status == accepted`.
- White/near-flat balloon: fill only the connected interior text strokes using a robust local background estimate. Preserve the border and artwork exactly.
- Textured/screentone balloon: use a local context crop and compare patch-copy/OpenCV/LaMa candidates on the synthetic benchmark; choose by texture spectrum, edge continuity, SSIM, and LPIPS.
- Text crossing line art: detect edge endpoints around the mask, reconstruct structure first, then tone. The edge-first strategy is supported by EdgeConnect, whose two-stage method predicts edges before image completion ([paper](https://arxiv.org/abs/1901.00212)). Its published code is CC BY-NC, so do not import it into a commercial product without resolving licensing ([repository](https://github.com/knazeri/edge-connect)).
- Assert bit-exact equality outside the allowed edit mask plus a documented antialiasing seam band.

LaMa is immediately usable under Apache-2.0 and is designed for high-resolution, large-mask, periodic-structure completion ([official repository/paper](https://github.com/advimman/lama)). However, a general photographic inpainting score is not evidence of preserved manga line art. Manga restoration research shows that screentones are structured bitonal signals requiring special handling ([CVPR 2021 paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Xie_Exploiting_Aliasing_for_Manga_Restoration_CVPR_2021_paper.pdf)).

### Experimental, promote only after A/B proof

- Phase-aligned screentone synthesis: estimate local tone period/orientation, copy or synthesize a phase-matched patch, and reject spectral discontinuity.
- Structure-guided generative inpainting for artwork text outside balloons.
- Diffusion-based erasing or direct text replacement. These may hallucinate art or glyphs and must not become the default until they beat deterministic/LaMa routes on holdout and blind human review.

## 5. Typesetting: optimize the actual shape

### Implement now

- Derive a safe interior polygon by eroding the balloon mask; use its per-line available widths, not one shrunken rectangle.
- Enumerate Unicode-valid Korean breakpoints, shape every candidate, and search font size, line breaks, tracking, and line height jointly. UAX #14 defines line-break opportunities and specifically covers Korean syllable blocks ([Unicode UAX #14](https://www.unicode.org/reports/tr14/)).
- Use `QTextLayout`/`QTextLine` for measured line widths and independent placement; Qt documents this exact variable-width flow-around-shape pattern ([Qt text layout](https://doc.qt.io/qt-6/richtext-layouts.html), [QTextLayout API](https://doc.qt.io/qt-6/qtextlayout.html)). Do not resize a rendered bitmap to make it fit.
- Optimize a transparent score: zero spill/clipping first, then maximum readable minimum font size, balanced line lengths, center-of-mass alignment, and consistency with nearby balloons.
- Use Korean-localized OpenType fonts and verify glyph coverage. Noto CJK provides language-specific Korean fonts and appropriate horizontal/vertical forms ([official repository](https://github.com/notofonts/noto-cjk)). If vertical Korean is supported, honor grapheme clusters and Unicode vertical orientation rules ([Unicode UAX #50](https://www.unicode.org/reports/tr50/)).
- Store text, polygon, font, layout, and style in an editable sidecar; a reviewer must be able to correct one region without rerunning destructive stages.

### Visual QA gates

- rendered alpha is wholly inside the safe polygon;
- no clipped glyph ink and no missing-glyph boxes;
- no Japanese glyph-like residue inside accepted cleaned regions;
- no overlap with protected bubble borders or artwork edges;
- optional Korean OCR round-trip matches normalized target text (warning, not sole pass/fail);
- full-page contact sheet exposes original, masks, cleaned image, final image, and error badges.

## Implementation order mapped to this repository

1. **Benchmark and state model** — add gold-manifest/report tooling around `tests/`, `output/`, and `_build_dialogue_qa_report`; define explicit region states and fail-closed gates in `main.py`.
2. **Detection/OCR recall** — refactor `detection/yolo_detector.py`, `_prepare_ocr_crop`, and `run_ocr` to preserve proposal provenance, multi-view crops, OCR consensus, and review state.
3. **Chapter translation** — replace small isolated batches in `run_translation`/`translation/translator.py` with ordered scene input, chapter memory, candidate generation, MQM review, and versioned cache keys.
4. **Cleaning routes** — split `run_inpainting` into flat, screentone, and line-art strategies; enforce allowed-mask equality and synthetic reconstruction tests.
5. **Shape-aware lettering** — replace rectangle/bitmap-shrink behavior in `scripts/render_text.py` with measured polygon-aware `QTextLayout` search and hard visual gates.
6. **178-page regression** — run all pages, inspect the 12-page holdout blind, and merge only when no critical page regresses.

## What not to ship yet

- Magi weights/datasets without commercial permission (academic-research-only notice).
- EdgeConnect code in a commercial build without license clearance (CC BY-NC).
- A generative editor as the only cleaner or letterer.
- A single OCR, translator, or LLM judge as its own certificate.
- Any “100% coverage” report that excludes missed/failed OCR regions from the denominator.

## Decision rule

Implement the first five “implement now” slices behind reproducible benchmark reports. Promote an experimental component only if it improves the frozen 12-page holdout, does not create any new critical failure, passes a blind Japanese/Korean review, and has a compatible distribution license.
