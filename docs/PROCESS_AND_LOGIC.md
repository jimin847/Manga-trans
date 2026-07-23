# Manga-Trans v2 상세 프로세스 및 알고리즘 명세 (Process & Logic)

본 문서는 파이프라인의 각 단계별 핵심 처리 로직과 최근 업그레이드된 수학적/알고리즘적 개선 사항을 상세히 설명합니다.

---

## Step 1: 정밀 영역 감지 (YOLO Detection)

### 1.1 투트랙 모델 구동 로직 (`detection/yolo_detector.py`)
- **말풍선 감지 (Speech Balloons)**: `models/manga109-speech-bubble-yolo11n.pt` (YOLOv11 최신 아키텍처)
- **글자 마스크 감지 (Text Segmenter)**: `models/manga-text-segmenter-yolov26s.pt` (세그멘테이션 특화)
- **필터링 규칙**:
  - 신뢰도(Confidence threshold): `config.yaml` 기준 `0.20` 이상만 승인.
  - 마스크 후처리: 감지된 다수의 글자 마스크 폴리곤을 하나의 전역 바이너리 마스크(`text_mask`)로 병합하되, 개별 대사 박스(`texts`) 정보는 인페인팅 및 OCR 좌표로 별도 보존합니다.

---

## Step 2: 하이브리드 스마트 클리닝 & 생성형 재구성 (Inpainting & GenAI Replacement)

클리닝 프로세스는 **1단계: 고속 흰색 판별 및 클리닝(`flat_fill`)** ➔ **2단계: 잔여 복잡 영역 로컬 AI 인페인팅 또는 생성형 편집(`inpainting/`)**으로 분기됩니다.

### 2.1 1단계: 파란색 글자 박스 기반 배경 판별 및 고속 클리닝 (`comfy_client.flat_fill`)
기존 방식의 치명적 단점(초록색 말풍선 테두리 노이즈 간섭으로 인한 오판)을 해결하기 위해 개선된 수학적 로직입니다:

1. **내부 샘플링 영역 한정 (Inner ROI Sampling)**:
   - 말풍선 바운딩 박스(`bx1, by1, bx2, by2`) 전체를 검사하지 않고, 해당 말풍선 내부에 포함된 **파란색 글자 바운딩 박스(`texts["bbox"]`)** 영역들만 우선 추출합니다.
   - 글자 박스 내부의 비글자 픽셀(`roi_mask <= 128`)은 만화 컷 테두리나 배경 그림 선화로부터 완전히 격리된 순수 말풍선 내부 배경입니다.
2. **중앙값 통계 판별 (Median Brightness Check)**:
   - 외곽 노이즈에 취약한 산술 평균(`mean`) 대신, 픽셀 밝기의 **중앙값(`median`)**이 임계치(`> 210`)를 초과하는지 검사합니다.
3. **잔상 제로 하이브리드 클리닝 (Hybrid Anti-aliasing Removal)**:
   - 순백색 말풍선으로 판정되면 다음 2가지 연산을 동시에 적용합니다:
     - **마스크 확장 지우기**: 글자 마스크(`roi_mask > 20`)를 `ndimage.binary_dilation(..., iterations=4)`로 주변 4픽셀 확장하여 반투명 회색 안티에일리싱 계단현상 자국을 완전히 포함하여 덮습니다.
     - **글자 박스 통째 클리닝**: 말풍선 내부의 파란색 글자 박스 영역 전체를 흰색(`fill_color`)으로 칠해 미세 잔상을 100% 원천 봉쇄합니다.

### 2.2 2단계: 로컬 AI 인페인팅 (`inpainting/lama_inpainter.py`)
`flat_fill`에서 처리되지 않고 남은 마스크(`remaining_mask`, 예: 스크린톤, 명암, 캐릭터 머리카락 위 글자)에 대해 구동됩니다:

1. **512x512 ROI 패치 분할 (Sliding / Crop Window Execution)**:
   - LaMa ONNX 가중치는 512x512 고정 입력을 요구합니다. 고해상도 전체 원고(2000px 등)를 리사이즈하면 화질이 열화되므로, 남은 마스크의 각 연결 요소(Connected Component) 주변으로 **512x512 크기의 중심 윈도우 크롭**을 생성합니다.
2. **ONNX 추론 및 마스크 정합 블렌딩**:
   - 512x512 크롭 영역만 모델에 넣어 선화와 스크린톤을 정밀 복원한 뒤, 원본 이미지의 해당 마스크 영역(`crop_mask > 0`)에만 정확하게 합성 반환합니다.

### 2.3 [차세대] 생성형 AI 이미지 편집 및 대체 (`inpainting/genai_inpainter.py`)
기존 픽셀 보간(Smoothing) 모델들이 복잡한 일러스트 위의 효과음(SFX)을 지울 때 선화를 뭉개는 한계를 극복하기 위한 생성형 재구성 로직입니다:
1. **크롭 기반 문맥 이해**: 텍스트 바운딩 박스 주변 영역만 크롭하여 생성형 이미지 모델(Nano-Banana, DuctTape2, Flux/SDXL Inpaint 등)에 전달합니다.
2. **프롬프트 기반 재구성**:
   - 지우기 모드: *"Erase the text completely and reconstruct the underlying drawing logically."*
   - 직접 대체 모드: *"Replace the text with Korean '[번역문]' while keeping original art style and dynamic typography aesthetics."*

---

## Step 3 & 4: VLM OCR 및 문맥 인식 번역 (`ocr/vlm_ocr.py`, `translation/translator.py`)
1. **OCR 영역 크롭 및 업스케일**: 감지된 각 글자 바운딩 박스 주변에 여백(`text_mask_margin: 8px`)을 주고 2배 업스케일하여 VLM API로 전송, 일본어 한자/가나/루비 문자를 완벽히 텍스트화합니다.
2. **일괄 문맥 번역 (Context-Aware Translation)**: 대사 목록 전체를 LLM에 한 번에 전달하여 대화 흐름, 존댓말/반말 어조를 유지하며 번역합니다.

---

## Step 5: 만화 전문 식자 합성 (`scripts/render_text.py`)
- 말풍선 가로세로 비율에 맞춰 한국어 단어 단위 자동 줄바꿈(Line wrapping)을 수행하며, 말풍선 중앙에 텍스트가 안착하도록 여백과 줄간격을 자동 계산하여 최종 합성합니다.
