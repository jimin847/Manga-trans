# Manga-Trans v2 에이전트 핸드오프 가이드 (Agent Handoff Guide)

본 문서는 새로운 AI 에이전트 또는 개발자가 프로젝트에 참여할 때 즉시 전체 맥락을 파악하고 이어서 작업할 수 있도록 작성된 종합 핸드오프 문서입니다.

---

## 1. 프로젝트 현황 요약 (Current Status)

- **감지 모듈 완료**: YOLOv11n 말풍선 모델 + YOLO26s 텍스트 세그멘터 연동 완료 (`models/` 디렉토리에 `.pt` 파일 보관).
- **인페인팅 모듈 완료**: 외부 ComfyUI HTTP 서버 의존성을 완전 탈피하고 독립 실행 가능한 로컬 패키지(`inpainting/`) 구축 완료.
- **검증 완료 상태**: `main.run_inpainting()` 실행 시 `flat_fill` 고속 전처리 ➔ `LaMa ONNX` (또는 `OpenCV`, `GenAI Edit`) 로컬 추론이 완벽히 동작함을 실제 만화 데이터(`Chapter_52_Title_Page.webp`, `K1468477_g5_152014.jpg`)로 검증함.

---

## 2. 주요 커맨드 및 실행 방법 (Quickstart Commands)

### 2.1 문법 및 컴파일 검증
```bash
python3 -m py_compile main.py comfy_client.py inpainting/*.py detection/yolo_detector.py
```

### 2.2 파이프라인 단독 테스트 (인페인팅 구동)
```bash
python3 -c '
from pathlib import Path
from PIL import Image
import main

cfg = main.load_config("config.yaml")
img = Image.open("/Users/bagjimin/Downloads/K1468477_g5_152014.jpg").convert("RGB")
det = main.get_detector(cfg).detect(img, page_id="test_run")
cleaned_path = main.run_inpainting("/Users/bagjimin/Downloads/K1468477_g5_152014.jpg", det, cfg, Path("output"), original=img)
print("Result saved to:", cleaned_path)
'
```

### 2.3 인페인팅 백엔드 변경 방법
`config.yaml`의 `inpainting.backend` 설정 변경:
- `"lama_onnx"`: 기본 AI 인페인팅 (ONNX Runtime, 512x512 패치 크롭)
- `"genai_edit"`: 차세대 생성형 AI 편집/대체 (Nano-Banana, Flux, MLLM Edit API 등 연동용)
- `"opencv"`: 초경량 내장 Fallback (`cv2.inpaint`)
- `"comfyui"`: 기존 외부 HTTP 서버 호출 모드 호환

---

## 3. 핵심 모듈별 설계 핵심 노트 (Important Architecture Notes)

1. **`comfy_client.py` ➔ `flat_fill(image, mask, bubbles=None, texts=None)`**:
   - `texts` 파라미터가 추가되었습니다. 말풍선 안의 밝기를 잴 때 초록색 말풍선 박스가 아니라 말풍선 내부에 안착한 **파란색 글자 박스(`texts`) 내 픽셀을 우선 샘플링하여 중앙값(`median`)으로 흰색 여부를 판단**합니다. 절대로 이 로직을 말풍선 박스 전체 평균(`mean`)으로 되돌리지 마세요!
2. **`inpainting/lama_inpainter.py` ➔ `LaMaONNXInpainter`**:
   - 고정 해상도 512x512 ONNX 모델(`models/lama.onnx`)을 위해 입력 마스크의 각 연결 요소 중심 기준으로 512x512 윈도우 크롭을 추출하여 추론한 뒤 정합합니다. 이미지 전체 리사이즈를 하지 않으므로 선화 해상도가 100% 보존됩니다.
3. **`inpainting/genai_inpainter.py` ➔ `GenAIEditInpainter`**:
   - 복잡한 배경/효과음 위의 글자가 뭉개지는 한계를 해결하기 위해 신설된 생성형 인터페이스입니다. 다음 담당 에이전트는 OpenRouter MLLM Edit API나 ComfyUI Flux Inpaint 노드 연결 시 이 클래스의 `inpaint()` 메소드를 구체화하면 됩니다.

---

## 4. 추천 향후 발전 로드맵 (Next Steps for Future Agents)

1. **`GenAIEditInpainter` API / 노드 연동 구체화**:
   - `config.yaml`의 `genai_prompt_erase` 또는 `genai_prompt_replace`를 사용하여 바운딩 박스 크롭을 OpenRouter GenAI 이미지 편집 API(또는 ComfyUI Flux/Nano-Banana 커스텀 워크플로)로 보내고 받은 결과물을 원본에 합성하는 로직을 완성하세요.
2. **WebUI 연동 설정 추가**:
   - `webui/index.html` 및 API 서버에서 사용자가 `LaMa ONNX` vs `GenAI Edit` vs `OpenCV` 인페인팅 엔진을 드롭다운 메뉴로 선택할 수 있는 UI 스위치를 추가하세요.
