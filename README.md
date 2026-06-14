# Manga-trans (v2)

일본어 만화(망가) 이미지 파일을 한국어로 자동 번역하고 식자(Typeset)하는 엔드투엔드(End-to-End) 자동화 파이프라인 프로젝트입니다.

> 💡 **프로젝트 안내**  
> 본 프로젝트는 개발용 AI 에이전트인 **Hermes-agent**와의 긴밀한 협업을 통해 제작되었습니다. 전체 아키텍처 기획 및 요구 사항(사양서)을 정의한 뒤, AI 에이전트가 코드를 빌드하고 디버깅하는 과정을 거쳐 완성되었습니다.

---

## 🛠️ 핵심 파이프라인 아키텍처

본 프로젝트는 다음과 같은 유기적인 흐름으로 만화 원서 이미지를 자동 번역합니다.

```
[입력 이미지]
      │
      ▼
1. YOLO 감지 ───────► (텍스트 영역 및 말풍선 Bbox 좌표 검출)
      │
      ▼
2. 하이브리드 지우기  ► (일반 흰 배경: Flat Fill 처리)
      │               (스크린톤/복잡 배경: ComfyUI LaMa Inpainting API 호출)
      ▼
3. VLM 병렬 OCR ────► (YOLO로 크롭된 영역을 OpenRouter Qwen3-VL에 보내 4병렬 OCR 추출)
      │
      ▼
4. 문맥 기반 번역 ──► (전후 페이지 흐름을 담아 OpenRouter LLM(GPT-120B / Gemini)으로 번역)
      │
      ▼
5. PyQt5 자동 식자 ──► (QPainter 및 QFontMetrics를 활용, 이진 탐색으로 폰트 크기 최적화 배치)
      │
      ▼
[최종 번역 완료 이미지]
```

---

## 🌟 주요 특징 (v2 개선 사항)

* **하이브리드 인페인팅**: 연산 효율을 위해 일반 말풍선은 로컬 수학 연산(Flat fill)으로 즉시 채우고, 복잡한 배경에만 무거운 AI 이미지 인페인팅(LaMa) 모델을 호출하도록 구성하여 처리 속도를 높였습니다.
* **CacheManager 탑재**: 동일 이미지 크롭 해시(MD5)와 번역 텍스트 해시를 관리하여 중복되는 API 요청을 최소화하고 요금을 최적화합니다.
* **문맥 유지(Context Continuity)**: 배치 처리 시 이전 페이지 대사의 문맥을 큐(Queue) 형식으로 유지하여 끊김 없이 매끄러운 번역을 수행합니다.
* **지능형 자동 식자 엔진**: 말풍선 Bbox 너비와 높이에 맞춰 줄바꿈을 포함한 최적의 폰트 크기(22~72pt)를 이진 탐색 알고리즘으로 자동 계산하여 최적의 가독성을 구현합니다.
* **중단 없는 안정적 배치**: `--resume` 및 `--skip-existing` 플래그를 통해 API 할당량 초과 등으로 인한 작업 중단 시 기존 성공 지점부터 다시 진행할 수 있습니다.

---

## ⚙️ 시스템 요구사항 및 구성

* **Language**: Python 3.10+
* **Dependencies**: PyQt5, Pillow, PyYAML, Scipy, Ultralytics (YOLO)
* **External Services**: 
  - **ComfyUI Server** (LaMa Inpainting REST API, default `port 8188`)
  - **OpenRouter API Key** (Gemma, Qwen, GPT-OSS 등 무료/유료 VLM 및 LLM 호출용)

---

## 🚀 실행 방법

### 1. 단일 페이지 실행
```bash
python3 main.py [입력이미지경로] --output [결과물출력디렉토리]
```

### 2. 폴더 내 이미지 일괄 배치 실행 (Docker 기반)
`translate_all.sh` 스크립트를 사용하여 대량의 폴더를 일괄 처리할 수 있습니다.

```bash
# OpenRouter API 키를 환경 변수로 등록 후 실행
export OPENROUTER_API_KEY="your_api_key_here"
bash translate_all.sh
```

---

## ⚖️ 저작권 및 면책 조항
본 저장소에는 저작권이 있는 만화 본문 이미지 및 번역 결과물은 포함되어 있지 않으며, 오직 이미지 처리 및 파이프라인 자동화 실행 코드만 제공됩니다.
