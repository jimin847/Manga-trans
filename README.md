# Manga-trans

일본어 만화 이미지를 감지, OCR, 한국어 번역, 독립 편집 검수, 원문 제거, 식자까지 처리하는 macOS 중심 파이프라인입니다.

이 프로젝트의 우선순위는 단순한 자동 완료율이 아니라 **번역 정확도와 원화 보존**입니다. 확실하지 않은 OCR·번역·리드로우 결과는 성공으로 숨기지 않고 `needs_review`로 남깁니다.

## 현재 상태

| 영역 | 기본 동작 | 상태 |
|---|---|---|
| 감지 | YOLO 말풍선·텍스트 분리 감지, 1280px 추론 | 자동 |
| OCR | tight/context 두 시야를 Gemini로 판독하고 합의 검사 | 자동 + 불일치 검수 |
| 번역 | Gemini 3.1 Pro 초벌, Claude Sonnet 독립 편집 검수 | 자동 + 완결성 검사 |
| 원문 제거 | 평탄한 흰 말풍선만 안전하게 제거 | 자동 |
| 복잡한 배경 | 망점·삽화·외부 글자는 원본 보존 | 수동 리드로우 필요 |
| 식자 | 실제 글꼴 메트릭 기반 크기 탐색과 문장부호 인접 줄바꿈 | 자동 |

현재 회귀 테스트는 `101 passed`입니다. 비공개 3페이지 개발 벤치마크에서는 일반 말풍선 페이지가 완전 통과했고, 복잡한 배경이 포함된 2페이지는 원화 보호 정책에 따라 부분 완료로 판정됩니다.

## 처리 흐름

```text
입력 이미지
  → YOLO 영역 감지
  → tight/context 합의형 OCR
  → 페이지 순서·이전 문맥 기반 번역
  → 독립 편집 검수 및 완결성 검사
  → 원화 보호 게이트
  → PyQt5 자동 식자
  → 번역 이미지 + 영역별 QA 보고서
```

주요 품질 장치:

- OCR 두 시야가 일치하지 않으면 자동 승인하지 않습니다.
- 긴 대사의 누락, 종결 문장부호 손실, 일본어 잔존, 잘못된 혼합 고어체를 검사합니다.
- `…`처럼 번역할 필요가 없는 반응 문장부호는 원본 그대로 보존합니다.
- 번역이 승인되지 않은 영역은 원문을 지우지 않습니다.
- 망점이나 그림 위 텍스트는 검증되지 않은 인페인팅으로 원화를 훼손하지 않습니다.
- 결과마다 `*_dialogue_qa.json`을 만들어 미완료 영역을 명시합니다.

## 요구 사항

- macOS 및 Python 3.10+
- PyQt5, Pillow, PyYAML, NumPy, SciPy, OpenCV, Ultralytics
- Google Antigravity CLI (`agy`)와 로그인된 Google 계정
- 로컬 감지 모델:
  - `models/manga-text-segmenter-yolov26s.pt`
  - `models/manga109-speech-bubble-yolo11n.pt`
- 선택 사항: `models/lama.onnx`

대용량 모델 파일은 Git에 포함하지 않습니다. 모델은 실행 환경에서 별도로 설치하며 `MODELS_ROOT`로 외부 모델 디렉터리를 지정할 수 있습니다.

### Antigravity CLI

```bash
curl -fsSL https://antigravity.google/cli/install.sh | bash
agy
```

기본 `config.yaml`은 로그인된 Antigravity 세션을 사용합니다. provider를 `openrouter` 또는 `google-ai-studio`로 변경한 경우에만 해당 API 키가 필요합니다.

## 실행

단일 페이지:

```bash
python3 main.py '/path/to/page.jpg' --output output
```

여러 페이지:

```bash
python3 run_batch.py --resume --skip-existing '/path/to/chapter/*.jpg'
```

주요 결과물:

- `output/<page>_ko.png`: 최종 이미지
- `output/<page>_dialogue_qa.json`: OCR·번역·렌더링 상태
- `output/cache.json`: 버전이 포함된 OCR·번역 캐시

## 검증

```bash
pytest -q
python3 scripts/quality_benchmark.py \
  --gold benchmarks/yappari_v01/gold.json \
  --results-dir output \
  --source-dir '/path/to/private/source'
```

벤치마크에는 저작권 이미지 대신 파일명, SHA-256, 기대 OCR·번역 조건만 저장합니다. 구현 및 품질 기준은 [전문 품질 파이프라인 문서](docs/research/PROFESSIONAL_QUALITY_PIPELINE.md)에 정리되어 있습니다.

## 알려진 한계

- 망점 말풍선과 삽화 위 세로 대사는 현재 자동 리드로우 대상이 아닙니다.
- LaMa와 단순 knockout은 원화를 훼손하거나 흰 잔상을 만들 수 있어 기본값에서 비활성화되어 있습니다.
- 자동 QA 통과만으로 출판 품질을 보증하지 않습니다. 최종 배포 전 일본어·한국어 검수와 페이지 시각 검수가 필요합니다.

## 저작권

저장소에는 저작권이 있는 만화 원본이나 생성된 번역 이미지가 포함되지 않습니다. 사용자는 처리 대상 콘텐츠의 권리와 해당 지역 법률을 준수해야 합니다.
