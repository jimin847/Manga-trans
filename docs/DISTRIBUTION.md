# Manga-trans 데스크탑 애플리케이션 배포 및 패키징 가이드

본 문서는 개발된 PyQt5 기반 데스크탑 애플리케이션(`desktop_app.py`)을 파이썬 비개발자(일반 사용자)도 더블 클릭만으로 바로 사용할 수 있는 **독립 소프트웨어 패키지(`.app` / `.exe`)로 빌드하고 배포하는 방법**을 안내합니다.

---

## 🍏 1. macOS 애플리케이션 번들 (`.app`) 빌드 방법

### 환경 준비
- Python 3.10 이상 및 `pyinstaller` 설치 필요
  ```bash
  pip install pyinstaller
  ```

### 자동 빌드 실행
프로젝트 루트 디렉토리에서 원클릭 빌드 스크립트를 실행합니다:
```bash
./build_mac_app.sh
```

### 빌드 결과물
- `dist/Manga-trans.app`: 파이썬 인터프리터, PyQt5, 파이프라인 모듈 및 설정 파일(`config.yaml`, `workflows/`)이 일체화된 독립 데스크탑 앱 번들입니다.
- **실행 방법**: Finder에서 `dist/Manga-trans.app`을 더블 클릭하여 바로 실행할 수 있으며, 이 파일을 `/Applications`(응용 프로그램) 폴더로 복사하거나 DMG로 압축하여 다른 Mac 사용자에게 배포할 수 있습니다.

---

## 🪟 2. Windows 실행 파일 (`.exe`) 빌드 방법

Windows OS 환경(또는 가상 머신)에서 다음 명령어를 실행하여 단일 실행 폴더 또는 `.exe` 파일을 생성할 수 있습니다:

```cmd
pyinstaller --clean Manga-trans.spec
```
* 생성된 `dist\Manga-trans\` 폴더 내의 `Manga-trans.exe`를 실행하면 콘솔 창 없이 GUI 대시보드가 오픈됩니다.

---

## 💡 배포 시 필수 안내 사항 (최종 사용자용 가이드)

본 프로그램은 엔드투엔드 만화 번역 파이프라인으로, 앱 실행 후 실제 번역을 수행하려면 다음 외부 서비스 연동이 필요합니다.

1. **OpenRouter API Key 설정**
   - 사용자 계정 환경 변수(`OPENROUTER_API_KEY`)가 설정되어 있어야 LLM 및 VLM 번역 모델을 호출할 수 있습니다.
   - 앱 상단 헤더 바의 상태 인디케이터에서 `● OpenRouter API [설정됨]` 여부를 실시간으로 확인할 수 있습니다.

2. **ComfyUI 인페인팅 백엔드 실행**
   - 고품질 배경 복원(LaMa 인페인팅)을 위해 로컬 또는 원격 ComfyUI 서버(`port 8188`)가 구동 중이어야 합니다.
   - 앱 상단 헤더 바에서 `● ComfyUI [정상 (8188)]` 초록색 신호등이 켜져 있는지 확인 후 배치 작업을 시작하세요.
