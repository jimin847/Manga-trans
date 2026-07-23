#!/usr/bin/env bash
#
# build_mac_app.sh — macOS 전용 Manga-trans 데스크탑 애플리케이션 빌드 스크립트
#
set -e

echo "======================================================="
echo "  🌙 Manga-trans Desktop App Build Script (PyInstaller)"
echo "======================================================="

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "🧹 1. 이전 빌드 캐시 (build/, dist/) 정리 중..."
rm -rf build/ dist/Manga-trans*

echo "⚙️ 2. PyInstaller로 데스크탑 번들 빌드 시작..."
pyinstaller --clean Manga-trans.spec

echo "✅ 빌드 완료!"
if [ -d "dist/Manga-trans.app" ]; then
    echo "📦 애플리케이션 번들 생성 위치: dist/Manga-trans.app"
    echo "💡 Finder에서 'dist/' 폴더로 이동하여 'Manga-trans.app'을 더블 클릭해 바로 실행할 수 있습니다."
else
    echo "❌ 오류: dist/Manga-trans.app 번들이 생성되지 않았습니다."
    exit 1
fi
