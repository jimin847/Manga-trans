#!/bin/bash
set -e

API_KEY="${OPENROUTER_API_KEY:-your_openrouter_api_key_here}"
CONFIG="/Users/bagjimin/Downloads/manga-trans-config.json"
INPUT_BASE="/Users/bagjimin/Downloads/machine"
OUTPUT_BASE="/Users/bagjimin/Downloads/Machine-trans"

for vol_dir in "$INPUT_BASE"/Machinetosoware-*; do
    vol_name=$(basename "$vol_dir")
    vol_num=$(echo "$vol_name" | grep -oE '[0-9]+$')
    
    # 5권부터만
    [ "$vol_num" -lt 5 ] 2>/dev/null && continue
    
    output_dir="$OUTPUT_BASE/$vol_num"
    mkdir -p "$output_dir"
    
    # 이미 번역된 파일 수 확인
    translated=$(find "$output_dir" -name "*.jpg" 2>/dev/null | wc -l)
    total=$(find "$vol_dir" -name "*.jpg" -o -name "*.png" 2>/dev/null | wc -l)
    
    echo "=== $vol_name: $translated/$total 완료 ==="
    
    if [ "$translated" -ge "$total" ] 2>/dev/null; then
        echo "스킵 (이미 완료)"
        continue
    fi
    
    echo "번역 시작: $vol_name"
    
    docker run -d --name "manga-vol-$vol_num" \
      -v "$vol_dir:/app/input" \
      -v "$output_dir:/app/result" \
      -v "$CONFIG:/app/config.json:ro" \
      -e CUSTOM_OPENAI_API_KEY="$API_KEY" \
      -e CUSTOM_OPENAI_API_BASE="https://openrouter.ai/api/v1" \
      -e CUSTOM_OPENAI_MODEL="nvidia/nemotron-3-super-120b-a12b:free" \
      --entrypoint="" \
      zyddnys/manga-image-translator:main \
      python -m manga_translator local \
        -i /app/input \
        -o /app/result \
        --config-file /app/config.json \
        --skip-no-text
    
    # 완료 대기
    echo "대기 중..."
    while docker ps -q --filter "name=manga-vol-$vol_num" | grep -q .; do
        sleep 10
    done
    
    echo "완료: $vol_name"
    docker rm "manga-vol-$vol_num" 2>/dev/null
done

echo "=== 전체 번역 완료 ==="
