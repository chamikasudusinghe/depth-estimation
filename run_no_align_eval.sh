#!/usr/bin/env bash
set -euo pipefail

DATASET_DIR="datasets/quest"
SESSIONS=(
    "20260421_191742"
    "20260421_202107"
    "20260421_202359"
)
METRIC_MODELS=(
    "apple/DepthPro-hf"
    "depth-anything/DA3METRIC-LARGE"
)

cd "$(dirname "$0")"

for session in "${SESSIONS[@]}"; do
    for model in "${METRIC_MODELS[@]}"; do
        echo "=== no-align eval | $session | $model ==="
        python eval_quest.py \
            --dataset-dir "$DATASET_DIR/$session" \
            --model "$model" \
            --no-align
    done
done

echo "Done."
