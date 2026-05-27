#!/bin/bash
# Smoke test: 1 model, 1 dataset, very few samples -> results_test/
set -e

cd "$(dirname "$0")/.."

echo "=== Smoke test starting ==="
python3.11 utils/run_models.py \
    --task regression \
    --model gaussian_tanh \
    --H 16 --L 1 \
    --warmup 50 --sample 50 \
    --N 100 --limit 1 \
    --output_dir results_test

echo "=== Smoke test PASSED ==="
echo "Output in: results_test/gaussian_tanh_H16_L1/"
