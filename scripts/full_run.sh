#!/bin/bash
# Full batch run: all 4 models x all H/L combos (skips H16/L1, already done)
# Logs go to logs/<model>_H<H>_L<L>.log
# Existing result dirs are skipped automatically (no --overwrite)
set -e

cd "$(dirname "$0")/.."
mkdir -p logs

MODELS=(
    #gaussian_tanh
    #regularized_horseshoe_tanh
    dirichlet_horseshoe_tanh_nodewise
    #dirichlet_student_t_tanh_nodewise
)

HL_COMBOS=(
    "16 1"
    # "16 2"
    # "16 3"
    "32 1"
    # "32 2"
    # "32 3"
    # "64 1"
    # "64 2"
    # "64 3"
)

for model in "${MODELS[@]}"; do
    for hl in "${HL_COMBOS[@]}"; do
        H=$(echo $hl | awk '{print $1}')
        L=$(echo $hl | awk '{print $2}')
        logfile="logs/${model}_H${H}_L${L}.log"
        echo "=== Starting: $model H=$H L=$L ===" | tee -a "$logfile"
        python3.11 utils/run_models.py \
        --task regression \
        --model "$model" \
        --H "$H" --L "$L" \
        --warmup 2000 --sample 2000 \
        --overwrite \
        --output_dir results_new \
        >> "$logfile" 2>&1
        echo "=== Done: $model H=$H L=$L ===" | tee -a "$logfile"
    done
done

echo "=== All runs complete ==="
