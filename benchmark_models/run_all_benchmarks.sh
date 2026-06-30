#!/bin/bash
#
# Run all benchmark models on all datasets
#
# Usage: ./run_all_benchmarks.sh [device_id]
#

DEVICE=${1:-0}
DATASETS="cora citeseer amac amap"
MODELS="KNN NeighAggre VAE GraphSAGE GAT SAT SVGA GraphRNA ARWMF MATE"

echo "=========================================="
echo "Running All Benchmarks"
echo "=========================================="
echo "Device: GPU $DEVICE"
echo "Datasets: $DATASETS"
echo "Models: $MODELS"
echo "=========================================="

# Create results directory
mkdir -p results
mkdir -p logs

# Activate virtual environment if it exists
if [ -d "../venv" ]; then
    source ../venv/bin/activate
fi

# Run each model on each dataset
for dataset in $DATASETS; do
    echo ""
    echo "=========================================="
    echo "Dataset: $dataset"
    echo "=========================================="

    for model in $MODELS; do
        echo ""
        echo "------------------------------------------"
        echo "Running: $model on $dataset"
        echo "------------------------------------------"

        LOG_FILE="logs/${model}_${dataset}_$(date +%Y%m%d_%H%M%S).log"

        python main_benchmark.py \
            --model $model \
            --dataset $dataset \
            --device $DEVICE \
            --seed 72 \
            --save_results \
            2>&1 | tee "$LOG_FILE"

        EXIT_CODE=${PIPESTATUS[0]}

        if [ $EXIT_CODE -eq 0 ]; then
            echo "✓ $model on $dataset completed successfully"
        else
            echo "✗ $model on $dataset failed with exit code $EXIT_CODE"
        fi
    done
done

echo ""
echo "=========================================="
echo "All Benchmarks Completed!"
echo "=========================================="
echo "Results saved to: results/"
echo "Logs saved to: logs/"
