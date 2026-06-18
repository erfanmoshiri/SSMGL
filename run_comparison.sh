#!/bin/bash

# Comparison experiment: Learnable vs Fixed features
# Run both variants on Cora dataset

echo "=========================================="
echo "EXPERIMENT: Learnable vs Fixed Features"
echo "Dataset: Cora"
echo "Epochs: 3000"
echo "=========================================="
echo ""

# Activate virtual environment
source venv/bin/activate

echo ">>> Running ORIGINAL (Learnable features)..."
echo "-------------------------------------------"
python main.py --dataset citeseer --device 0
echo ""
echo ">>> ORIGINAL completed!"
echo ""

echo ">>> Running FIXED features variant..."
echo "-------------------------------------------"
python main_fixed.py --dataset citeseer --device 0
echo ""
echo ">>> FIXED completed!"
echo ""

echo "=========================================="
echo "COMPARISON COMPLETE"
echo "=========================================="
echo ""
echo "Check terminal output above for metrics comparison!"
