# Multi-view Graph Imputation Network (MATE)

Implementation of "Multi-view Graph Imputation Network" (Information Fusion 2024).

Paper: [Multi-view graph imputation network](https://github.com/XinPeng97/MATE)

---

## Project Structure

```
MATE/
├── models/              # Model architectures
│   ├── MATE.py         # Original learnable parameters version
│   └── MATE_fixed.py   # Fixed initialization baseline
├── runs/               # Training runs with timestamps
│   └── {dataset}_{model}_{timestamp}/
│       ├── config.json
│       ├── training_log.jsonl
│       ├── final_results.json
│       ├── model_best.pkl
│       └── loss_history.pkl
├── data/               # Datasets (Cora, Citeseer, etc.)
├── main.py            # Training script (learnable)
├── main_fixed.py      # Training script (fixed baseline)
├── utils.py           # Data loading and evaluation
├── loss.py            # Loss functions
└── configs.yml        # Hyperparameters per dataset
```

---

## Setup

### Requirements
```bash
python >= 3.8
torch >= 2.0.0
torch-geometric >= 2.3.0
scipy, scikit-learn, networkx, PyYAML, tqdm
```

### Installation
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Quick Start

### Train Original MATE (Learnable Features)
```bash
source venv/bin/activate
python main.py --dataset cora --device 0
```

### Train Fixed Baseline (Neighbor Averaging)
```bash
python main_fixed.py --dataset cora --device 0
```

### Run Comparison
```bash
./run_comparison.sh  # Runs both variants sequentially
```

---

## Method Overview

### Problem Setup
- **Input**: Graph with missing node features (60% nodes have features set to 0)
- **Goal**: Impute missing node features using graph structure + observable features

### MATE Approach
1. **Multi-view Generation**:
   - View 1: Learnable parameters + message passing
   - View 2: PPR diffusion

2. **Dual Constraint Strategy (DCS)**:
   - Contrastive loss: Node consistency across views
   - Structure loss: Edge reconstruction across views

3. **Training**:
   - Reconstruction loss on observable nodes (40%)
   - Learnable parameters optimized via backprop

### Fixed Baseline
- Replaces learnable parameters with **neighbor averaging**
- Uses full graph structure, only observable node features
- Parameters frozen (only GNN/decoder learn)

---

## Experimental Findings

### Key Insight
Fixed initialization (neighbor averaging) performs **comparably to learnable parameters** on citation networks (Cora, Citeseer).

**Implications**:
- Strong structural homophily → neighbor averaging is highly effective
- Multi-view architecture + DCS provide most benefits
- Learning provides marginal gains over smart initialization (for this task/domain)

### Results (Citeseer, 60% missing)

| Method | Recall@50 | NDCG@50 | Acc (X) | Acc (A+X) |
|--------|-----------|---------|---------|-----------|
| Learnable | 0.272 | 0.294 | 0.689 | 0.693 |
| Fixed | 0.272 | 0.296 | 0.671 | 0.681 |

---

## Configuration

Hyperparameters in `configs.yml`:

```yaml
cora:
  encoder_channels: 512
  hidden_channels: 256
  encoder_layers: 2
  temp: 0.2           # Contrastive temperature
  p: 0.8              # Edge masking probability
  epoch: 3000
  lr: 0.001
```

---

## Datasets

Supported: `cora`, `citeseer`, `amac` (Amazon Computers), `amap` (Amazon Photo)

Data split:
- 40% training (observable features)
- 10% validation
- 50% test (imputation targets)

---

## Evaluation Metrics

### Feature Imputation Quality
- **Recall@K**: Fraction of ground-truth features in top-K predictions
- **NDCG@K**: Normalized discounted cumulative gain

### Downstream Tasks
- **Classification (X)**: Using only imputed features
- **Classification (A+X)**: Using features + graph structure (GCN)

---

## Citation

```bibtex
@article{peng2024multi,
  title={Multi-view graph imputation network},
  author={Peng, Xin and Cheng, Jieren and Tang, Xiangyan and Zhang, Bin and Tu, Wenxuan},
  journal={Information Fusion},
  volume={102},
  pages={102024},
  year={2024},
  publisher={Elsevier}
}
```

---

## Notes

- GPU strongly recommended (training takes ~1-2 minutes per dataset on RTX 6000)
- Edge masking is training augmentation, not part of imputation problem
- Fixed baseline uses realistic setup: full graph structure, partial features
- Run folders automatically timestamped in `runs/`



Evaluation baselines to test against:
- KNN
- NeighbourAggr: a simple approach that aggregates the features of neighboring nodes through mean pooling
- VAE
- GraphSAGE
- GAT
- GraphRNA
- SVGA 
- ARWMF: Attributed Random Walk as Matrix Factorization
- ITR ?
- SAT
- MATE
