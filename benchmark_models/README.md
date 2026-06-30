# Benchmark Models for Graph Attribute Imputation

10 baseline models for comparing against MATE on attribute-missing graphs (60% missing features).

## Quick Start

```bash
# Single model
python main_benchmark.py --model SAT --dataset cora --device 0

# All models on all datasets
./run_all_benchmarks.sh 0

# View results
python summarize_results.py
```

## Models

| Model | Type | Init Strategy | Time (Cora) |
|-------|------|---------------|-------------|
| KNN | Traditional | Direct imputation | ~5s |
| NeighAggre | Traditional | Direct aggregation | ~0.2s |
| VAE | Generative | Neighbor aggregation | ~5min |
| GraphSAGE | GNN | Zero init | ~1min |
| GAT | GNN | Zero init | ~1min |
| SAT | GAE | Zero init | ~2min |
| SVGA | GAE | Structure-only | ~10min |
| GraphRNA | Random walk | Node embeddings | ~8min |
| ARWMF | Matrix factorization | Node embeddings | ~5min |
| **MATE** | Multi-view | Zero init + learned features | ~2min |

## Key Hyperparameters

**All models use paper-based defaults.** Override only when needed:

```bash
# Reduce epochs for speed
python main_benchmark.py --model SAT --dataset cora --epochs 200 --device 0

# Reduce memory usage
python main_benchmark.py --model GAT --dataset cora --hidden 128 --device 0
```

**Default hyperparameters** (from papers/original code):
- **SAT**: hidden=64, lambda_cross=10.0, epochs=1000
- **SVGA**: hidden=256, lamda=1.0, beta=1.0, lr=0.001, epochs=2000
- **GraphSAGE**: hidden=256, lr=0.01, epochs=200
- **GAT**: hidden=256, heads=8, dropout=0.6, epochs=200
- **VAE**: hidden=64, lr=0.005, epochs=1000
- **ITR**: hidden=128, refine_iters=2, epochs=100 (optimized for speed)
- **GraphRNA**: embedding=128, walks=20, walk_len=10, epochs=100
- **ARWMF**: embedding=128, walks=40, window=5, lr=0.025, epochs=100
- **MATE**: encoder=128, hidden=64, lr=0.001, temp=0.2, epochs=1000

## Missing Node Initialization

- **KNN, NeighAggre**: Direct imputation from observable neighbors
- **VAE**: Neighbor aggregation before training
- **GraphSAGE, GAT, SAT**: Zero initialization (GNN learns from structure)
- **SVGA**: Identity/diagonal features (structure-only)
- **ITR**: Structure encoder (learns from graph only)
- **GraphRNA, ARWMF**: Random initialized node embeddings
- **MATE**: Zero init + learnable feature matrix

## Evaluation Metrics

- **Recall@K**: Fraction of true features in top-K predictions (K=10,20,50)
- **NDCG@K**: Normalized Discounted Cumulative Gain (ranking quality)

Metrics computed on:
- **Validation set**: 10% of nodes (for tuning)
- **Test set**: 50% of nodes (for final comparison)

## Expected Performance (Cora, 60% missing)

| Model | Recall@50 | NDCG@50 |
|-------|-----------|---------|
| NeighAggre | 0.275 | 0.257 |
| GraphSAGE | 0.259 | 0.263 |
| ITR | 0.284 | 0.261 |
| SAT | ~0.34 | ~0.32 |
| SVGA | ~0.37 | ~0.35 |

## Output

Results saved to `results/MODEL_DATASET_seed72.json`:
```json
{
  "model": "SAT",
  "dataset": "cora",
  "recall@50": 0.3429,
  "ndcg@50": 0.3212,
  "train_time": 120.5
}
```

Summary: `python summarize_results.py` → `results/summary.csv`

## Troubleshooting

**CUDA OOM**: Reduce `--hidden` or `--batch_size`  
**Slow training**: Reduce `--epochs`  
**ITR too slow**: Use `--refine_iterations 1` or `--epochs 50`

## References

Papers in `/home/erfan/MATE/papers/benchmarks/`:
- SAT (TPAMI 2020), SVGA (KDD 2022), ITR (IJCAI 2022)
- GraphRNA (KDD 2019), ARWMF (NeurIPS Workshop 2019)
- GraphSAGE (NeurIPS 2017), GAT (ICLR 2018)
