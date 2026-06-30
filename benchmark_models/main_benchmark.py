"""
Unified Benchmark Runner for Graph Attribute Imputation

Runs any baseline model with standardized evaluation.

Usage:
    python main_benchmark.py --model SAT --dataset cora --device 0
"""

import argparse
import sys
import time
import json
import os
from datetime import datetime

sys.path.insert(0, '../src')
from utils import load_data, RECALL_NDCG, data_split

# Import all model training functions
from KNN import train_KNN
from NeighAggre import train_NeighAggre
from VAE import train_VAE
from GraphSAGE import train_GraphSAGE
from GAT import train_GAT
from SAT import train_SAT
from SVGA import train_SVGA
from ITR import train_ITR
from GraphRNA import train_GraphRNA
from ARWMF import train_ARWMF
from MATE import train_MATE


# Model registry
MODELS = {
    'KNN': train_KNN,
    'NeighAggre': train_NeighAggre,
    'VAE': train_VAE,
    'GraphSAGE': train_GraphSAGE,
    'GAT': train_GAT,
    'SAT': train_SAT,
    'SVGA': train_SVGA,
    'ITR': train_ITR,
    'GraphRNA': train_GraphRNA,
    'ARWMF': train_ARWMF,
    'MATE': train_MATE,
}


def save_results(results, output_dir='results'):
    """Save results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{results['model']}_{results['dataset']}_seed{results['seed']}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\nResults saved to: {filepath}')


def main():
    parser = argparse.ArgumentParser(description='Unified Benchmark Runner')

    # Model and dataset
    parser.add_argument('--model', type=str, required=True, choices=MODELS.keys(),
                        help='Model to run')
    parser.add_argument('--dataset', type=str, default='cora',
                        choices=['cora', 'citeseer', 'amac', 'amap'],
                        help='Dataset to use')

    # Common hyperparameters
    parser.add_argument('--seed', type=int, default=72)
    parser.add_argument('--train_fts_ratio', type=float, default=0.4)
    parser.add_argument('--generative_flag', type=bool, default=True)
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--device', type=int, default=0)

    # Model-specific hyperparameters (with defaults)
    # These will be used if the model needs them
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=100)

    # KNN specific
    parser.add_argument('--K', type=int, default=3)

    # GAT specific
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--alpha', type=float, default=0.2)

    # SAT specific
    parser.add_argument('--lambda_recon', type=float, default=1.0)
    parser.add_argument('--lambda_cross', type=float, default=10.0)
    parser.add_argument('--lambda_gan', type=float, default=1.0)
    parser.add_argument('--n_gene', type=int, default=2)
    parser.add_argument('--n_disc', type=int, default=1)
    parser.add_argument('--enc_name', type=str, default='GCN')

    # SVGA specific
    parser.add_argument('--lamda', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0)

    # ITR specific
    parser.add_argument('--refine_iterations', type=int, default=3)

    # GraphRNA specific
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--walk_length', type=int, default=10)
    parser.add_argument('--walks_per_node', type=int, default=20)

    # ARWMF specific
    parser.add_argument('--num_walks', type=int, default=40)
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--num_negative', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=256)

    # Output
    parser.add_argument('--save_results', action='store_true', default=True)
    parser.add_argument('--output_dir', type=str, default='results')

    args = parser.parse_args()

    # Set device
    if args.cuda:
        import torch
        torch.cuda.set_device(args.device)
        device = f'cuda:{args.device}'
    else:
        device = 'cpu'

    # Set num_classes for GraphSAGE/GAT
    num_classes_map = {'cora': 7, 'citeseer': 6, 'amac': 10, 'amap': 8}
    args.num_classes = num_classes_map.get(args.dataset, 7)

    print('=' * 70)
    print(f'Benchmark Runner - {args.model} on {args.dataset}'.center(70))
    print('=' * 70)
    print(f'Device: {device}')
    print(f'Seed: {args.seed}')
    print(f'Train ratio: {args.train_fts_ratio}')
    print('-' * 70)

    # Load data
    print('Loading data...')
    adj, diff, norm_adj, true_features, node_labels, indices = load_data(args)
    train_id, vali_id, test_id, vali_test_id = data_split(args, adj)

    if args.cuda:
        import torch
        adj = adj.cuda()
        true_features = true_features.cuda()
        train_id = train_id.cuda()
        vali_test_id = vali_test_id.cuda()
        test_id = test_id.cuda()
        vali_id = vali_id.cuda()

    print(f'Nodes: {true_features.size(0)}, Features: {true_features.size(1)}')
    print(f'Train: {len(train_id)}, Val: {len(vali_id)}, Test: {len(test_id)}')
    print('-' * 70)

    # Get training function
    train_fn = MODELS[args.model]

    # Train model
    print(f'\nTraining {args.model}...\n')
    start_time = time.time()

    # MATE needs separate vali_id and test_id for validation during training
    if args.model == 'MATE':
        imputed_features = train_fn(adj, true_features, train_id, vali_id, test_id, vali_test_id, args)
    else:
        imputed_features = train_fn(adj, true_features, train_id, vali_test_id, args)

    train_time = time.time() - start_time

    print('\n' + '-' * 70)
    print(f'Training completed in {train_time:.2f}s')
    print('-' * 70)

    # Evaluate on test set
    print('\n=== Test Set Evaluation ===')
    test_imputed = imputed_features[test_id].cpu().numpy()
    test_true = true_features[test_id].cpu().numpy()

    results = {
        'model': args.model,
        'dataset': args.dataset,
        'seed': args.seed,
        'train_time': train_time,
        'timestamp': datetime.now().isoformat(),
    }

    for topK in [10, 20, 50]:
        recall, ndcg = RECALL_NDCG(test_imputed, test_true, topN=topK)
        results[f'recall@{topK}'] = recall
        results[f'ndcg@{topK}'] = ndcg
        print(f'Recall@{topK}: {recall:.4f}, NDCG@{topK}: {ndcg:.4f}')

    # Evaluate on validation set (for completeness)
    print('\n=== Validation Set Evaluation ===')
    vali_imputed = imputed_features[vali_id].cpu().numpy()
    vali_true = true_features[vali_id].cpu().numpy()

    for topK in [10, 20, 50]:
        recall, ndcg = RECALL_NDCG(vali_imputed, vali_true, topN=topK)
        results[f'val_recall@{topK}'] = recall
        results[f'val_ndcg@{topK}'] = ndcg
        print(f'Recall@{topK}: {recall:.4f}, NDCG@{topK}: {ndcg:.4f}')

    # Save results
    if args.save_results:
        save_results(results, args.output_dir)

    print('\n' + '=' * 70)
    print('Benchmark completed successfully!'.center(70))
    print('=' * 70)


if __name__ == '__main__':
    main()
