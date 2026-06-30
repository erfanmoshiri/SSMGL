"""
NeighAggre (Neighbor Aggregation) Baseline for Graph Attribute Imputation

Simple probabilistic method for message passing using local information measures.
Imputes missing node features by averaging observable neighbor features.

This is a non-learning baseline - no parameters to train.
"""

import numpy as np
import torch
import argparse
import sys
sys.path.insert(0, '../src')
from utils import load_data, RECALL_NDCG, data_split
import time


def train_NeighAggre(adj, true_features, train_id, vali_test_id, args):
    """
    NeighAggre: Mean pooling aggregation of 1-hop neighbor features.

    For each missing node:
    - Average features from observable neighbors
    - If no observable neighbors, use global mean

    Args:
        adj: Sparse adjacency matrix
        true_features: [N, F] tensor with ground truth features
        train_id: Indices of observable nodes (40%)
        vali_test_id: Indices of missing nodes (60%)
        args: Argparse namespace

    Returns:
        imputed_features: [N, F] tensor with imputed values for missing nodes
    """
    device = true_features.device
    num_nodes = true_features.size(0)

    # Convert to dense if sparse
    if adj.is_sparse:
        adj_dense = adj.to_dense()
    else:
        adj_dense = adj

    # Create mask for observable nodes
    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    train_mask[train_id] = True

    # Global mean as fallback
    global_mean = true_features[train_id].mean(dim=0)

    # Initialize output with original features
    imputed_features = true_features.clone()

    # Create masked adjacency: only edges to observable neighbors
    mask_adj = torch.zeros_like(adj_dense)
    mask_adj[vali_test_id, :] = adj_dense[vali_test_id, :]
    mask_adj[:, ~train_mask] = 0  # Mask out edges to missing nodes

    # Aggregate features from observable neighbors
    # aggregation_fts = torch.mm(mask_adj, true_features) / neighbor_count
    neighbor_counts = mask_adj.sum(dim=1, keepdim=True) + 1e-24  # Avoid division by zero
    aggregation_fts = torch.mm(mask_adj, true_features) / neighbor_counts

    # For nodes without observable neighbors, use global mean
    no_neighbors_mask = (mask_adj.sum(dim=1) == 0)
    aggregation_fts[no_neighbors_mask] = global_mean

    # Assign aggregated features to missing nodes
    imputed_features[vali_test_id] = aggregation_fts[vali_test_id]

    print(f'NeighAggre imputation complete. Imputed {len(vali_test_id)} nodes.')
    return imputed_features


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NeighAggre Baseline for Graph Attribute Imputation')
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'amac', 'amap'])
    parser.add_argument('--seed', type=int, default=72)
    parser.add_argument('--train_fts_ratio', type=float, default=0.4)
    parser.add_argument('--generative_flag', type=bool, default=True)
    parser.add_argument('--cuda', action='store_true', default=torch.cuda.is_available())
    parser.add_argument('--device', type=int, default=0)

    args = parser.parse_args()

    # Set device
    if args.cuda:
        torch.cuda.set_device(args.device)
        device = torch.device(f'cuda:{args.device}')
    else:
        device = torch.device('cpu')

    print(f'\n=== NeighAggre Baseline ===')
    print(f'Dataset: {args.dataset}')
    print(f'Device: {device}')

    # Load data
    adj, diff, norm_adj, true_features, node_labels, indices = load_data(args)
    train_id, vali_id, test_id, vali_test_id = data_split(args, adj)

    if args.cuda:
        adj = adj.cuda()
        true_features = true_features.cuda()
        train_id = train_id.cuda()
        vali_test_id = vali_test_id.cuda()
        test_id = test_id.cuda()

    # Run NeighAggre imputation
    start_time = time.time()
    imputed_features = train_NeighAggre(adj, true_features, train_id, vali_test_id, args)
    train_time = time.time() - start_time

    print(f'\nTraining time: {train_time:.2f}s')

    # Evaluate on test set only
    test_imputed = imputed_features[test_id].cpu().numpy()
    test_true = true_features[test_id].cpu().numpy()

    print(f'\n=== Evaluation Results ===')
    for topK in [10, 20, 50]:
        recall, ndcg = RECALL_NDCG(test_imputed, test_true, topN=topK)
        print(f'Recall@{topK}: {recall:.4f}, NDCG@{topK}: {ndcg:.4f}')
