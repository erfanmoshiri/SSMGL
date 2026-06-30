"""
K-Nearest Neighbors (KNN) Baseline for Graph Attribute Imputation

Simple baseline that imputes missing node features by averaging the features
of K nearest neighbors in the graph structure.

Hyperparameters:
- K: 3 (number of nearest neighbors to consider)
"""

import numpy as np
import torch
import argparse
import sys
sys.path.insert(0, '../src')
from utils import load_data, RECALL_NDCG, data_split
from scipy.sparse import csgraph
import time


def train_KNN(adj, true_features, train_id, vali_test_id, args):
    """
    KNN-based feature imputation using graph distances.

    For each missing node:
    1. Find K nearest neighbors by shortest path distance
    2. Average features from observable neighbors only
    3. If no observable neighbors within K hops, use global mean

    Args:
        adj: Sparse adjacency matrix (scipy or torch.sparse)
        true_features: [N, F] tensor with ground truth features
        train_id: Indices of observable nodes (40%)
        vali_test_id: Indices of missing nodes (60%)
        args: Argparse namespace with K parameter

    Returns:
        imputed_features: [N, F] tensor with imputed values for missing nodes
    """
    K = getattr(args, 'K', 3)  # Default K=3
    device = true_features.device
    num_nodes = true_features.size(0)
    num_features = true_features.size(1)

    # Convert adjacency to scipy sparse format if needed
    if isinstance(adj, torch.Tensor):
        if adj.is_sparse:
            adj_scipy = adj.to_dense().cpu().numpy()
        else:
            adj_scipy = adj.cpu().numpy()
    else:
        adj_scipy = adj.toarray() if hasattr(adj, 'toarray') else adj

    # Compute shortest path distances between all nodes
    print(f'Computing shortest path distances for KNN (K={K})...')
    distances = csgraph.shortest_path(adj_scipy, directed=False, unweighted=True)

    # Create mask for observable nodes
    train_mask = np.zeros(num_nodes, dtype=bool)
    train_mask[train_id.cpu().numpy() if isinstance(train_id, torch.Tensor) else train_id] = True

    # Global mean as fallback
    global_mean = true_features[train_id].mean(dim=0)

    # Initialize output with original features
    imputed_features = true_features.clone()

    # Impute each missing node
    vali_test_list = vali_test_id.cpu().numpy() if isinstance(vali_test_id, torch.Tensor) else vali_test_id

    for node_idx in vali_test_list:
        # Get distances to all other nodes
        node_distances = distances[node_idx, :]

        # Filter to only observable nodes
        observable_distances = node_distances.copy()
        observable_distances[~train_mask] = np.inf  # Mask out missing nodes

        # Get K nearest observable neighbors
        # Sort by distance and take top K
        sorted_indices = np.argsort(observable_distances)
        k_nearest = []

        for idx in sorted_indices:
            if len(k_nearest) >= K:
                break
            if train_mask[idx] and not np.isinf(observable_distances[idx]):
                k_nearest.append(idx)

        # Impute features
        if len(k_nearest) > 0:
            k_nearest_tensor = torch.LongTensor(k_nearest).to(device)
            imputed_features[node_idx] = true_features[k_nearest_tensor].mean(dim=0)
        else:
            # No observable neighbors within reach, use global mean
            imputed_features[node_idx] = global_mean

    print(f'KNN imputation complete. Imputed {len(vali_test_list)} nodes.')
    return imputed_features


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KNN Baseline for Graph Attribute Imputation')
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'amac', 'amap'])
    parser.add_argument('--K', type=int, default=3, help='Number of nearest neighbors')
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

    print(f'\n=== KNN Baseline (K={args.K}) ===')
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

    # Run KNN imputation
    start_time = time.time()
    imputed_features = train_KNN(adj, true_features, train_id, vali_test_id, args)
    train_time = time.time() - start_time

    print(f'\nTraining time: {train_time:.2f}s')

    # Evaluate on test set only (not validation)
    test_imputed = imputed_features[test_id].cpu().numpy()
    test_true = true_features[test_id].cpu().numpy()

    print(f'\n=== Evaluation Results ===')
    for topK in [10, 20, 50]:
        recall, ndcg = RECALL_NDCG(test_imputed, test_true, topN=topK)
        print(f'Recall@{topK}: {recall:.4f}, NDCG@{topK}: {ndcg:.4f}')
