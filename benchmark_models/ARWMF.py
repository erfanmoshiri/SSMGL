"""
ARWMF: Attributed Random Walk as Matrix Factorization

Paper: Chen et al., "Attributed Random Walk as Matrix Factorization", NeurIPS Workshop 2019

Key approach:
1. Attributed random walk (same as GraphRNA's AttriWalk)
2. Matrix factorization using Shifted PPMI (NOT sampling-based skip-gram)
3. Closed-form solution (no training loop)

Hyperparameters:
- Alpha: 0.5 (probability of graph walk vs attribute walk)
- Window size T: 5
- Embedding dim: 128 (for low-rank variant)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import argparse
import sys
sys.path.insert(0, '../src')
from utils import load_data, RECALL_NDCG, data_split
import scipy.sparse as sp
from scipy.sparse.linalg import svds


# ============================================================================
# ARWMF: Matrix Factorization
# ============================================================================

def construct_transition_matrix(adj, features, alpha=0.5):
    """
    Construct transition matrix for attributed random walk.

    Same as GraphRNA's AttriWalk mechanism (Equation 1-2 in paper).

    Args:
        adj: [N, N] adjacency matrix
        features: [N, F] node features
        alpha: Probability of walking on graph (vs. through attributes)

    Returns:
        P_tilde: [(N+F), (N+F)] transition probability matrix
    """
    if adj.is_sparse:
        adj = adj.to_dense()

    N = adj.size(0)
    F = features.size(1)

    # Degree matrix
    degrees = adj.sum(dim=1, keepdim=True)
    degrees[degrees == 0] = 1  # Avoid division by zero

    # Normalize adjacency: A_bar = D^-1 * A
    A_normalized = adj / degrees

    # Attribute matrix: X[i,j] = 1 if node i has feature j (non-zero)
    X = (features != 0).float()

    # Normalize X: X_bar[i,j] = X[i,j] / sum_k(X[i,k])
    X_row_sums = X.sum(dim=1, keepdim=True)
    X_row_sums[X_row_sums == 0] = 1
    X_normalized = X / X_row_sums

    # Construct full transition matrix (Equation 1-2)
    # P_tilde = [ α*A_bar        (1-α)*D*X_bar  ]
    #           [ (1-α)*X_bar^T*D      0        ]

    device = adj.device
    P_tilde = torch.zeros(N + F, N + F, device=device)

    # Top-left: α * A_bar
    P_tilde[:N, :N] = alpha * A_normalized

    # Top-right: (1-α) * D * X_bar
    D_X = torch.diag(degrees.squeeze()) @ X_normalized
    P_tilde[:N, N:] = (1 - alpha) * D_X

    # Bottom-left: (1-α) * X_bar^T * D
    X_D = X_normalized.T @ torch.diag(degrees.squeeze())
    P_tilde[N:, :N] = (1 - alpha) * X_D

    # Bottom-right: 0 (already initialized to zero)

    return P_tilde


def compute_ppmi_matrix(P_tilde, window_size=5, num_nodes=None):
    """
    Compute Shifted PPMI matrix from transition matrix.

    Following Equation 6 in paper:
    M = vol(A_tilde) * D_tilde^{-1/2} * (1/T * sum_r=1^T P_tilde^r) * D_tilde^{-1/2}
    S = max(log(M) - log(k), 0)  # Shifted PPMI

    Args:
        P_tilde: Transition matrix
        window_size: Context window size T
        num_nodes: Number of nodes (to extract node-only embeddings)

    Returns:
        S: Shifted PPMI matrix (for nodes only)
    """
    device = P_tilde.device
    N_total = P_tilde.size(0)

    # Compute degree matrix D_tilde
    # For transition matrix, row sums should be 1, but we use the implicit degree
    # from A_tilde (before normalization)
    # Approximation: use identity since P is already normalized
    D_tilde = torch.eye(N_total, device=device)

    # Compute sum of powers: (1/T) * sum_{r=1}^T P^r
    P_power_sum = torch.zeros_like(P_tilde)
    P_current = P_tilde.clone()

    print(f'Computing matrix powers (window_size={window_size})...')
    for r in range(1, window_size + 1):
        P_power_sum += P_current
        if r < window_size:
            P_current = P_current @ P_tilde
        if r % 2 == 0:
            print(f'  Computed P^{r}...')

    P_power_sum = P_power_sum / window_size

    # Compute M (Equation 6)
    # M = vol(A_tilde) * D^{-1/2} * P_avg * D^{-1/2}
    # For simplicity, vol(A_tilde) ≈ N_total (approximation)
    vol_A = float(N_total)

    # Since D_tilde is identity (P already normalized), simplify:
    M = vol_A * P_power_sum

    # Compute PMI: log(M)
    M_clipped = torch.clamp(M, min=1e-10)  # Avoid log(0)
    PMI = torch.log(M_clipped)

    # Shifted PPMI: S = max(PMI - log(k), 0)
    # k is negative sampling parameter, typically k=1 or k=5
    k = 1.0
    S = torch.clamp(PMI - np.log(k), min=0)

    # Extract node-only part (first N rows/cols)
    if num_nodes is not None:
        S = S[:num_nodes, :num_nodes]

    return S


def low_rank_approximation(S, embedding_dim):
    """
    Compute low-rank approximation of S using SVD.

    S ≈ H @ H.T where H is [N, d]

    Args:
        S: [N, N] PPMI matrix
        embedding_dim: Target embedding dimension d

    Returns:
        H: [N, d] node embeddings
    """
    print(f'Computing SVD (rank={embedding_dim})...')

    # Move to CPU for SVD (scipy doesn't support GPU)
    S_np = S.cpu().numpy()

    # Truncated SVD: S ≈ U @ Sigma @ V.T
    # Take H = U @ sqrt(Sigma)
    k = min(embedding_dim, S_np.shape[0] - 1)
    U, Sigma, Vt = svds(S_np, k=k)

    # H = U @ sqrt(Sigma)
    H = U @ np.diag(np.sqrt(Sigma))

    return torch.from_numpy(H).float()


# ============================================================================
# Training Function
# ============================================================================

def train_ARWMF(adj, true_features, train_id, vali_test_id, args):
    """
    Train ARWMF model for attribute imputation.

    Uses matrix factorization (closed-form, no training loop).

    Args:
        adj: Sparse adjacency matrix
        true_features: [N, F] tensor with ground truth features
        train_id: Indices of observable nodes (40%)
        vali_test_id: Indices of missing nodes (60%)
        args: Argparse namespace

    Returns:
        imputed_features: [N, F] tensor with imputed values
    """
    device = true_features.device
    embedding_dim = getattr(args, 'embedding_dim', 128)
    alpha = getattr(args, 'alpha', 0.5)
    window_size = getattr(args, 'window_size', 5)

    num_nodes = true_features.size(0)
    num_features = true_features.size(1)

    print(f'ARWMF: Matrix factorization approach (α={alpha}, window={window_size}, dim={embedding_dim})')

    # Step 1: Construct transition matrix P_tilde
    print('Constructing attributed random walk transition matrix...')
    P_tilde = construct_transition_matrix(adj, true_features, alpha)

    # Step 2: Compute Shifted PPMI matrix
    print('Computing Shifted PPMI matrix...')
    S = compute_ppmi_matrix(P_tilde, window_size, num_nodes)

    # Step 3: Low-rank approximation (SVD)
    print('Computing low-rank approximation...')
    H = low_rank_approximation(S, embedding_dim).to(device)

    # Step 4: Train decoder (embeddings -> features)
    print(f'Training decoder ({embedding_dim} -> {num_features})...')

    decoder = nn.Sequential(
        nn.Linear(embedding_dim, embedding_dim),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(embedding_dim, num_features)
    ).to(device)

    optimizer = optim.Adam(decoder.parameters(), lr=0.01, weight_decay=5e-4)

    # Loss function
    is_binary = (args.dataset in ['cora', 'citeseer', 'amac', 'amap'])
    if is_binary:
        train_features = true_features[train_id]
        num_pos = (train_features != 0.0).sum().item()
        num_neg = (train_features == 0.0).sum().item()
        pos_weight = torch.tensor([num_neg / (num_pos + 1e-8)], device=device)
        criterion_bce = nn.BCEWithLogitsLoss(reduction='none')

        def criterion(pred, target):
            weight = torch.where(target != 0.0, pos_weight, torch.ones_like(pos_weight))
            return (criterion_bce(pred, target) * weight).mean()
    else:
        criterion = nn.MSELoss()

    # Train decoder on observable nodes
    epochs = 5000
    for epoch in range(1, epochs + 1):
        decoder.train()
        optimizer.zero_grad()

        # Decode embeddings
        x_hat = decoder(H)

        # Loss on observable nodes only
        loss = criterion(x_hat[train_id], true_features[train_id])

        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f'Decoder Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}')

    # Generate imputed features
    print('Generating imputed features...')
    decoder.eval()
    with torch.no_grad():
        x_hat = decoder(H)

        if is_binary:
            x_hat = torch.sigmoid(x_hat)

    # Combine: keep original for train, use reconstructed for missing
    imputed_features = true_features.clone()
    imputed_features[vali_test_id] = x_hat[vali_test_id]

    print(f'ARWMF imputation complete. Imputed {len(vali_test_id)} nodes.')
    return imputed_features


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ARWMF Baseline')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=72)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--train_fts_ratio', type=float, default=0.4)
    parser.add_argument('--generative_flag', type=bool, default=False)

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    print(f'Loading {args.dataset} dataset...')
    adj, diff, norm_adj, true_features, node_labels, indices = load_data(args)
    train_id, vali_id, test_id, vali_test_id = data_split(args, adj)

    if args.cuda:
        adj = adj.cuda()
        true_features = true_features.cuda()
        train_id = train_id.cuda()
        vali_test_id = vali_test_id.cuda()

    imputed_features = train_ARWMF(adj, true_features, train_id, vali_test_id, args)

    print('\nEvaluating...')
    for topK in [10, 20, 50]:
        recall, ndcg = RECALL_NDCG(
            imputed_features[vali_test_id].cpu().numpy(),
            true_features[vali_test_id].cpu().numpy(),
            topN=topK
        )
        print(f'Recall@{topK}: {recall:.4f}, NDCG@{topK}: {ndcg:.4f}')
