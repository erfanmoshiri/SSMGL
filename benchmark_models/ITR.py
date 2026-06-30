"""
ITR (Initializing Then Refining) for Graph Attribute Imputation

Two-stage approach:
1. Initialize: Learn structure embeddings using GCN on graph structure only
2. Refine: Aggregate attribute-observed neighbor embeddings via learned affinity

Paper: Tu et al., "Initializing Then Refining: A Simple Graph Attribute Imputation Network", IJCAI 2022

Hyperparameters:
- Hidden dim: 128
- Structure encoder layers: 2
- Refine iterations: 3
- Dropout: 0.5
- Learning rate: 0.01
- Epochs: 500
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
from torch_geometric.nn import GCNConv
import time


# ============================================================================
# ITR Model Components
# ============================================================================

class StructureEncoder(nn.Module):
    """
    Encode structure information using GCN on learned node embeddings.

    Uses learnable node embeddings instead of one-hot identity (more efficient).
    """

    def __init__(self, num_nodes, hidden_dim, num_layers, dropout):
        super().__init__()
        self.num_nodes = num_nodes

        # Learnable node embeddings (replaces one-hot identity)
        self.node_embeddings = nn.Embedding(num_nodes, hidden_dim)

        self.convs = nn.ModuleList()
        self.dropout = dropout

        # GCN layers operate on hidden_dim
        for i in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

    def forward(self, edge_index, device):
        """Forward pass with learned embeddings."""
        # Get learned embeddings for all nodes
        node_ids = torch.arange(self.num_nodes, device=device)
        x = self.node_embeddings(node_ids)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class AffinityNetwork(nn.Module):
    """
    Learn affinity between node embeddings for refinement.

    Affinity measures how much to weight neighbors during aggregation.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z_i, z_j):
        """
        Compute affinity between node i and node j.

        Args:
            z_i: [batch, hidden_dim] embeddings of nodes i
            z_j: [batch, hidden_dim] embeddings of nodes j

        Returns:
            affinity: [batch, 1] affinity scores
        """
        concat = torch.cat([z_i, z_j], dim=-1)
        affinity = self.mlp(concat)
        return affinity


class AttributeDecoder(nn.Module):
    """Decode embeddings to attribute space."""

    def __init__(self, hidden_dim, num_features):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_features)
        )

    def forward(self, z):
        return self.decoder(z)


class ITRModel(nn.Module):
    """ITR: Initializing Then Refining model."""

    def __init__(self, num_nodes, num_features, hidden_dim, num_layers, dropout):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        # Stage 1: Structure encoder
        self.structure_encoder = StructureEncoder(num_nodes, hidden_dim, num_layers, dropout)

        # Stage 2: Affinity network for refinement
        self.affinity_net = AffinityNetwork(hidden_dim)

        # Decoder: embeddings -> features
        self.decoder = AttributeDecoder(hidden_dim, num_features)

    def initialize_embeddings(self, edge_index, device):
        """Stage 1: Initialize embeddings using structure only."""
        z_init = self.structure_encoder(edge_index, device)
        return z_init

    def refine_embeddings(self, z, adj, obs_features, obs_mask, num_iterations=3):
        """
        Stage 2: Refine embeddings by aggregating from observable neighbors.

        OPTIMIZED: Only refines missing nodes, modifies in-place.

        Args:
            z: [N, hidden_dim] initial structure embeddings
            adj: [N, N] adjacency matrix (dense)
            obs_features: [N, F] observed features (zeros for missing)
            obs_mask: [N] boolean mask (True for observable nodes)
            num_iterations: Number of refinement iterations

        Returns:
            z_refined: [N, hidden_dim] refined embeddings
        """
        z_refined = z.clone()
        missing_mask = ~obs_mask
        missing_indices = missing_mask.nonzero(as_tuple=True)[0]

        # If no missing nodes, return as-is
        if len(missing_indices) == 0:
            return z_refined

        for iteration in range(num_iterations):
            # Only refine missing nodes (60% instead of 100%)
            for node_i in missing_indices:
                # Get neighbors
                neighbors = adj[node_i].nonzero(as_tuple=True)[0]

                if len(neighbors) == 0:
                    continue

                # Filter to observable neighbors only
                obs_neighbors = neighbors[obs_mask[neighbors]]

                if len(obs_neighbors) == 0:
                    # No observable neighbors, keep current embedding
                    continue

                # Compute affinities (vectorized over neighbors)
                z_i = z_refined[node_i].unsqueeze(0).expand(len(obs_neighbors), -1)
                z_j = z_refined[obs_neighbors]

                affinities = self.affinity_net(z_i, z_j)  # [num_neighbors, 1]
                weights = F.softmax(affinities.squeeze(-1), dim=0)  # [num_neighbors]

                # Weighted aggregation - update in place
                z_refined[node_i] = (weights.unsqueeze(1) * z_j).sum(dim=0)

        return z_refined

    def forward(self, edge_index, adj, obs_features, obs_mask, device, refine_iterations=3):
        """
        Full forward pass: Initialize + Refine + Decode.

        Args:
            edge_index: [2, E] edge indices for GCN
            adj: [N, N] dense adjacency for refinement
            obs_features: [N, F] observed features
            obs_mask: [N] boolean mask
            device: torch device
            refine_iterations: Number of refinement iterations

        Returns:
            x_hat: [N, F] predicted features
        """
        # Stage 1: Initialize from structure
        z_init = self.initialize_embeddings(edge_index, device)

        # Stage 2: Refine using observable neighbors
        z_refined = self.refine_embeddings(z_init, adj, obs_features, obs_mask, refine_iterations)

        # Decode to feature space
        x_hat = self.decoder(z_refined)

        return x_hat, z_init, z_refined


# ============================================================================
# Training Function
# ============================================================================

def train_ITR(adj, true_features, train_id, vali_test_id, args):
    """
    Train ITR model for attribute imputation.

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
    hidden_dim = getattr(args, 'hidden', 128)
    num_layers = getattr(args, 'layers', 2)
    dropout = getattr(args, 'dropout', 0.5)
    refine_iterations = getattr(args, 'refine_iterations', 3)  # Paper default
    lr = getattr(args, 'lr', 0.01)
    weight_decay = getattr(args, 'weight_decay', 5e-4)
    epochs = getattr(args, 'epochs', 500)  # Paper default (now faster with learned embeddings)

    num_nodes = true_features.size(0)
    num_features = true_features.size(1)

    # Prepare data
    if adj.is_sparse:
        edge_index = adj.coalesce().indices()
        adj_dense = adj.to_dense()
    else:
        edge_index = adj.nonzero(as_tuple=False).t()
        adj_dense = adj

    # Observable mask
    obs_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    obs_mask[train_id] = True

    # Observable features (zeros for missing)
    obs_features = true_features.clone()
    obs_features[vali_test_id] = 0.0

    # Create model
    model = ITRModel(
        num_nodes=num_nodes,
        num_features=num_features,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

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

    # Split validation from vali_test
    vali_id = vali_test_id[:len(vali_test_id) // 6]
    test_id = vali_test_id[len(vali_test_id) // 6:]

    print(f'Training ITR for {epochs} epochs (hidden={hidden_dim}, refine_iters={refine_iterations})...')
    print(f'Missing nodes to refine: {len(vali_test_id)} ({100*len(vali_test_id)/num_nodes:.1f}%)')

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        x_hat, z_init, z_refined = model(
            edge_index, adj_dense, obs_features, obs_mask, device, refine_iterations
        )

        # Loss on observable nodes only
        loss = criterion(x_hat[train_id], true_features[train_id])

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:  # Print more frequently
            print(f'Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}')

    # Generate imputed features
    print('Generating imputed features...')
    model.eval()
    with torch.no_grad():
        x_hat, _, _ = model(edge_index, adj_dense, obs_features, obs_mask, device, refine_iterations)

        if is_binary:
            x_hat = torch.sigmoid(x_hat)

    # Combine: keep original for train, use reconstructed for missing
    imputed_features = true_features.clone()
    imputed_features[vali_test_id] = x_hat[vali_test_id]

    print(f'ITR imputation complete. Imputed {len(vali_test_id)} nodes.')
    return imputed_features


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ITR Baseline for Graph Attribute Imputation')
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'amac', 'amap'])
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--refine_iterations', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=500)
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

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    print(f'\n=== ITR Baseline ===')
    print(f'Dataset: {args.dataset}')
    print(f'Device: {device}')
    print(f'Hidden: {args.hidden}, Layers: {args.layers}, Refine Iterations: {args.refine_iterations}')

    # Load data
    adj, diff, norm_adj, true_features, node_labels, indices = load_data(args)
    train_id, vali_id, test_id, vali_test_id = data_split(args, adj)

    if args.cuda:
        adj = adj.cuda()
        true_features = true_features.cuda()
        train_id = train_id.cuda()
        vali_test_id = vali_test_id.cuda()
        test_id = test_id.cuda()

    # Train ITR
    start_time = time.time()
    imputed_features = train_ITR(adj, true_features, train_id, vali_test_id, args)
    train_time = time.time() - start_time

    print(f'\nTotal time: {train_time:.2f}s')

    # Evaluate on test set
    test_imputed = imputed_features[test_id].cpu().numpy()
    test_true = true_features[test_id].cpu().numpy()

    print(f'\n=== Evaluation Results ===')
    for topK in [10, 20, 50]:
        recall, ndcg = RECALL_NDCG(test_imputed, test_true, topN=topK)
        print(f'Recall@{topK}: {recall:.4f}, NDCG@{topK}: {ndcg:.4f}')
