"""
SVGA (Structured Variational Graph Autoencoder) for Graph Attribute Imputation

Uses Gaussian Markov Random Fields (GMRF) prior on graph structure for
structured variational inference.

Paper: Yoo et al., "Accurate Node Feature Estimation with Structured Variational Graph Autoencoder", KDD 2022

Hyperparameters:
- Hidden dim: 256
- Layers: 2
- Lambda (GMRF weight): 1.0
- Beta (GMRF temperature): 1.0
- Dropout: 0.5
- Learning rate: 0.001
- Epochs: 2000
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
from torch_geometric.utils import get_laplacian
import time


# ============================================================================
# Loss Functions
# ============================================================================

class BernoulliLoss(nn.Module):
    """Loss term for binary features with class balancing."""

    def __init__(self, balanced=True):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction='none')
        self.balanced = balanced

    def forward(self, input, target):
        if self.balanced:
            pos_ratio = (target > 0).float().mean()
            weight = torch.ones_like(target)
            weight[target > 0] = 1 / (2 * pos_ratio + 1e-8)
            weight[target == 0] = 1 / (2 * (1 - pos_ratio) + 1e-8)
            loss = self.loss(input, target) * weight
        else:
            loss = self.loss(input, target)
        return loss.mean()


class GMRFLoss(nn.Module):
    """
    Gaussian Markov Random Field loss for structured prior.

    Encourages smoothness: adjacent nodes should have similar embeddings.
    """

    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
        self.cached_laplacian = None

    def forward(self, features, edge_index, num_nodes):
        """
        Compute GMRF loss: tr(Z^T L Z) - log|I + Z^T Z / beta|
        """
        if self.cached_laplacian is None:
            # Compute normalized graph Laplacian
            edge_index_lap, edge_weight = get_laplacian(
                edge_index, normalization='sym', num_nodes=num_nodes
            )
            self.cached_laplacian = torch.sparse_coo_tensor(
                edge_index_lap, edge_weight,
                size=(num_nodes, num_nodes),
                device=edge_index.device
            )

        hidden_dim = features.size(1)

        # tr(Z^T L Z) - smoothness term
        l_z = torch.sparse.mm(self.cached_laplacian, features)
        smoothness = (features * l_z).sum()

        # log|I + Z^T Z / beta| - complexity term
        eye = torch.eye(hidden_dim, device=features.device)
        ztz = features.t().matmul(features)
        complexity = (eye + ztz / self.beta).logdet()

        return (smoothness - complexity / 2) / num_nodes


# ============================================================================
# SVGA Model
# ============================================================================

class Encoder(nn.Module):
    """GCN-based encoder."""

    def __init__(self, in_channels, hidden_channels, num_layers, dropout):
        super().__init__()
        self.convs = nn.ModuleList()
        self.dropout = dropout

        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            self.convs.append(GCNConv(in_dim, hidden_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class SVGA(nn.Module):
    """Structured Variational Graph Autoencoder."""

    def __init__(self, num_nodes, num_features, hidden_size, num_layers, dropout, lamda, beta):
        super().__init__()
        self.num_nodes = num_nodes
        self.lamda = lamda
        self.dropout_layer = nn.Dropout(dropout)

        # Create diagonal feature matrix (one-hot identity)
        indices = torch.arange(num_nodes).unsqueeze(0).repeat(2, 1)
        values = torch.ones(num_nodes)
        self.register_buffer('diag_indices', indices)
        self.register_buffer('diag_values', values)

        # Encoder: diagonal features -> embeddings
        self.encoder = Encoder(num_nodes, hidden_size, num_layers, dropout)

        # Decoder: embeddings -> features
        self.decoder = nn.Linear(hidden_size, num_features, bias=False)

        # Losses
        self.recon_loss = BernoulliLoss(balanced=True)
        self.gmrf_loss = GMRFLoss(beta=beta)

    def get_diagonal_features(self):
        """Get diagonal feature matrix (identity matrix)."""
        return torch.sparse_coo_tensor(
            self.diag_indices, self.diag_values,
            size=(self.num_nodes, self.num_nodes),
            device=self.diag_indices.device
        ).to_dense()

    def forward(self, edge_index):
        """Forward pass."""
        # Get diagonal features
        x_diag = self.get_diagonal_features()

        # Encode
        z = self.encoder(x_diag, edge_index)

        # Unit normalize embeddings
        z = z / (z.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-8)

        # Decode
        z_dropped = self.dropout_layer(z)
        x_hat = self.decoder(z_dropped)

        return z, x_hat

    def compute_losses(self, edge_index, x_nodes, x_features):
        """Compute reconstruction and GMRF losses."""
        z, x_hat = self.forward(edge_index)

        # Reconstruction loss on observable nodes
        loss_recon = self.recon_loss(x_hat[x_nodes], x_features)

        # GMRF regularization
        loss_gmrf = self.lamda * self.gmrf_loss(z, edge_index, self.num_nodes)

        return loss_recon, loss_gmrf


# ============================================================================
# Training Function
# ============================================================================

def train_SVGA(adj, true_features, train_id, vali_test_id, args):
    """
    Train SVGA model for attribute imputation.

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
    hidden_size = getattr(args, 'hidden', 256)
    num_layers = getattr(args, 'layers', 2)
    dropout = getattr(args, 'dropout', 0.5)
    lamda = getattr(args, 'lamda', 1.0)
    beta = getattr(args, 'beta', 1.0)
    lr = getattr(args, 'lr', 0.001)
    epochs = getattr(args, 'epochs', 2000)
    patience = getattr(args, 'patience', 100)

    num_nodes = true_features.size(0)
    num_features = true_features.size(1)

    # Prepare edge index
    if adj.is_sparse:
        edge_index = adj.coalesce().indices()
    else:
        edge_index = adj.nonzero(as_tuple=False).t()

    # Observable features
    x_nodes = train_id
    x_features = true_features[x_nodes]

    # Create model
    model = SVGA(
        num_nodes=num_nodes,
        num_features=num_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        lamda=lamda,
        beta=beta
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Split validation from vali_test
    vali_id = vali_test_id[:len(vali_test_id) // 6]
    test_id = vali_test_id[len(vali_test_id) // 6:]

    print(f'Training SVGA for {epochs} epochs (hidden={hidden_size}, lamda={lamda}, beta={beta})...')

    best_val_loss = float('inf')
    bad_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Compute losses
        loss_recon, loss_gmrf = model.compute_losses(edge_index, x_nodes, x_features)
        loss_total = loss_recon + loss_gmrf

        loss_total.backward()
        optimizer.step()

        # Validation
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                _, x_hat_val = model(edge_index)
                loss_val = model.recon_loss(x_hat_val[vali_id], true_features[vali_id])

            print(f'Epoch {epoch}/{epochs}, Recon: {loss_recon.item():.4f}, GMRF: {loss_gmrf.item():.4f}, Val: {loss_val.item():.4f}')

            # Early stopping
            if loss_val.item() < best_val_loss:
                best_val_loss = loss_val.item()
                bad_counter = 0
            else:
                bad_counter += 1
                if bad_counter >= patience:
                    print(f'Early stopping at epoch {epoch}')
                    break

    # Generate imputed features
    print('Generating imputed features...')
    model.eval()
    with torch.no_grad():
        _, x_hat = model(edge_index)
        x_hat = torch.sigmoid(x_hat)  # Apply sigmoid for binary features

    # Combine: keep original for train, use reconstructed for missing
    imputed_features = true_features.clone()
    imputed_features[vali_test_id] = x_hat[vali_test_id]

    print(f'SVGA imputation complete. Imputed {len(vali_test_id)} nodes.')
    return imputed_features


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SVGA Baseline for Graph Attribute Imputation')
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'amac', 'amap'])
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lamda', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--patience', type=int, default=100)
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

    print(f'\n=== SVGA Baseline ===')
    print(f'Dataset: {args.dataset}')
    print(f'Device: {device}')
    print(f'Hidden: {args.hidden}, Layers: {args.layers}, Lambda: {args.lamda}')

    # Load data
    adj, diff, norm_adj, true_features, node_labels, indices = load_data(args)
    train_id, vali_id, test_id, vali_test_id = data_split(args, adj)

    if args.cuda:
        adj = adj.cuda()
        true_features = true_features.cuda()
        train_id = train_id.cuda()
        vali_test_id = vali_test_id.cuda()
        test_id = test_id.cuda()

    # Train SVGA
    start_time = time.time()
    imputed_features = train_SVGA(adj, true_features, train_id, vali_test_id, args)
    train_time = time.time() - start_time

    print(f'\nTotal time: {train_time:.2f}s')

    # Evaluate on test set
    test_imputed = imputed_features[test_id].cpu().numpy()
    test_true = true_features[test_id].cpu().numpy()

    print(f'\n=== Evaluation Results ===')
    for topK in [10, 20, 50]:
        recall, ndcg = RECALL_NDCG(test_imputed, test_true, topN=topK)
        print(f'Recall@{topK}: {recall:.4f}, NDCG@{topK}: {ndcg:.4f}')
