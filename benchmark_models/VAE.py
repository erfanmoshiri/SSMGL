"""
VAE (Variational AutoEncoder) Baseline for Graph Attribute Imputation

Standard VAE with reparameterization trick. For missing nodes, features are initialized
using neighbor aggregation before being passed through the VAE.

Hyperparameters:
- Hidden dim: 64
- Dropout: 0.5
- Learning rate: 0.005
- Epochs: 1000
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
import time


class VAE(nn.Module):
    """Variational AutoEncoder for feature imputation."""

    def __init__(self, n_fts, n_hid, dropout):
        super(VAE, self).__init__()
        self.n_fts = n_fts
        self.n_hid = n_hid
        self.dropout = dropout

        # Encoder
        self.fc1 = nn.Linear(n_fts, 200)
        self.fc21 = nn.Linear(200, n_hid)  # mu
        self.fc22 = nn.Linear(200, n_hid)  # logvar

        # Decoder
        self.fc3 = nn.Linear(n_hid, 200)
        self.fc4 = nn.Linear(200, n_fts)

    def encode(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + eps * std"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h3 = F.dropout(h3, self.dropout, training=self.training)
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_function(recon_x, x, mu, logvar, pos_weight, neg_weight, is_binary=True):
    """
    VAE loss: Reconstruction + KL divergence.

    For binary features (Cora, Citeseer): BCE with class weights
    For continuous features: MSE
    """
    if is_binary:
        BCE = nn.BCEWithLogitsLoss(reduction='none')
        output_reshape = recon_x.reshape(-1)
        target_reshape = x.reshape(-1)

        # Apply weights: higher weight for positive class (sparse features)
        weight_mask = torch.where(target_reshape != 0.0, pos_weight, neg_weight)
        recon_loss = torch.mean(BCE(output_reshape, target_reshape) * weight_mask)
    else:
        # Continuous features
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')

    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_loss, recon_loss, kl_loss


def aggregate_neighbors(adj, true_features, train_id, missing_id):
    """
    Initialize missing node features by aggregating observable neighbor features.
    This is used as preprocessing before VAE training.
    """
    device = true_features.device
    num_nodes = true_features.size(0)

    # Convert to dense
    if adj.is_sparse:
        adj_dense = adj.to_dense()
    else:
        adj_dense = adj

    # Mask for observable nodes
    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    train_mask[train_id] = True

    # Global mean fallback
    global_mean = true_features[train_id].mean(dim=0)

    # Initialize with aggregation
    init_features = true_features.clone()

    for node_idx in missing_id:
        neighbors = adj_dense[node_idx].nonzero(as_tuple=True)[0]
        obs_neighbors = neighbors[train_mask[neighbors]]

        if len(obs_neighbors) > 0:
            init_features[node_idx] = true_features[obs_neighbors].mean(dim=0)
        else:
            init_features[node_idx] = global_mean

    return init_features


def train_VAE(adj, true_features, train_id, vali_test_id, args):
    """
    Train VAE on observable nodes and impute missing node features.

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
    hidden_dim = getattr(args, 'hidden', 64)
    dropout = getattr(args, 'dropout', 0.5)
    lr = getattr(args, 'lr', 0.005)
    weight_decay = getattr(args, 'weight_decay', 5e-4)
    epochs = getattr(args, 'epochs', 1000)
    patience = getattr(args, 'patience', 100)

    # Initialize missing features with neighbor aggregation
    print('Initializing missing features with neighbor aggregation...')
    init_features = aggregate_neighbors(adj, true_features, train_id, vali_test_id)

    # Determine if binary or continuous features
    is_binary = (args.dataset in ['cora', 'citeseer', 'amac', 'amap'])

    # Compute class weights for binary features
    if is_binary:
        train_features = true_features[train_id]
        num_pos = (train_features != 0.0).sum().item()
        num_neg = (train_features == 0.0).sum().item()
        pos_weight = torch.tensor([num_neg / num_pos], device=device)
        neg_weight = torch.tensor([1.0], device=device)
    else:
        pos_weight = torch.tensor([1.0], device=device)
        neg_weight = torch.tensor([1.0], device=device)

    # Create model
    model = VAE(n_fts=true_features.size(1), n_hid=hidden_dim, dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Split validation from vali_test
    vali_id = vali_test_id[:len(vali_test_id) // 6]  # ~10% of total
    test_id = vali_test_id[len(vali_test_id) // 6:]  # ~50% of total

    # Training loop
    print(f'Training VAE for {epochs} epochs...')
    best_loss = float('inf')
    bad_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Train on observable nodes
        train_fts = true_features[train_id]
        recon, mu, logvar = model(train_fts)
        loss, recon_loss, kl_loss = loss_function(
            recon, train_fts, mu, logvar, pos_weight, neg_weight, is_binary
        )

        loss.backward()
        optimizer.step()

        # Validation
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                vali_fts = true_features[vali_id]
                recon_val, mu_val, logvar_val = model(vali_fts)
                loss_val, _, _ = loss_function(
                    recon_val, vali_fts, mu_val, logvar_val, pos_weight, neg_weight, is_binary
                )

            print(f'Epoch {epoch}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {loss_val.item():.4f}')

            # Early stopping
            if loss_val.item() < best_loss:
                best_loss = loss_val.item()
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
        # Use initialized features for encoding
        recon_all, _, _ = model(init_features)

        if is_binary:
            recon_all = torch.sigmoid(recon_all)

    # Combine: keep original for train, use reconstructed for vali_test
    imputed_features = true_features.clone()
    imputed_features[vali_test_id] = recon_all[vali_test_id]

    print(f'VAE imputation complete. Imputed {len(vali_test_id)} nodes.')
    return imputed_features


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE Baseline for Graph Attribute Imputation')
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'amac', 'amap'])
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=1000)
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

    print(f'\n=== VAE Baseline ===')
    print(f'Dataset: {args.dataset}')
    print(f'Device: {device}')
    print(f'Hidden: {args.hidden}, Dropout: {args.dropout}')

    # Load data
    adj, diff, norm_adj, true_features, node_labels, indices = load_data(args)
    train_id, vali_id, test_id, vali_test_id = data_split(args, adj)

    if args.cuda:
        adj = adj.cuda()
        true_features = true_features.cuda()
        train_id = train_id.cuda()
        vali_test_id = vali_test_id.cuda()
        test_id = test_id.cuda()

    # Train VAE
    start_time = time.time()
    imputed_features = train_VAE(adj, true_features, train_id, vali_test_id, args)
    train_time = time.time() - start_time

    print(f'\nTotal time: {train_time:.2f}s')

    # Evaluate on test set
    test_imputed = imputed_features[test_id].cpu().numpy()
    test_true = true_features[test_id].cpu().numpy()

    print(f'\n=== Evaluation Results ===')
    for topK in [10, 20, 50]:
        recall, ndcg = RECALL_NDCG(test_imputed, test_true, topN=topK)
        print(f'Recall@{topK}: {recall:.4f}, NDCG@{topK}: {ndcg:.4f}')
