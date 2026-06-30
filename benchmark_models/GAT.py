"""
GAT (Graph Attention Networks) Baseline for Graph Attribute Imputation

Graph attention model adapted for attribute-missing graphs.
Missing nodes initialized with zeros, then learned via node classification.

Hyperparameters:
- Hidden dim: 256
- Attention heads: 8 (first layer), 1 (output layer)
- Dropout: 0.6
- Alpha: 0.2 (LeakyReLU negative slope)
- Learning rate: 0.005
- Epochs: 200
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
from torch_geometric.nn import GATConv
import time


class GAT(nn.Module):
    """2-layer Graph Attention Network."""

    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.6, alpha=0.2):
        super(GAT, self).__init__()
        self.dropout = dropout

        # First GAT layer: multi-head attention
        self.conv1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            dropout=dropout,
            negative_slope=alpha
        )

        # Second GAT layer: single head
        self.conv2 = GATConv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            concat=False,
            dropout=dropout,
            negative_slope=alpha
        )

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def train_GAT(adj, true_features, train_id, vali_test_id, args):
    """
    Train GAT with zero-initialized missing nodes.

    Strategy:
    1. Initialize missing node features with zeros
    2. Train on node classification (40% labeled)
    3. Extract learned embeddings as "imputed features"
    4. Decode embeddings back to feature space

    Args:
        adj: Sparse adjacency matrix
        true_features: [N, F] tensor with ground truth features
        train_id: Indices of observable nodes (40%)
        vali_test_id: Indices of missing nodes (60%)
        args: Argparse namespace

    Returns:
        imputed_features: [N, F] tensor (decoded embeddings)
    """
    device = true_features.device
    hidden_dim = getattr(args, 'hidden', 256)
    heads = getattr(args, 'heads', 8)
    dropout = getattr(args, 'dropout', 0.6)
    alpha = getattr(args, 'alpha', 0.2)
    lr = getattr(args, 'lr', 0.005)
    weight_decay = getattr(args, 'weight_decay', 5e-4)
    epochs = getattr(args, 'epochs', 200)

    # Prepare edge index
    if adj.is_sparse:
        edge_index = adj.coalesce().indices()
    else:
        edge_index = adj.nonzero(as_tuple=False).t()

    # Initialize missing node features with zeros
    init_features = true_features.clone()
    init_features[vali_test_id] = 0.0

    # Build model
    num_classes = int(args.num_classes) if hasattr(args, 'num_classes') else 7
    model = GAT(
        in_channels=true_features.size(1),
        hidden_channels=hidden_dim,
        out_channels=hidden_dim,
        heads=heads,
        dropout=dropout,
        alpha=alpha
    ).to(device)

    # Classifier head for node classification
    classifier = nn.Linear(hidden_dim, num_classes).to(device)

    optimizer = optim.Adam(
        list(model.parameters()) + list(classifier.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )

    # Get node labels
    try:
        # Ensure args has required attributes for load_data
        if not hasattr(args, 'generative_flag'):
            args.generative_flag = False
        # load_data returns: adj, diff, adj_norm, features, labels, indices
        _, _, _, _, node_labels, _ = load_data(args)
        node_labels = node_labels.to(device)
        # Ensure labels are 1D (class indices, not one-hot)
        if node_labels.dim() > 1:
            node_labels = node_labels.argmax(dim=1)
        has_labels = True
    except Exception as e:
        has_labels = False
        print(f"Warning: No node labels available. Training with reconstruction loss. Error: {e}")

    # Split validation from vali_test
    vali_id = vali_test_id[:len(vali_test_id) // 6]
    test_id = vali_test_id[len(vali_test_id) // 6:]

    print(f'Training GAT for {epochs} epochs (heads={heads}, hidden={hidden_dim})...')
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        classifier.train()
        optimizer.zero_grad()

        # Forward pass
        embeddings = model(init_features, edge_index)

        if has_labels:
            # Classification loss on labeled nodes
            logits = classifier(embeddings[train_id])
            loss = F.cross_entropy(logits, node_labels[train_id])
        else:
            # Reconstruction loss on observable features
            recon = embeddings[train_id]
            target = true_features[train_id]
            loss = F.mse_loss(recon, target)

        loss.backward()
        optimizer.step()

        # Validation every 10 epochs
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                embeddings_val = model(init_features, edge_index)

                if has_labels:
                    logits_val = classifier(embeddings_val[vali_id])
                    preds_val = logits_val.argmax(dim=1)
                    val_acc = (preds_val == node_labels[vali_id]).float().mean().item()
                else:
                    val_acc = 0.0

            print(f'Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}, Val Acc: {val_acc:.4f}')

            if val_acc > best_val_acc:
                best_val_acc = val_acc

    # Extract final embeddings
    print('Extracting learned embeddings...')
    model.eval()
    with torch.no_grad():
        final_embeddings = model(init_features, edge_index)

    # Train decoder to map embeddings back to feature space
    decoder = nn.Linear(hidden_dim, true_features.size(1)).to(device)
    optimizer_dec = optim.Adam(decoder.parameters(), lr=0.01)

    print('Training decoder to map embeddings to feature space...')
    for epoch in range(50):
        decoder.train()
        optimizer_dec.zero_grad()

        # Decode embeddings of observable nodes
        decoded = decoder(final_embeddings[train_id])
        loss_dec = F.mse_loss(decoded, true_features[train_id])

        loss_dec.backward()
        optimizer_dec.step()

        if epoch % 10 == 0:
            print(f'Decoder Epoch {epoch}/50, Loss: {loss_dec.item():.4f}')

    # Final imputation
    decoder.eval()
    with torch.no_grad():
        imputed_features = true_features.clone()
        decoded_missing = decoder(final_embeddings[vali_test_id])
        imputed_features[vali_test_id] = decoded_missing

    print(f'GAT imputation complete. Imputed {len(vali_test_id)} nodes.')
    return imputed_features


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAT Baseline for Graph Attribute Imputation')
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'amac', 'amap'])
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.6)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=200)
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

    # Set num_classes per dataset
    num_classes_map = {'cora': 7, 'citeseer': 6, 'amac': 10, 'amap': 8}
    args.num_classes = num_classes_map.get(args.dataset, 7)

    print(f'\n=== GAT Baseline ===')
    print(f'Dataset: {args.dataset}')
    print(f'Device: {device}')
    print(f'Hidden: {args.hidden}, Heads: {args.heads}, Dropout: {args.dropout}')

    # Load data
    adj, diff, norm_adj, true_features, node_labels, indices = load_data(args)
    train_id, vali_id, test_id, vali_test_id = data_split(args, adj)

    if args.cuda:
        adj = adj.cuda()
        true_features = true_features.cuda()
        train_id = train_id.cuda()
        vali_test_id = vali_test_id.cuda()
        test_id = test_id.cuda()

    # Train GAT
    start_time = time.time()
    imputed_features = train_GAT(adj, true_features, train_id, vali_test_id, args)
    train_time = time.time() - start_time

    print(f'\nTotal time: {train_time:.2f}s')

    # Evaluate on test set
    test_imputed = imputed_features[test_id].cpu().numpy()
    test_true = true_features[test_id].cpu().numpy()

    print(f'\n=== Evaluation Results ===')
    for topK in [10, 20, 50]:
        recall, ndcg = RECALL_NDCG(test_imputed, test_true, topN=topK)
        print(f'Recall@{topK}: {recall:.4f}, NDCG@{topK}: {ndcg:.4f}')
