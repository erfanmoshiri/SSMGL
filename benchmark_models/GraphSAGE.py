"""
GraphSAGE Baseline for Graph Attribute Imputation

Inductive graph learning model adapted for attribute-missing graphs.
Missing nodes initialized with zeros, then learned via node classification.

Hyperparameters:
- Hidden dim: 256
- Dropout: 0.5
- Learning rate: 0.01
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
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import time


class GraphSAGE(nn.Module):
    """2-layer GraphSAGE for node classification."""

    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GraphSAGE, self).__init__()
        self.dropout = dropout

        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def train_GraphSAGE(adj, true_features, train_id, vali_test_id, args):
    """
    Train GraphSAGE with zero-initialized missing nodes.

    Strategy:
    1. Initialize missing node features with zeros
    2. Train on node classification (40% labeled)
    3. Extract learned embeddings as "imputed features"

    Args:
        adj: Sparse adjacency matrix
        true_features: [N, F] tensor with ground truth features
        train_id: Indices of observable nodes (40%)
        vali_test_id: Indices of missing nodes (60%)
        args: Argparse namespace

    Returns:
        imputed_features: [N, F] tensor (learned embeddings)
    """
    device = true_features.device
    hidden_dim = getattr(args, 'hidden', 256)
    dropout = getattr(args, 'dropout', 0.5)
    lr = getattr(args, 'lr', 0.01)
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
    num_classes = int(args.num_classes) if hasattr(args, 'num_classes') else 7  # Default for Cora
    model = GraphSAGE(
        in_channels=true_features.size(1),
        hidden_channels=hidden_dim,
        out_channels=hidden_dim,  # Output embeddings, not classes
        dropout=dropout
    ).to(device)

    # For node classification, we need a classifier head
    classifier = nn.Linear(hidden_dim, num_classes).to(device)

    optimizer = optim.Adam(
        list(model.parameters()) + list(classifier.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )

    # Get node labels (if available)
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
        print(f"Warning: No node labels available. Training without supervision. Error: {e}")

    # Split validation from vali_test
    vali_id = vali_test_id[:len(vali_test_id) // 6]
    test_id = vali_test_id[len(vali_test_id) // 6:]

    print(f'Training GraphSAGE for {epochs} epochs...')
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

    # Extract final embeddings as imputed features
    print('Extracting learned embeddings...')
    model.eval()
    with torch.no_grad():
        final_embeddings = model(init_features, edge_index)

    # Create imputed features: keep original for train, use embeddings for missing
    # Note: Embeddings have different dimensionality, so we need to decode back to feature space
    # For simplicity, use the embeddings directly (will need adjustment in evaluation)

    # Option 1: Use embeddings as-is (different dim)
    # Option 2: Add a decoder to map embeddings back to feature space

    # Let's add a simple linear decoder
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

    print(f'GraphSAGE imputation complete. Imputed {len(vali_test_id)} nodes.')
    return imputed_features


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphSAGE Baseline for Graph Attribute Imputation')
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'amac', 'amap'])
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
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

    print(f'\n=== GraphSAGE Baseline ===')
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

    # Train GraphSAGE
    start_time = time.time()
    imputed_features = train_GraphSAGE(adj, true_features, train_id, vali_test_id, args)
    train_time = time.time() - start_time

    print(f'\nTotal time: {train_time:.2f}s')

    # Evaluate on test set
    test_imputed = imputed_features[test_id].cpu().numpy()
    test_true = true_features[test_id].cpu().numpy()

    print(f'\n=== Evaluation Results ===')
    for topK in [10, 20, 50]:
        recall, ndcg = RECALL_NDCG(test_imputed, test_true, topN=topK)
        print(f'Recall@{topK}: {recall:.4f}, NDCG@{topK}: {ndcg:.4f}')
