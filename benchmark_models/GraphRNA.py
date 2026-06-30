"""
GraphRNA: Graph Recurrent Networks with Attributed Random Walks

Paper: Huang et al., "Graph Recurrent Networks with Attributed Random Walks", KDD 2019

Key innovations:
1. AttriWalk: Bipartite random walks on graph + attribute network
2. GRN: Bidirectional LSTM on attributed walk sequences
3. Uses actual node attributes (not learned embeddings)

Hyperparameters:
- Embedding dim: 128
- Hidden dim: 128
- Walk length: 10
- Walks per node: 20
- Alpha: 0.5 (probability of graph walk vs attribute walk)
- Epochs: 100
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


# ============================================================================
# AttriWalk: Attributed Random Walk Generator
# ============================================================================

class AttributedRandomWalkGenerator:
    """
    AttriWalk: Joint random walks on graph and attributes.

    Creates bipartite network where nodes can walk to attribute categories.
    Walk sequence contains both node indices and attribute indices.
    """

    def __init__(self, adj, features, walk_length, walks_per_node, alpha=0.5):
        """
        Args:
            adj: [N, N] adjacency matrix
            features: [N, F] node features (binary attributes)
            walk_length: Length of each walk
            walks_per_node: Number of walks per node
            alpha: Probability of walking on graph (vs. through attributes)
        """
        self.adj = adj.to_dense() if adj.is_sparse else adj
        self.features = features
        self.num_nodes = features.size(0)
        self.num_features = features.size(1)
        self.walk_length = walk_length
        self.walks_per_node = walks_per_node
        self.alpha = alpha

        # Construct bipartite attribute network A
        # A[i,k] = 1 if node i has attribute k (feature value != 0)
        self.A = (features != 0).float()  # [N, F] binary attribute matrix

        # Normalize A: A_bar[i,k] = A[i,k] / sum_p(A[i,p])
        row_sums = self.A.sum(dim=1, keepdim=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        self.A_normalized = self.A / row_sums  # [N, F]

        # Normalize adjacency: G_bar[i,j] = G[i,j] / sum_p(G[i,p])
        row_sums_adj = self.adj.sum(dim=1, keepdim=True)
        row_sums_adj[row_sums_adj == 0] = 1
        self.adj_normalized = self.adj / row_sums_adj

    def generate_walks(self, start_nodes):
        """
        Generate attributed random walks using AttriWalk.

        Walk encoding:
        - Node i: represented as index i (0 to N-1)
        - Attribute k: represented as index N+k (N to N+F-1)

        Returns:
            walks: List of walks, each walk is a list of indices
        """
        walks = []

        for start_node in start_nodes:
            for _ in range(self.walks_per_node):
                walk = [start_node]  # Start with a node
                current = start_node
                current_is_node = True  # Track if current is node or attribute

                for step in range(self.walk_length - 1):
                    # Flip biased coin
                    coin = torch.rand(1).item()

                    if current_is_node:
                        # Currently at a node
                        if coin < self.alpha:
                            # Head: Walk on graph (node -> node)
                            neighbors = self.adj[current].nonzero(as_tuple=True)[0]
                            if len(neighbors) == 0:
                                break

                            # Sample next node proportional to edge weights
                            probs = self.adj_normalized[current][neighbors]
                            next_idx = torch.multinomial(probs, 1).item()
                            next_node = neighbors[next_idx].item()

                            walk.append(next_node)
                            current = next_node
                            current_is_node = True
                        else:
                            # Tail: Walk through attributes (node -> attribute)
                            attr_probs = self.A_normalized[current]
                            nonzero_attrs = attr_probs.nonzero(as_tuple=True)[0]

                            if len(nonzero_attrs) == 0:
                                break

                            probs = attr_probs[nonzero_attrs]
                            attr_idx = torch.multinomial(probs, 1).item()
                            attr_k = nonzero_attrs[attr_idx].item()

                            # Add attribute to walk (offset by num_nodes)
                            walk.append(self.num_nodes + attr_k)
                            current = attr_k
                            current_is_node = False
                    else:
                        # Currently at an attribute category
                        # Walk: attribute -> node
                        nodes_with_attr = self.A[:, current].nonzero(as_tuple=True)[0]

                        if len(nodes_with_attr) == 0:
                            break

                        # Compute probabilities: A[j,k] / sum_q(A[q,k])
                        col_sum = self.A[:, current].sum()
                        if col_sum == 0:
                            break
                        probs = self.A[nodes_with_attr, current] / col_sum

                        next_idx = torch.multinomial(probs, 1).item()
                        next_node = nodes_with_attr[next_idx].item()

                        walk.append(next_node)
                        current = next_node
                        current_is_node = True

                if len(walk) > 1:
                    walks.append(walk)

        return walks


# ============================================================================
# Graph Recurrent Networks (GRN)
# ============================================================================

class GraphRNAEncoder(nn.Module):
    """
    Bidirectional LSTM encoder for attributed walk sequences.

    Uses actual node attributes (not learned embeddings).
    """

    def __init__(self, num_nodes, num_features, embedding_dim, hidden_dim):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.embedding_dim = embedding_dim

        # FC layer to map attributes/one-hot to embedding space
        # Paper: x_j = σ(a_j * W_a + b_a) or x_j = σ(e_j * W_a + b_a)
        self.fc = nn.Linear(num_features, embedding_dim)

        # Bidirectional LSTM (GRN)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.hidden_dim = hidden_dim

    def forward(self, walks, node_features):
        """
        Encode walks using bidirectional LSTM.

        Args:
            walks: List of walk sequences (each contains node/attribute indices)
            node_features: [N, F] node attribute matrix

        Returns:
            walk_embeddings: [num_walks, hidden_dim*2] (bidirectional)
        """
        device = node_features.device

        # Convert walks to input tensors
        max_len = max(len(w) for w in walks)
        batch_size = len(walks)

        # Create input matrix: [batch_size, max_len, num_features]
        walk_inputs = torch.zeros(batch_size, max_len, self.num_features, device=device)

        for i, walk in enumerate(walks):
            for j, idx in enumerate(walk):
                if idx < self.num_nodes:
                    # It's a node: use its attribute vector a_j
                    walk_inputs[i, j] = node_features[idx]
                else:
                    # It's an attribute: use one-hot vector e_j
                    attr_idx = idx - self.num_nodes
                    walk_inputs[i, j, attr_idx] = 1.0

        # Pass through FC layer: x_j = σ(input * W_a + b_a)
        x = torch.sigmoid(self.fc(walk_inputs))  # [batch, max_len, emb_dim]

        # Bidirectional LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use final hidden states (forward + backward)
        # h_n shape: [2, batch, hidden_dim]
        forward_hidden = h_n[0]  # [batch, hidden_dim]
        backward_hidden = h_n[1]  # [batch, hidden_dim]
        walk_embeddings = torch.cat([forward_hidden, backward_hidden], dim=1)  # [batch, hidden_dim*2]

        return walk_embeddings


class GraphRNA(nn.Module):
    """GraphRNA: Complete model with encoder + decoder."""

    def __init__(self, num_nodes, num_features, embedding_dim, hidden_dim):
        super().__init__()
        self.num_nodes = num_nodes

        # Encoder: walks -> embeddings
        self.encoder = GraphRNAEncoder(num_nodes, num_features, embedding_dim, hidden_dim)

        # Node embeddings: one per node
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, hidden_dim * 2))

        # Decoder: embeddings -> features
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_features)
        )

    def forward(self, walks, node_features):
        """
        Forward pass: encode walks and aggregate to node embeddings.

        Args:
            walks: List of walks
            node_features: [N, F] node attributes

        Returns:
            x_hat: [N, F] reconstructed features
        """
        # Encode walks
        walk_embeddings = self.encoder(walks, node_features)  # [num_walks, hidden*2]

        # Aggregate walk embeddings to node embeddings (mean pooling)
        # Group walks by starting node
        node_walk_embs = {i: [] for i in range(self.num_nodes)}
        for walk_idx, walk in enumerate(walks):
            start_node = walk[0]  # First element is always a node
            if start_node < self.num_nodes:  # Make sure it's a valid node
                node_walk_embs[start_node].append(walk_embeddings[walk_idx])

        # Update node embeddings (mean pooling)
        for node, embs in node_walk_embs.items():
            if len(embs) > 0:
                self.node_embeddings.data[node] = torch.stack(embs).mean(dim=0)

        # Decode to features
        x_hat = self.decoder(self.node_embeddings)

        return x_hat


# ============================================================================
# Training Function
# ============================================================================

def train_GraphRNA(adj, true_features, train_id, vali_test_id, args):
    """
    Train GraphRNA model for attribute imputation.

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
    hidden_dim = getattr(args, 'hidden', 128)
    walk_length = getattr(args, 'walk_length', 10)
    walks_per_node = getattr(args, 'walks_per_node', 20)  # Paper default
    alpha = getattr(args, 'alpha', 0.5)  # AttriWalk parameter
    lr = getattr(args, 'lr', 0.01)
    weight_decay = getattr(args, 'weight_decay', 5e-4)
    epochs = getattr(args, 'epochs', 100)  # Paper default

    num_nodes = true_features.size(0)
    num_features = true_features.size(1)

    # Create model
    model = GraphRNA(
        num_nodes=num_nodes,
        num_features=num_features,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim
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

    # Generate attributed random walks using AttriWalk
    print(f'Generating attributed random walks (AttriWalk: α={alpha}, length={walk_length}, per_node={walks_per_node})...')
    walk_generator = AttributedRandomWalkGenerator(
        adj, true_features, walk_length, walks_per_node, alpha
    )

    # Pre-generate walks from observable nodes (for efficiency)
    print(f'Pre-generating walks from {len(train_id)} observable nodes...')
    train_walks = walk_generator.generate_walks(train_id.cpu().tolist())
    print(f'Generated {len(train_walks)} walks')

    if len(train_walks) == 0:
        print("Error: No walks generated from observable nodes!")
        return true_features

    print(f'Training GraphRNA for {epochs} epochs (emb={embedding_dim}, hidden={hidden_dim})...')

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        x_hat = model(train_walks, true_features)

        # Loss on observable nodes only
        loss = criterion(x_hat[train_id], true_features[train_id])

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}')

    # Generate imputed features
    print('Generating imputed features...')
    model.eval()
    with torch.no_grad():
        # Generate walks from all nodes for final prediction
        all_walks = walk_generator.generate_walks(list(range(num_nodes)))
        x_hat = model(all_walks, true_features)

        if is_binary:
            x_hat = torch.sigmoid(x_hat)

    # Combine: keep original for train, use reconstructed for missing
    imputed_features = true_features.clone()
    imputed_features[vali_test_id] = x_hat[vali_test_id]

    print(f'GraphRNA imputation complete. Imputed {len(vali_test_id)} nodes.')
    return imputed_features


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphRNA Baseline')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=72)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--walk_length', type=int, default=10)
    parser.add_argument('--walks_per_node', type=int, default=20)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=100)
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

    imputed_features = train_GraphRNA(adj, true_features, train_id, vali_test_id, args)

    print('\nEvaluating...')
    for topK in [10, 20, 50]:
        recall, ndcg = RECALL_NDCG(
            imputed_features[vali_test_id].cpu().numpy(),
            true_features[vali_test_id].cpu().numpy(),
            topN=topK
        )
        print(f'Recall@{topK}: {recall:.4f}, NDCG@{topK}: {ndcg:.4f}')
