"""
SAT (Structure and Attribute Together) for Graph Attribute Imputation

Dual-stream architecture: AE path (attribute) + GAE path (structure)
Uses distribution matching (GAN) to align latent spaces.

Paper: Chen et al., "Learning on attribute-missing graphs", TPAMI 2020

Hyperparameters:
- Hidden dim: 64
- Dropout: 0.5
- Learning rate: 0.005
- lambda_cross: 10.0 (cross-stream loss weight)
- n_gene: 2 (generator training steps)
- n_disc: 1 (discriminator training steps)
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


# ============================================================================
# GNN Layers (from layers.py)
# ============================================================================

class GCNLayer(nn.Module):
    """Simple GCN layer."""

    def __init__(self, in_features, out_features, dropout):
        super(GCNLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.Tensor(in_features, out_features),
                gain=np.sqrt(2.0)
            ),
            requires_grad=True
        )

    def forward(self, input, sp_adj, is_sparse_input=False):
        if is_sparse_input:
            h = torch.spmm(input, self.W)
        else:
            h = torch.mm(input, self.W)
        h_prime = torch.spmm(sp_adj, h)
        return F.elu(h_prime)


class GATLayer(nn.Module):
    """Simple GAT layer."""

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.Tensor(in_features, out_features),
                gain=np.sqrt(2.0)
            ),
            requires_grad=True
        )

        self.a1 = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.Tensor(out_features, 1),
                gain=np.sqrt(2.0)
            ),
            requires_grad=True
        )

        self.a2 = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.Tensor(out_features, 1),
                gain=np.sqrt(2.0)
            ),
            requires_grad=True
        )

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj, is_sparse_input=False):
        if is_sparse_input:
            h = torch.spmm(input, self.W)
        else:
            h = torch.mm(input, self.W)

        N = h.size()[0]

        f_1 = h @ self.a1
        f_2 = h @ self.a2
        e = self.leakyrelu(f_1 + f_2.transpose(0, 1))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        return F.elu(h_prime)


# ============================================================================
# SAT Model (from SAT_models.py)
# ============================================================================

class Discriminator(nn.Module):
    """Discriminator for distribution matching."""

    def __init__(self, n_fts, n_hid, dropout):
        super(Discriminator, self).__init__()
        self.dropout = dropout

        self.fc1 = nn.Linear(n_fts, n_hid)
        self.fc2 = nn.Linear(n_hid, 1)

    def forward(self, x):
        h1 = self.fc1(x)
        h1 = F.dropout(F.relu(h1), self.dropout, training=self.training)
        h2 = self.fc2(h1)
        return h2


class LFI(nn.Module):
    """Learning from Incomplete Information (SAT model)."""

    def __init__(self, n_nodes, n_fts, n_hid, dropout, enc_name='GCN', alpha=0.2):
        super(LFI, self).__init__()
        self.n_fts = n_fts
        self.n_hid = n_hid
        self.dropout = dropout

        # Encoder for AE (attribute path)
        self.ae_fc1 = nn.Linear(n_fts, 200)
        self.ae_fc2 = nn.Linear(200, n_hid)

        # Encoder for GAE (structure path)
        if enc_name == 'GCN':
            self.GCN1 = GCNLayer(n_nodes, 200, dropout=dropout)
            self.GCN2 = GCNLayer(200, n_hid, dropout=dropout)
        elif enc_name == 'GAT':
            self.GCN1 = GATLayer(n_nodes, 200, dropout=dropout, alpha=alpha)
            self.GCN2 = GATLayer(200, n_hid, dropout=dropout, alpha=alpha)

        # Decoder for features
        self.G_ae_fc1 = nn.Linear(n_hid, 200)
        self.G_ae_fc2 = nn.Linear(200, n_fts)

        # Decoder for adjacency
        self.G_gae_fc1 = nn.Linear(n_hid, n_hid)
        self.G_gae_fc2 = nn.Linear(n_hid, n_hid)

        # Discriminator
        self.disc = Discriminator(n_hid, n_hid, dropout)

    def decode_fts(self, z):
        """Decode features from latent representation."""
        fts1 = F.relu(self.G_ae_fc1(z))
        fts1 = F.dropout(fts1, self.dropout, training=self.training)
        fts2 = self.G_ae_fc2(fts1)
        return fts2

    def decode_adj(self, z):
        """Decode adjacency from latent representation."""
        adj_z1 = F.relu(self.G_gae_fc1(z))
        adj_z1 = F.dropout(adj_z1, self.dropout, training=self.training)
        adj_z2 = self.G_gae_fc2(adj_z1)
        return torch.mm(adj_z2, adj_z2.t())

    def forward(self, x, adj, diag_fts):
        # AE path: encode attributes
        x = F.dropout(x, self.dropout, training=self.training)
        ae_h1 = F.relu(self.ae_fc1(x))
        ae_h1 = F.dropout(ae_h1, self.dropout, training=self.training)
        ae_z = self.ae_fc2(ae_h1)

        # GAE path: encode structure
        gae_h1 = self.GCN1(diag_fts, adj, is_sparse_input=True)
        gae_h1 = F.dropout(gae_h1, self.dropout, training=self.training)
        gae_z = self.GCN2(gae_h1, adj)

        # Decode
        ae_fts = self.decode_fts(ae_z)
        gae_fts = self.decode_fts(gae_z)
        ae_adj = self.decode_adj(ae_z)
        gae_adj = self.decode_adj(gae_z)

        return ae_z, ae_fts, ae_adj, gae_z, gae_fts, gae_adj


# ============================================================================
# Loss Functions
# ============================================================================

def fts_loss_discrete(recon_x, x, pos_weight, neg_weight):
    """Feature reconstruction loss for binary features."""
    BCE = nn.BCEWithLogitsLoss(reduction='none')
    output_reshape = recon_x.reshape(-1)
    target_reshape = x.reshape(-1)
    weight_mask = torch.where(target_reshape != 0.0, pos_weight, neg_weight)
    loss_bce = torch.mean(BCE(output_reshape, target_reshape) * weight_mask)
    return loss_bce


def graph_loss_func(graph_recon, pos_indices, neg_indices, pos_values, neg_values):
    """Graph reconstruction loss."""
    BCE = nn.BCEWithLogitsLoss(reduction='none')
    loss_indices = torch.cat([pos_indices, neg_indices], dim=0)
    preds_logits = graph_recon[loss_indices[:, 0], loss_indices[:, 1]]
    labels = torch.cat([pos_values, neg_values])
    loss_bce = torch.mean(BCE(preds_logits, labels))
    return loss_bce


# ============================================================================
# Training Function
# ============================================================================

def train_SAT(adj, true_features, train_id, vali_test_id, args):
    """
    Train SAT model for attribute imputation.

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
    lambda_recon = getattr(args, 'lambda_recon', 1.0)
    lambda_cross = getattr(args, 'lambda_cross', 10.0)
    lambda_gan = getattr(args, 'lambda_gan', 1.0)
    n_gene = getattr(args, 'n_gene', 2)
    n_disc = getattr(args, 'n_disc', 1)
    enc_name = getattr(args, 'enc_name', 'GCN')
    alpha = getattr(args, 'alpha', 0.2)

    num_nodes = true_features.size(0)
    num_features = true_features.size(1)

    # Prepare data
    train_fts = true_features[train_id]

    # Create diagonal feature matrix for GAE input
    indices = torch.LongTensor(np.stack([np.arange(num_nodes), np.arange(num_nodes)], axis=0))
    values = torch.FloatTensor(np.ones(indices.shape[1]))
    diag_fts = torch.sparse.FloatTensor(indices, values, torch.Size([num_nodes, num_nodes])).to(device)

    # Compute normalized adjacency
    if adj.is_sparse:
        norm_adj = adj
    else:
        # Convert to sparse for efficiency
        indices = adj.nonzero(as_tuple=False).t()
        values = adj[indices[0], indices[1]]
        norm_adj = torch.sparse.FloatTensor(indices, values, adj.size()).to(device)

    # Convert to dense for GAT
    if enc_name == 'GAT':
        norm_adj_dense = norm_adj.to_dense() if norm_adj.is_sparse else norm_adj
    else:
        norm_adj_dense = norm_adj

    # Compute class weights for features
    is_binary = (args.dataset in ['cora', 'citeseer', 'amac', 'amap'])
    if is_binary:
        num_pos = (train_fts != 0.0).sum().item()
        num_neg = (train_fts == 0.0).sum().item()
        pos_weight = torch.tensor([num_neg / num_pos], device=device)
        neg_weight = torch.tensor([1.0], device=device)
    else:
        pos_weight = torch.tensor([1.0], device=device)
        neg_weight = torch.tensor([1.0], device=device)

    # Sample negative edges for graph reconstruction
    print('Sampling negative edges...')
    norm_adj_np = norm_adj.to_dense().cpu().numpy() if norm_adj.is_sparse else norm_adj.cpu().numpy()
    pos_indices = np.where(norm_adj_np != 0)
    pos_indices = torch.LongTensor(np.stack(pos_indices, axis=1)).to(device)
    pos_values = torch.ones(pos_indices.size(0), device=device)

    zero_indices = np.where(norm_adj_np == 0)
    neg_sample_size = min(len(zero_indices[0]), pos_indices.size(0))
    neg_idx = np.random.choice(len(zero_indices[0]), neg_sample_size, replace=False)
    neg_indices = torch.LongTensor(np.stack([zero_indices[0][neg_idx], zero_indices[1][neg_idx]], axis=1)).to(device)
    neg_values = torch.zeros(neg_indices.size(0), device=device)

    # Create model
    model = LFI(
        n_nodes=num_nodes,
        n_fts=num_features,
        n_hid=hidden_dim,
        dropout=dropout,
        enc_name=enc_name,
        alpha=alpha
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    BCE = nn.BCEWithLogitsLoss()

    print(f'Training SAT for {epochs} epochs (hidden={hidden_dim}, lambda_cross={lambda_cross})...')

    for epoch in range(1, epochs + 1):
        # ===== Train Generator =====
        for param in model.disc.parameters():
            param.requires_grad = False

        for _ in range(n_gene):
            model.train()
            optimizer.zero_grad()

            ae_z, ae_fts, ae_adj, gae_z, gae_fts, gae_adj = model(
                train_fts, norm_adj_dense, diag_fts
            )

            # Reconstruction losses
            fts_ae_loss = lambda_recon * fts_loss_discrete(ae_fts, train_fts, pos_weight, neg_weight)
            fts_gae_loss = lambda_cross * fts_loss_discrete(gae_fts[train_id], train_fts, pos_weight, neg_weight)
            adj_gae_loss = lambda_recon * graph_loss_func(gae_adj, pos_indices, neg_indices, pos_values, neg_values)

            # GAN losses
            fake_logits_ae = model.disc(ae_z).reshape(-1)
            fake_logits_gae = model.disc(gae_z[train_id]).reshape(-1)
            G_lbls = torch.ones_like(fake_logits_ae)
            G_loss = lambda_gan * (BCE(fake_logits_ae, G_lbls) + BCE(fake_logits_gae, G_lbls))

            gene_loss = fts_ae_loss + fts_gae_loss + adj_gae_loss + G_loss
            gene_loss.backward()
            optimizer.step()

        # ===== Train Discriminator =====
        for param in model.disc.parameters():
            param.requires_grad = True

        for _ in range(n_disc):
            model.train()
            optimizer.zero_grad()

            ae_z, _, _, gae_z, _, _ = model(train_fts, norm_adj_dense, diag_fts)

            # Sample from prior
            prior_z = torch.randn_like(ae_z)

            # Discriminator losses
            real_logits = model.disc(prior_z).reshape(-1)
            fake_logits_ae = model.disc(ae_z.detach()).reshape(-1)
            fake_logits_gae = model.disc(gae_z[train_id].detach()).reshape(-1)

            real_lbls = torch.ones_like(real_logits)
            fake_lbls = torch.zeros_like(fake_logits_ae)

            D_loss = BCE(real_logits, real_lbls) + BCE(fake_logits_ae, fake_lbls) + BCE(fake_logits_gae, fake_lbls)
            D_loss.backward()
            optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}/{epochs}, G Loss: {gene_loss.item():.4f}, D Loss: {D_loss.item():.4f}')

    # Generate imputed features
    print('Generating imputed features...')
    model.eval()
    with torch.no_grad():
        _, _, _, gae_z, gae_fts, _ = model(true_features, norm_adj_dense, diag_fts)

        if is_binary:
            gae_fts = torch.sigmoid(gae_fts)

    # Combine: keep original for train, use reconstructed for missing
    imputed_features = true_features.clone()
    imputed_features[vali_test_id] = gae_fts[vali_test_id]

    print(f'SAT imputation complete. Imputed {len(vali_test_id)} nodes.')
    return imputed_features


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SAT Baseline for Graph Attribute Imputation')
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'amac', 'amap'])
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lambda_recon', type=float, default=1.0)
    parser.add_argument('--lambda_cross', type=float, default=10.0)
    parser.add_argument('--lambda_gan', type=float, default=1.0)
    parser.add_argument('--n_gene', type=int, default=2)
    parser.add_argument('--n_disc', type=int, default=1)
    parser.add_argument('--enc_name', type=str, default='GCN', choices=['GCN', 'GAT'])
    parser.add_argument('--alpha', type=float, default=0.2)
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

    print(f'\n=== SAT Baseline ===')
    print(f'Dataset: {args.dataset}')
    print(f'Device: {device}')
    print(f'Hidden: {args.hidden}, Encoder: {args.enc_name}')

    # Load data
    adj, diff, norm_adj, true_features, node_labels, indices = load_data(args)
    train_id, vali_id, test_id, vali_test_id = data_split(args, adj)

    if args.cuda:
        adj = adj.cuda()
        true_features = true_features.cuda()
        train_id = train_id.cuda()
        vali_test_id = vali_test_id.cuda()
        test_id = test_id.cuda()

    # Train SAT
    start_time = time.time()
    imputed_features = train_SAT(adj, true_features, train_id, vali_test_id, args)
    train_time = time.time() - start_time

    print(f'\nTotal time: {train_time:.2f}s')

    # Evaluate on test set
    test_imputed = imputed_features[test_id].cpu().numpy()
    test_true = true_features[test_id].cpu().numpy()

    print(f'\n=== Evaluation Results ===')
    for topK in [10, 20, 50]:
        recall, ndcg = RECALL_NDCG(test_imputed, test_true, topN=topK)
        print(f'Recall@{topK}: {recall:.4f}, NDCG@{topK}: {ndcg:.4f}')
