"""
MATE: Multi-view Graph Imputation Network

Original implementation for attribute imputation on attribute-missing graphs.
This is the baseline MATE model (not the fixed version).
"""

import sys
sys.path.insert(0, '../src')

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops, negative_sampling
from torch_sparse import SparseTensor
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from tqdm import tqdm
import os

from utils import *
import yaml


# ============================================================================
# Loss Functions
# ============================================================================

def fts_rec_loss(recon_x=None, x=None, p_weight=None, n_weight=None):
    BCE = torch.nn.BCEWithLogitsLoss(reduction='none')
    output_fts_reshape = torch.reshape(recon_x, shape=[-1])
    out_fts_lbls_reshape = torch.reshape(x, shape=[-1])
    weight_mask = torch.where(out_fts_lbls_reshape != 0.0, p_weight, n_weight)
    loss_bce = torch.mean(BCE(output_fts_reshape, out_fts_lbls_reshape) * weight_mask)
    return loss_bce


def ce_loss(pos_out, neg_out):
    pos_loss = F.binary_cross_entropy(pos_out.sigmoid(), torch.ones_like(pos_out))
    neg_loss = F.binary_cross_entropy(neg_out.sigmoid(), torch.zeros_like(neg_out))
    return pos_loss + neg_loss


def calc_loss(x, x_aug, temperature=2.0, sym=True):
    batch_size = x.shape[0]
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / (torch.einsum('i,j->ij', x_abs, x_aug_abs) + 1e-8)
    sim_matrix = torch.exp(sim_matrix / temperature)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]

    if sym:
        loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss_0 = - torch.log(loss_0).mean()
        loss_1 = - torch.log(loss_1).mean()
        loss = (loss_0 + loss_1) / 2.0
    else:
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
    return loss


# ============================================================================
# Model Components
# ============================================================================

def creat_gnn_layer(name, first_channels, second_channels, heads):
    if name == "gcn":
        layer = GCNConv(first_channels, second_channels)
    else:
        raise ValueError(name)
    return layer


def creat_activation_layer(activation):
    if activation is None:
        return nn.Identity()
    if activation == "elu":
        return nn.ELU()
    if activation == "relu":
        return nn.ReLU()
    else:
        raise ValueError("Unknown activation")


class GNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, bn=False, layer="gcn", activation="elu", use_node_feats=True):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        bn = nn.BatchNorm1d if bn else nn.Identity
        self.use_node_feats = use_node_feats

        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            heads = 1 if i == num_layers - 1 or 'gat' not in layer else 4
            self.convs.append(creat_gnn_layer(layer, first_channels, second_channels, heads))
            self.bns.append(bn(second_channels * heads))

        self.dropout = nn.Dropout(dropout)
        self.activation = creat_activation_layer(activation)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = self.dropout(x)
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = self.activation(x)
        x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)
        x = self.activation(x)
        return x


class Con_Projector(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1, num_layers=2,
                 dropout=0.5, activation='relu'):
        super().__init__()
        self.proj = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        return self.proj(x)


class Projector(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1, num_layers=2,
                 dropout=0.5, activation='relu'):
        super().__init__()
        self.mlps = nn.ModuleList()

        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.mlps.append(nn.Linear(first_channels, second_channels))

        self.dropout = nn.Dropout(dropout)
        self.activation = creat_activation_layer(activation)

    def forward(self, x):
        for i, mlp in enumerate(self.mlps[:-1]):
            x = self.dropout(x)
            x = mlp(x)
            x = self.activation(x)
        x = self.dropout(x)
        x = self.mlps[-1](x)
        x = self.activation(x)
        return x


class EdgeDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1, num_layers=2,
                 dropout=0.5, activation='relu'):
        super().__init__()
        self.mlps = nn.ModuleList()

        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.mlps.append(nn.Linear(first_channels, second_channels))

        self.dropout = nn.Dropout(dropout)
        self.activation = creat_activation_layer(activation)

    def forward(self, z_1, z_2, edge, sigmoid=True, reduction=False):
        x = z_1[edge[0]] * z_2[edge[1]]

        if reduction:
            x = x.mean(1)

        for i, mlp in enumerate(self.mlps[:-1]):
            x = self.dropout(x)
            x = mlp(x)
            x = self.activation(x)
        x = self.mlps[-1](x)

        if sigmoid:
            return x.sigmoid()
        else:
            return x


class Feature_learner(nn.Module):
    def __init__(self, node, feature):
        super(Feature_learner, self).__init__()
        self.node = node
        self.feature = feature
        self.feature_init = torch.eye(self.node).cuda()
        self.fc = nn.Parameter(torch.randn((self.node, self.feature)))

    def forward(self, x):
        z = torch.mm(self.feature_init, self.fc)
        return z


class Model(nn.Module):
    def __init__(self, encoder, edge_decoder, projector, con_projector, temp,
                 pos_weight_tensor, neg_weight_tensor, mask=None,
                 random_negative_sampling=False, loss="ce"):
        super().__init__()
        self.encoder = encoder
        self.edge_decoder = edge_decoder
        self.projector = projector
        self.con_projector = con_projector
        self.mask = mask
        self.temp = temp
        self.pos_weight_tensor = pos_weight_tensor
        self.neg_weight_tensor = neg_weight_tensor

        if loss == "ce":
            self.loss_edgefn = ce_loss
        else:
            raise ValueError(loss)

        self.contrastive_loss = calc_loss
        self.rec_loss = fts_rec_loss

        if random_negative_sampling:
            self.negative_sampler = lambda edge_index, num_nodes, num_neg_samples: \
                torch.randint(0, num_nodes, size=(2, num_neg_samples)).to(edge_index)
        else:
            self.negative_sampler = negative_sampling

    def forward(self, data_1, data_2, norm_adj, feature_learner, train_fts_idx, vali_test_fts_idx):
        x_1_, edge_index_1 = data_1.x, data_1.edge_index
        x_learn = feature_learner(x_1_)
        zero_ = torch.zeros_like(x_learn, device=x_learn.device)
        zero = torch.zeros_like(x_learn, device=x_learn.device)
        zero[vali_test_fts_idx] = zero_[vali_test_fts_idx] + x_learn[vali_test_fts_idx]
        x_1__ = x_1_ + zero
        x_1 = torch.mm(norm_adj, x_1__)
        x_2, edge_index_2 = data_2.x, data_2.edge_index

        z_1 = self.encoder(x_1, edge_index_1)
        z_2 = self.encoder(x_2, edge_index_1)
        z = (z_1 + z_2) * 0.5
        out = self.projector(z)
        return out

    def train_one_epoch(self, data_1, data_2, norm_adj, feature_learner, train_fts_idx,
                       vali_test_fts_idx, batch_size=2 ** 16):
        x_1_, edge_index_1 = data_1.x, data_1.edge_index
        x_learn = feature_learner(x_1_)
        zero_ = torch.zeros_like(x_learn, device=x_learn.device)
        zero = torch.zeros_like(x_learn, device=x_learn.device)
        zero[vali_test_fts_idx] = zero_[vali_test_fts_idx] + x_learn[vali_test_fts_idx]
        x_1__ = x_1_ + zero
        x_1 = torch.mm(norm_adj, x_1__)
        x_2, edge_index_2 = data_2.x, data_2.edge_index
        remaining_edges, masked_edges = self.mask(edge_index_1)

        aug_edge_index, _ = add_self_loops(edge_index_1)
        neg_edges = self.negative_sampler(
            aug_edge_index,
            num_nodes=data_1.num_nodes,
            num_neg_samples=masked_edges.view(2, -1).size(1),
        ).view_as(masked_edges)

        for perm in DataLoader(range(masked_edges.size(1)), batch_size=batch_size, shuffle=True):
            z_1 = self.encoder(x_1, remaining_edges)
            z_2 = self.encoder(x_2, remaining_edges)

            batch_masked_edges = masked_edges[:, perm]
            batch_neg_edges = neg_edges[:, perm]

            pos_out_1 = self.edge_decoder(z_1, z_2, batch_masked_edges, sigmoid=False)
            neg_out_1 = self.edge_decoder(z_1, z_2, batch_neg_edges, sigmoid=False)
            pos_out_2 = self.edge_decoder(z_2, z_1, batch_masked_edges, sigmoid=False)
            neg_out_2 = self.edge_decoder(z_2, z_1, batch_neg_edges, sigmoid=False)

            loss_edge = (self.loss_edgefn(pos_out_1, neg_out_1) + self.loss_edgefn(pos_out_2, neg_out_2))

            z_1_p = z_1
            z_2_p = z_2
            loss_con = self.contrastive_loss(z_1_p, z_2_p, temperature=self.temp)
            z = (z_1 + z_2) * 0.5

            x_recon = self.projector(z)
            loss_recon = self.rec_loss(x_recon[train_fts_idx], x_1_[train_fts_idx],
                                      self.pos_weight_tensor, self.neg_weight_tensor)

            loss_total = loss_edge + loss_con + loss_recon

        return loss_total


# ============================================================================
# Training Function
# ============================================================================

def train_MATE(adj, true_features, train_id, vali_id, test_id, vali_test_id, args):
    """
    Train original MATE model for attribute imputation.

    Args:
        adj: Sparse adjacency matrix
        true_features: [N, F] tensor with ground truth features
        train_id: Indices of observable nodes (40%)
        vali_id: Indices of validation nodes (10%)
        test_id: Indices of test nodes (50%)
        vali_test_id: Indices of all missing nodes (60% = vali + test)
        args: Argparse namespace

    Returns:
        imputed_features: [N, F] tensor with imputed values
    """
    device = true_features.device

    # Load best configs from configs.yml (like original main.py does)
    args = load_best_configs(args, "../configs.yml")

    # Hyperparameters (from configs or defaults)
    epochs = getattr(args, 'epoch', 3000)  # Note: configs uses 'epoch' not 'epochs'
    lr = getattr(args, 'lr', 0.001)
    weight_decay = getattr(args, 'weight_decay', 5e-5)
    p = getattr(args, 'p', 0.8)
    encoder_channels = getattr(args, 'encoder_channels', 512)
    hidden_channels = getattr(args, 'hidden_channels', 256)
    decoder_channels = getattr(args, 'decoder_channels', 64)
    encoder_layers = getattr(args, 'encoder_layers', 2)
    eproj_layer = getattr(args, 'eproj_layer', 2)
    decoder_layers = getattr(args, 'decoder_layers', 2)
    encoder_dropout = getattr(args, 'encoder_dropout', 0.4)
    eproj_dropout = getattr(args, 'eproj_dropout', 0.6)
    decoder_dropout = getattr(args, 'decoder_dropout', 0.8)
    temp = getattr(args, 'temp', 0.2)

    # Ensure generative_flag for load_data
    if not hasattr(args, 'generative_flag'):
        args.generative_flag = True

    # Load data
    _, diff, norm_adj, _, node_labels, indices = load_data(args)

    # Prepare views
    x_view1_ = true_features.clone()
    x_view1_[vali_test_id] = 0.0
    data_1 = Data(x=x_view1_, y=node_labels, edge_index=indices)

    x_view_ = true_features.clone()
    x_view_[vali_test_id] = 0.0
    x_view_ = x_view_.to(device)
    diff = diff.to(device)
    x_view2 = torch.mm(diff, x_view_).cpu()

    data_2 = Data(x=x_view2, y=node_labels, edge_index=indices)

    # Loss weights
    _, pos_weight_tensor, neg_weight_tensor = loss_weight(args, true_features, train_id)

    # Initialize models
    mask = MaskEdge(p=p)
    feature_learn = Feature_learner(true_features.size(0), true_features.size(1))

    encoder = GNNEncoder(data_1.num_features, encoder_channels, hidden_channels,
                        num_layers=encoder_layers, dropout=encoder_dropout,
                        bn=False, layer='gcn', activation='elu')

    edge_decoder = EdgeDecoder(hidden_channels, decoder_channels,
                              num_layers=eproj_layer, dropout=eproj_dropout)

    projector = Projector(hidden_channels, encoder_channels, out_channels=data_1.num_features,
                         num_layers=decoder_layers, dropout=decoder_dropout)

    con_projector = Con_Projector(hidden_channels, encoder_channels, out_channels=data_1.num_features,
                                 num_layers=decoder_layers, dropout=decoder_dropout)

    model = Model(encoder, edge_decoder, projector, con_projector, temp,
                 pos_weight_tensor, neg_weight_tensor, mask)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_learner = optim.Adam(feature_learn.parameters(), lr=1e-3, weight_decay=weight_decay)

    # Learning rate scheduler (cosine annealing, like original)
    def scheduler_fn(epoch):
        return (1 + np.cos((epoch) * np.pi / epochs)) * 0.5
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_fn)

    # Move to device
    data_1 = data_1.to(device)
    data_2 = data_2.to(device)
    model = model.to(device)
    feature_learn = feature_learn.to(device)
    norm_adj = norm_adj.to(device)

    print(f'Training MATE for {epochs} epochs...')

    # Training loop with validation and model selection (like original)
    best_val_recall = 0.0
    best_model_state = None
    best_learner_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        feature_learn.train()

        loss = model.train_one_epoch(data_1, data_2, norm_adj, feature_learn,
                                    train_id, vali_test_id)

        optimizer.zero_grad()
        optimizer_learner.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_learner.step()
        scheduler.step()  # Update learning rate

        # Validation every 20 epochs (like original)
        if epoch % 20 == 0:
            model.eval()
            feature_learn.eval()
            with torch.no_grad():
                x_hat_val = model(data_1, data_2, norm_adj, feature_learn, train_id, vali_test_id)
                val_recall, val_ndcg = RECALL_NDCG(
                    x_hat_val[vali_id].cpu().numpy(),
                    true_features[vali_id].cpu().numpy(),
                    topN=50
                )

            if val_recall > best_val_recall:
                best_val_recall = val_recall
                # Save best model state (in memory, not to disk)
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_learner_state = {k: v.cpu().clone() for k, v in feature_learn.state_dict().items()}

            if epoch % 100 == 0:
                print(f'Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}, Val Recall@50: {val_recall:.4f}')

    # Load best model (like original)
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        feature_learn.load_state_dict({k: v.to(device) for k, v in best_learner_state.items()})
        print(f'Loaded best model with validation Recall@50: {best_val_recall:.4f}')

    # Generate imputed features
    print('Generating imputed features...')
    model.eval()
    feature_learn.eval()
    with torch.no_grad():
        x_hat = model(data_1, data_2, norm_adj, feature_learn, train_id, vali_test_id)

    # Combine: keep original for train, use reconstructed for missing
    imputed_features = true_features.clone()
    imputed_features[vali_test_id] = x_hat[vali_test_id]

    print(f'MATE imputation complete. Imputed {len(vali_test_id)} nodes.')
    return imputed_features


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='MATE Baseline')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=72)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--train_fts_ratio', type=float, default=0.4)
    parser.add_argument('--generative_flag', type=bool, default=True)

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

    imputed_features = train_MATE(adj, true_features, train_id, vali_test_id, args)

    print('\nEvaluating...')
    for topK in [10, 20, 50]:
        recall, ndcg = RECALL_NDCG(
            imputed_features[vali_test_id].cpu().numpy(),
            true_features[vali_test_id].cpu().numpy(),
            topN=topK
        )
        print(f'Recall@{topK}: {recall:.4f}, NDCG@{topK}: {ndcg:.4f}')
