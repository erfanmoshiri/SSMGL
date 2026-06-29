import argparse
from torch import optim
import sys
sys.path.insert(0, '..')
from models.MATE_fixed import *
from src.utils import *
import warnings
import random
from tqdm import tqdm
from torch_geometric.data import Data
import torch

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--method_name', type=str, default='Model_Fixed')
parser.add_argument('--topK_list', type=list, default=[10, 20, 50])
parser.add_argument('--seed', type=int, default=72)
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--p', type=float, default=0.7)
parser.add_argument('--weight_decay', type=float, default=5e-5)
parser.add_argument('--train_fts_ratio', type=float, default=0.4)
parser.add_argument('--generative_flag', type=bool, default=True)
parser.add_argument('--cuda', action='store_true',
                    default=torch.cuda.is_available())

parser.add_argument("--layer", nargs="?", default="gcn",
                    help="GNN layer, (default: gcn)")
parser.add_argument("--encoder_activation", nargs="?", default="elu",
                    help="Activation function for GNN encoder, (default: elu)")
parser.add_argument('--encoder_channels', type=int, default=128,
                    help='Channels of GNN encoder layers. (default: 128)')
parser.add_argument('--hidden_channels', type=int, default=64,
                    help='Channels of hidden representation. (default: 64)')
parser.add_argument('--decoder_channels', type=int, default=32,
                    help='Channels of decoder layers. (default: 32)')
parser.add_argument('--encoder_layers', type=int, default=2,
                    help='Number of layers for encoder. (default: 2)')
parser.add_argument('--eproj_layer', type=int, default=2,
                    help='Number of layers for edge_projector. (default: 2)')
parser.add_argument('--decoder_layers', type=int, default=2,
                    help='Number of layers for decoders. (default: 2)')
parser.add_argument('--encoder_dropout', type=float, default=0.8,
                    help='Dropout probability of encoder. (default: 0.8)')
parser.add_argument('--eproj_dropout', type=float, default=0.2,
                    help='Dropout probability of edge_projector. (default: 0.2)')
parser.add_argument('--decoder_dropout', type=float, default=0.2,
                    help='Dropout probability of decoder. (default: 0.2)')
parser.add_argument('--bn', type=bool, default=False)
parser.add_argument('--device', type=int, default=3)
parser.add_argument('--alpha', type=int, default=1)
parser.add_argument('--beta', type=int, default=1)
parser.add_argument('--temp', type=float, default=0.2)


def main(args):
    from train_utils import create_run_folder, save_config, save_epoch_metrics, save_final_results

    # Create timestamped run folder
    run_dir = create_run_folder(args.dataset, model_type="fixed")
    save_config(run_dir, args)

    set_random_seed(72)
    adj, diff, norm_adj, true_features, node_labels, indices = load_data(args)

    Adj, Diag, Ture_feature, A_temp = input_matrix(
        args, adj, norm_adj, true_features)
    train_id, vali_id, test_id, vali_test_id = data_split(args, adj)


    x_view1_ = true_features
    x_view1_[vali_test_id] = 0.0
    data_1 = Data(x=x_view1_, y=node_labels, edge_index=indices)


    x_view_ = true_features
    x_view_[vali_test_id] = 0.0
    x_view_ = x_view_.cuda()
    diff = diff.cuda()
    x_view2 = torch.mm(diff, x_view_).cpu()

    data_2 = Data(x=x_view2, y=node_labels, edge_index=indices)
    fts_loss_func, pos_weight_tensor, neg_weight_tensor = loss_weight(
        args, true_features, train_id)

    set_random_seed(args.seed)
    mask = MaskEdge(p=args.p)

    # FIXED FEATURES: Compute once before training
    # Uses FULL adjacency but only observable node features (realistic scenario)
    print('Computing fixed features using neighbor averaging...')
    print(f'  - Using FULL graph structure')
    print(f'  - Averaging only from {len(train_id)} observable nodes ({args.train_fts_ratio*100:.0f}%)')

    fixed_features = compute_fixed_features(adj, true_features, train_id, vali_test_id)
    print(f'Fixed features computed for {len(vali_test_id)} missing nodes')

    # Compute neighborhood density SSL target (once before training)
    print('Computing neighborhood density for SSL objective...')
    target_density = compute_neighborhood_density(adj, true_features, train_id)
    print(f'Neighborhood density computed for all nodes')

    encoder = GNNEncoder(data_1.num_features, args.encoder_channels, args.hidden_channels,
                         num_layers=args.encoder_layers, dropout=args.encoder_dropout,
                         bn=args.bn, layer=args.layer, activation=args.encoder_activation)

    edge_decoder = EdgeDecoder(args.hidden_channels, args.decoder_channels,
                               num_layers=args.eproj_layer, dropout=args.eproj_dropout)

    projector = Projector(args.hidden_channels, args.encoder_channels, out_channels=data_1.num_features,
                          num_layers=args.decoder_layers, dropout=args.decoder_dropout)

    con_projector = Con_Projector(args.hidden_channels, args.encoder_channels, out_channels=data_1.num_features,
                          num_layers=args.decoder_layers, dropout=args.decoder_dropout)

    model = Model(encoder, edge_decoder, projector, con_projector, args.temp, pos_weight_tensor, neg_weight_tensor, mask,
                  feature_dim=data_1.num_features, hidden_dim=args.hidden_channels)

    # ONLY ONE OPTIMIZER - no optimizer_learner!
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)

    def scheduler(epoch): return (
        1 + np.cos((epoch) * np.pi / args.epoch)) * 0.5
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=scheduler)

    if args.cuda:
        data_1 = data_1.cuda()
        data_2 = data_2.cuda()
        model = model.cuda()
        fixed_features = fixed_features.cuda()
        target_density = target_density.cuda()
        norm_adj = norm_adj.cuda()

    eva_values_list = []
    best = 0.0

    # Track losses
    loss_history = {
        'total': [],
        'edge': [],
        'contrastive': [],
        'reconstruction': [],
        'density': []
    }

    print('---------------------start training (FIXED features)------------------------')
    for epoch in tqdm(range(1, 1 + args.epoch)):
        model.train()

        # NO feature_learn.train() - features are FIXED!
        loss_total, loss_edge, loss_con, loss_recon, loss_density = model.train_one_epoch(
            data_1, data_2, norm_adj, fixed_features, train_id, vali_test_id, target_density=target_density)

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        scheduler.step()

        # Track losses every epoch
        loss_history['total'].append(loss_total.item())
        loss_history['edge'].append(loss_edge.item())
        loss_history['contrastive'].append(loss_con.item())
        loss_history['reconstruction'].append(loss_recon.item())
        loss_history['density'].append(loss_density.item())

        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                X_hat = model(data_1, data_2, norm_adj, fixed_features, train_id, vali_test_id)
            gene_fts = X_hat[vali_id].cpu().numpy()
            gt_fts = Ture_feature[vali_id].cpu().numpy()
            avg_recall, avg_ndcg = RECALL_NDCG(
                gene_fts, gt_fts, topN=args.topK_list[2])
            eva_values_list.append(avg_recall)

            # Log epoch metrics
            save_epoch_metrics(run_dir, epoch, {
                'loss_total': loss_total.item(),
                'loss_edge': loss_edge.item(),
                'loss_contrastive': loss_con.item(),
                'loss_reconstruction': loss_recon.item(),
                'loss_density': loss_density.item(),
                'recall': avg_recall,
                'ndcg': avg_ndcg
            })

            if eva_values_list[-1] > best:
                torch.save(model.state_dict(),
                           os.path.join('..', 'best_model',
                                        'final_model_fixed_{}_{}.pkl'.format(args.dataset, args.train_fts_ratio)))
                # Save loss history
                torch.save(loss_history, os.path.join('..', 'best_model',
                                        'loss_history_fixed_{}_{}.pkl'.format(args.dataset, args.train_fts_ratio)))
                best = eva_values_list[-1]


    model.load_state_dict(
        torch.load(os.path.join('..', 'best_model', 'final_model_fixed_{}_{}.pkl'.format(args.dataset, args.train_fts_ratio))))

    model.eval()
    recall_50, ndcg_50 = test_model_fixed(args, model, norm_adj, fixed_features, Ture_feature,
                                data_1, data_2, train_id, vali_id, vali_test_id, test_id)
    with torch.no_grad():
        x_hat = model(data_1, data_2, norm_adj, fixed_features, train_id, vali_test_id)
        gene_data = x_hat[test_id]
        labels_of_gene = node_labels[test_id]
    adj = adj.to_dense()
    acc_x = test_X(gene_data.cpu().numpy(), labels_of_gene.cpu().numpy())
    acc_ax = test_AX(gene_data.cpu().numpy(), labels_of_gene.cpu().numpy(), adj[test_id, :][:, test_id].cpu().numpy())

    # Save final test results
    save_final_results(run_dir, {
        'recall@50': recall_50,
        'ndcg@50': ndcg_50,
        'accuracy_X': acc_x,
        'accuracy_AX': acc_ax,
        'dataset': args.dataset,
        'train_ratio': args.train_fts_ratio
    })

    # Print final loss summary
    print("\n" + "="*60)
    print("LOSS SUMMARY (FIXED Features)")
    print("="*60)
    print(f"Final Total Loss: {loss_history['total'][-1]:.4f}")
    print(f"Final Edge Loss: {loss_history['edge'][-1]:.4f}")
    print(f"Final Contrastive Loss: {loss_history['contrastive'][-1]:.4f}")
    print(f"Final Reconstruction Loss: {loss_history['reconstruction'][-1]:.4f}")
    print(f"Final Density Loss: {loss_history['density'][-1]:.4f}")
    print("="*60)


def test_model_fixed(args, model, norm_adj, fixed_features, T, data_1, data_2, train_id, vali_id, vali_test_id, test_id):
    """Test function for fixed features model"""
    print('Loading well-trained model (FIXED features)')

    model.load_state_dict(
        torch.load(os.path.join('..', 'best_model', 'final_model_fixed_{}_{}.pkl'.format(args.dataset, args.train_fts_ratio))))

    model.eval()

    with torch.no_grad():
        X_hat = model(data_1, data_2, norm_adj, fixed_features, train_id, vali_test_id)
    gene_fts = X_hat[test_id]

    reture_recall = 0.0
    reture_ndcg = 0.0
    print('Profiling performance on {}:'.format(args.dataset))
    if args.cuda:
        gene_fts = gene_fts.data.cpu().numpy()
        gt_fts = T[test_id].cpu().numpy()
    else:
        gene_fts = gene_fts.data.numpy()
        gt_fts = T[test_id].numpy()
    for topK in args.topK_list:
        avg_recall, avg_ndcg = RECALL_NDCG(gene_fts, gt_fts, topN=topK)
        print('topK: {}, recall: {}, ndcg: {}'.format(topK, avg_recall, avg_ndcg))
        if topK == 50:
            reture_recall = avg_recall
            reture_ndcg = avg_ndcg
    save_generative_fts(args, gene_fts, T, train_id, vali_id, test_id)
    if args.cuda:
        T = T.cpu().data.numpy()
    else:
        T = T.data.numpy()

    return reture_recall, reture_ndcg


if __name__ == "__main__":
    args = parser.parse_args()
    args = load_best_configs(args, "../configs.yml")
    print(args)
    torch.cuda.set_device(f'cuda:{args.device}')
    main(args)
