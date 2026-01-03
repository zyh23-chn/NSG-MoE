'''Link prediction
Caution: you have to use GPU to run this code, as I replace .to(device) with .cuda() due to a pytorch issue:
https://github.com/pytorch/pytorch/issues/21819#issuecomment-553310128
If you want to run on CPU, please do some minor modifications by yourself.
'''

import os
import os.path as osp
import hydra
import logging
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
import dgl
from torch_geometric.data import Data
from torch_geometric.nn import GAE
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import negative_sampling
import numpy as np
from tqdm import tqdm, trange

from geo_datasets import LinkPredictionDataset
from ud_models import NSG, NSG_MOE, GCN, GraphSAGE, GAT
from eval_metrics import Evaluator

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

HGNN_MODELS = ['SAGE-h', 'GAT-h', 'HAN', 'HGT']


# @torch.no_grad
# def evaluate(data: Data, model, src, pos, neg):
#     '''src: (num_nodes,), pos: (num_nodes,), neg: (num_nodes, num_neg)'''
#     model.eval()
#     z = model.encode(data.x, data.edge_index)  # node embeddings
#     pos_pred = model.decode(z, torch.vstack((src, pos)))
#     src_repeat = src.unsqueeze(-1).repeat(1, neg.size(1)).view(-1)
#     neg_edge_label_index = torch.vstack((src_repeat, neg.view(-1)))
#     neg_pred = model.decode(z, neg_edge_label_index)
#     evaluator = Evaluator('mrr')
#     y_pred_pos, y_pred_neg = pos_pred, neg_pred.view(neg.size(0), neg.size(1))
#     input_dict = {'y_pred_pos': y_pred_pos, 'y_pred_neg': y_pred_neg}
#     result = evaluator.eval(input_dict)
#     h1 = float(result['hits@1_list'].mean())
#     h3 = float(result['hits@3_list'].mean())
#     h10 = float(result['hits@10_list'].mean())
#     mrr = float(result['mrr_list'].mean())
#     return {'Hits@1': h1, 'Hits@3': h3, 'Hits@10': h10, 'MRR': mrr}


@torch.no_grad
def evaluate(data: Data, model, src0, pos0, neg0, batch_size=1024):
    '''src: (num_nodes,), pos: (num_nodes,), neg: (num_nodes, num_neg)'''
    model.eval()
    z = model.encode(data.x, data.edge_index)  # node embeddings
    num_nodes = src0.size(0)
    n_neg = neg0.size(1)
    h1 = h3 = h10 = mrr = 0.0
    for i in range(0, num_nodes, batch_size):
        src = src0[i : min(i + batch_size, num_nodes)]
        pos = pos0[i : min(i + batch_size, num_nodes)]
        neg = neg0[i : min(i + batch_size, num_nodes)]
        pos_pred = model.decode(z, torch.vstack((src, pos)))
        src_repeat = src.unsqueeze(-1).repeat(1, n_neg).view(-1)
        neg_edge_label_index = torch.vstack((src_repeat, neg.view(-1)))
        neg_pred = model.decode(z, neg_edge_label_index)
        evaluator = Evaluator('mrr')
        y_pred_pos, y_pred_neg = pos_pred, neg_pred.view(neg.size(0), neg.size(1))
        input_dict = {'y_pred_pos': y_pred_pos, 'y_pred_neg': y_pred_neg}
        result = evaluator.eval(input_dict)
        h1 += float(result['hits@1_list'].sum())
        h3 += float(result['hits@3_list'].sum())
        h10 += float(result['hits@10_list'].sum())
        mrr += float(result['mrr_list'].sum())
    h1 /= num_nodes
    h3 /= num_nodes
    h10 /= num_nodes
    mrr /= num_nodes
    return {'Hits@1': h1, 'Hits@3': h3, 'Hits@10': h10, 'MRR': mrr}


def train_one_epoch(data: Data, model, optimizer):
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.edge_index)
    loss = model.recon_loss(z, data.pos_edge_label_index)
    if hasattr(model, 'aux_loss'):
        loss += model.aux_loss
    loss.backward()
    optimizer.step()
    return loss.item()


@hydra.main(config_path='../configs', config_name="config_lp", version_base=None)
def main(cfg: DictConfig):
    ori_dir = hydra.utils.get_original_cwd()
    # print(os.getcwd())
    # print(os.getcwd())
    # print(os.getcwd())
    # print(os.getcwd())
    data_path = osp.join(ori_dir, '../data')
    verbose = True
    device_id = 3
    torch.cuda.set_device(device_id)
    device = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'

    dataset = LinkPredictionDataset(root=osp.join(data_path, cfg.dataset.name), feat_name=cfg.dataset.feat, verbose=verbose, device=device)
    graph = dataset.train_g.cuda()
    edge_split = dataset.edge_split
    val_src, val_pos, val_neg = edge_split['valid']['source_node'], edge_split['valid']['target_node'], edge_split['valid']['target_node_neg']
    test_src, test_pos, test_neg = edge_split['test']['source_node'], edge_split['test']['target_node'], edge_split['test']['target_node_neg']

    # train_g, val_g, test_g = dataset.train_g.cuda(), dataset.val_g.cuda(), dataset.test_g.cuda()
    in_channels = graph.x.size(1)

    h1_list = []
    h3_list = []
    h10_list = []
    mrr_list = []

    hidden_dim = cfg.model.hidden_dim
    num_layers = cfg.model.num_layers
    for run in range(cfg.runs):
        if cfg.model.name == "SAGE":
            model = GraphSAGE(in_channels, hidden_dim, num_layers, hidden_dim)
        elif cfg.model.name == "GCN":
            model = GCN(in_channels, hidden_dim, num_layers, hidden_dim)
        elif cfg.model.name == "GAT":
            model = GAT(in_channels, hidden_dim, num_layers, hidden_dim)
        elif cfg.model.name in HGNN_MODELS:
            '''heterogeneous gnns - NSG'''
            if cfg.model.mode == 'self':
                model = NSG(cfg.model.name, in_channels, hidden_dim, num_layers, hidden_dim, [768], device, et=1)
            elif cfg.model.mode == 'cross':
                model = NSG(cfg.model.name, in_channels, hidden_dim, num_layers, hidden_dim, [768], device, et=2)
            elif cfg.model.mode == 'hybrid':
                model = NSG(cfg.model.name, in_channels, hidden_dim, num_layers, hidden_dim, [768], device, et=3)
            elif cfg.model.mode == 'moe':
                model = NSG_MOE(cfg.model.name, in_channels, hidden_dim, num_layers, hidden_dim, [768], device)
            else:
                raise ValueError(f'{cfg.model.mode}')
        else:
            raise ValueError(f'{cfg.model.name}')
        model = GAE(model).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.lr, weight_decay=cfg.model.weight_decay)

        print(f'Run: {run}')
        for epoch in range(1, cfg.n_epochs + 1):
            loss = train_one_epoch(graph, model, optimizer)
            results = evaluate(graph, model, val_src, val_pos, val_neg)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Hits@10: {results["Hits@10"]:.4f}, Val MRR: {results["MRR"]:.4f}')

        res = evaluate(graph, model, test_src, test_pos, test_neg)
        log.info(f'Test Results for run {run}: Hits@1: {res["Hits@1"]:.4f}, Hits@3: {res["Hits@3"]:.4f}, Hits@10: {res["Hits@10"]:.4f}, MRR: {res["MRR"]:.4f}')
        h1_list.append(res['Hits@1'])
        h3_list.append(res['Hits@3'])
        h10_list.append(res['Hits@10'])
        mrr_list.append(res['MRR'])

    log.info(f'Model: {cfg.model.name}, dataset: {cfg.dataset.name}, feat: {cfg.dataset.feat}')
    log.info(f'Hits@1: {np.mean(h1_list)*100:.2f} ±{np.std(h1_list)*100:.2f}')
    log.info(f'Hits@3: {np.mean(h3_list)*100:.2f} ±{np.std(h3_list)*100:.2f}')
    log.info(f'Hits@10: {np.mean(h10_list)*100:.2f} ±{np.std(h10_list)*100:.2f}')
    log.info(f'MRR: {np.mean(mrr_list)*100:.2f} ±{np.std(mrr_list)*100:.2f}')


if __name__ == '__main__':
    main()
