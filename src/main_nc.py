'''Node classification'''

import os
import os.path as osp
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
import dgl
from torch_geometric.data import Data, HeteroData

# from torch_geometric.nn import GCN, GraphSAGE, GAT
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_dense_adj, degree
import logging
from tqdm import tqdm
import numpy as np

from geo_datasets import NodeClassificationDataset, magb, abide
from ud_models import NSG, NSG_MOE, GCN, GraphSAGE, GAT

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# BIG_GRAPHS = ['books-nc']  # big graphs that need inductive learning
BIG_GRAPHS = []
HGNN_MODELS = ['SAGE-h', 'HAN', 'HGT']


def train_one_epoch(data: Data, model, optimizer):
    train_mask = data.train_mask
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_mask], data.y[train_mask])
    if hasattr(model, 'aux_loss'):
        loss += model.aux_coef * model.aux_loss
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad
def evaluate(data: Data, model, mask):
    model.eval()

    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = (pred[mask] == data.y[mask]).sum()
    acc = int(correct) / int(mask.sum())
    return float(acc)


@hydra.main(config_path='../configs', config_name="config_nc", version_base=None)
def main(cfg: DictConfig):
    ori_dir = hydra.utils.get_original_cwd()
    # print(os.getcwd())
    # print(os.getcwd())
    # print(os.getcwd())
    # print(os.getcwd())
    data_path = osp.join(ori_dir, '../data')
    verbose = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset_name = cfg.dataset.name
    if dataset_name in ['Movies', 'Toys', 'Grocery', 'Reddit-S', 'Reddit-M']:
        dataset = magb(root=osp.join(ori_dir, '../magb', dataset_name), feat_name=cfg.dataset.feat)
    elif dataset_name == 'ele-fashion':
        dataset = NodeClassificationDataset(root=osp.join(data_path, dataset_name), feat_name=cfg.dataset.feat, verbose=verbose, device=device)
    elif dataset_name == 'ABIDE':
        dataset = abide(root=osp.join(data_path, dataset_name))
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    data = dataset.graph.to(device)
    in_channels = data.x.size(1)

    hidden_dim = cfg.model.hidden_dim
    num_layers = cfg.model.num_layers
    acc_list = []
    for run in range(cfg.runs):
        if cfg.model.name == "SAGE":
            model = GraphSAGE(in_channels, hidden_dim, num_layers, dataset.n_classes, dropout=0.5)
        elif cfg.model.name == "GCN":
            model = GCN(in_channels, hidden_dim, num_layers, dataset.n_classes)
        elif cfg.model.name == "GAT":
            model = GAT(in_channels, hidden_dim, num_layers, dataset.n_classes, heads=8, dropout=0.6)
        elif cfg.model.name in HGNN_MODELS:
            '''heterogeneous gnns - NSG'''
            if cfg.model.mode == 'self':
                model = NSG(cfg.model.name, in_channels, hidden_dim, num_layers, dataset.n_classes, dataset.feat_split, device, et=1)
            elif cfg.model.mode == 'cross':
                model = NSG(cfg.model.name, in_channels, hidden_dim, num_layers, dataset.n_classes, dataset.feat_split, device, et=2)
            elif cfg.model.mode == 'hybrid':
                model = NSG(cfg.model.name, in_channels, hidden_dim, num_layers, dataset.n_classes, dataset.feat_split, device, et=3)
            elif cfg.model.mode == 'moe':
                model = NSG_MOE(cfg.model.name, in_channels, hidden_dim, num_layers, dataset.n_classes, dataset.feat_split, device)
            else:
                raise ValueError(f'{cfg.model.mode}')
        else:
            raise ValueError(f'{cfg.model.name}')
        model = model.to(device)
        print(f'Run: {run}')
        for epoch in range(1, cfg.n_epochs + 1):
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.model.lr, weight_decay=cfg.model.weight_decay)
            loss = train_one_epoch(data, model, optimizer)
            acc = evaluate(data, model, data.val_mask)
            print(f'Epoch: {epoch:03d}, Loss: {loss*100:.2f}, Valid Acc: {acc*100:.2f}')

        if hasattr(model, 'aux_loss'):
            print('Aux loss:', float(model.aux_loss))
        acc = evaluate(data, model, data.test_mask)
        print(f'Test acc: {acc*100:.2f}')
        acc_list.append(acc)
    log.info(f'Model: {cfg.model.name}, dataset: {cfg.dataset.name}' + (f', feat: {cfg.dataset.feat}' if 'feat' in cfg.dataset else ''))
    log.info(f'Final Test Accuracy: {np.mean(acc_list)*100:.2f} ±{np.std(acc_list)*100:.2f}')


if __name__ == '__main__':
    main()
