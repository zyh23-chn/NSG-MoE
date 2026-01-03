import os
import os.path as osp
import pandas as pd
import numpy as np
import torch
import dgl
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


class NodeClassificationDataset(object):
    def __init__(self, root: str, feat_name: str, verbose: bool = True, device: str = "cpu"):
        """
        Args:
            root (str): root directory to store the dataset folder.
            feat_name (str): the name of the node features, e.g., "t5vit".
            verbose (bool): whether to print the information.
            device (str): device to use.
        """
        root = osp.normpath(root)
        self.name = osp.basename(root)
        self.verbose = verbose
        self.root = root
        self.feat_name = feat_name
        self.device = device
        if self.verbose:
            print(f"Dataset name: {self.name}")
            print(f'Feature name: {self.feat_name}')
            print(f'Device: {self.device}')

        edge_path = osp.join(root, 'nc_edges-nodeid.pt')
        self.edge = torch.tensor(torch.load(edge_path, weights_only=True), dtype=torch.int64).to(self.device)
        feat_path = osp.join(root, f'{self.feat_name}_feat.pt')
        feat = torch.load(feat_path, map_location=self.device, weights_only=True)
        self.num_nodes = feat.shape[0]

        self.graph = Data(x=feat, edge_index=self.edge.t())
        # src, dst = self.edge.t()[0], self.edge.t()[1]
        # self.graph = dgl.graph((src, dst), num_nodes=self.num_nodes).to(self.device)
        # self.graph.ndata['feat'] = feat

        labels_path = osp.join(root, 'labels-w-missing.pt')
        labels = torch.tensor(torch.load(labels_path, weights_only=True), dtype=torch.int64).to(self.device)
        # self.graph.ndata['label'] = self.labels
        self.labels = labels
        self.graph.y = labels
        self.n_classes = labels.max().item() + 1

        node_split_path = osp.join(root, 'split.pt')
        self.node_split = torch.load(node_split_path, weights_only=True)

        train_mask = torch.zeros(self.num_nodes, dtype=torch.bool).to(self.device)
        val_mask = torch.zeros(self.num_nodes, dtype=torch.bool).to(self.device)
        test_mask = torch.zeros(self.num_nodes, dtype=torch.bool).to(self.device)

        train_mask[self.node_split['train_idx']] = True
        val_mask[self.node_split['val_idx']] = True
        test_mask[self.node_split['test_idx']] = True

        # self.graph.ndata['train_mask'] = train_mask
        # self.graph.ndata['val_mask'] = val_mask
        # self.graph.ndata['test_mask'] = test_mask
        self.graph.train_mask = train_mask
        self.graph.val_mask = val_mask
        self.graph.test_mask = test_mask
        self.feat_split=[768]

    def get_idx_split(self):
        return self.node_split

    def __getitem__(self, idx: int):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))


def split_set(nodes_num, train_ratio=0.6, val_ratio=0.2):
    np.random.seed(42)
    indices = np.random.permutation(nodes_num)
    train_size = int(nodes_num * train_ratio)
    val_size = int(nodes_num * val_ratio)

    train_mask = torch.zeros(nodes_num, dtype=torch.bool)
    val_mask = torch.zeros(nodes_num, dtype=torch.bool)
    test_mask = torch.zeros(nodes_num, dtype=torch.bool)
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size : train_size + val_size]] = True
    test_mask[indices[train_size + val_size :]] = True
    return train_mask, val_mask, test_mask


def split_set_id(nodes_num, train_ratio=0.6, val_ratio=0.2):
    np.random.seed(42)
    indices = np.random.permutation(nodes_num)
    train_size = int(nodes_num * train_ratio)
    val_size = int(nodes_num * val_ratio)

    train_ids = indices[:train_size]
    val_ids = indices[train_size : train_size + val_size]
    test_ids = indices[train_size + val_size :]

    return train_ids, val_ids, test_ids


class magb:
    def __init__(self, root, feat_name):
        dataset_name = osp.basename(root)
        dataset_name=dataset_name.replace('-', '')
        graph = dgl.load_graphs(osp.join(root, dataset_name + 'Graph.pt'))[0][0]
        if feat_name == 'llama':
            feat_file = dataset_name + '_Llama-3.2-11B-Vision-Instruct_tv.npy'
            feat = np.load(osp.join(root, 'MMFeature', feat_file), allow_pickle=True)
        elif feat_name == 'qwen':
            feat_file = dataset_name + '_Qwen2-VL-7B-Instruct_tv.npy'
            feat = np.load(osp.join(root, 'MMFeature', feat_file), allow_pickle=True)
        elif feat_name == 'clip':
            feat_file = dataset_name + '_LLAMA8B_CLIP.npy'
            feat = np.load(osp.join(root, 'MMFeature', feat_file), allow_pickle=True)
            self.feat_split = [4096]  # text, visual 4096+768=4864
        else:
            raise ValueError(f'Unknown feature name: {feat_name}')

        label=torch.tensor(graph.ndata['label'], dtype=torch.int64)
        self.n_classes = label.max().item() + 1
        self.graph = Data(x=torch.tensor(feat, dtype=torch.float32), edge_index=torch.tensor(torch.vstack(graph.edges()), dtype=torch.int64), y=label)
        self.graph.train_mask, self.graph.val_mask, self.graph.test_mask = split_set(graph.num_nodes())


class abide:
    def __init__(self, root, k=10):
        graph = np.load(osp.join(root, 'ABIDE_weighted-cosine_graph.npz'), allow_pickle=True)
        # construct kNN graph
        adj = graph['adj']
        np.fill_diagonal(adj, -np.inf)
        n_nodes = adj.shape[0]
        knn_idx = np.argsort(-adj, axis=1)[:, :k]
        row_idx = np.repeat(np.arange(n_nodes), k)
        col_idx = knn_idx.flatten()
        edge_index = np.stack([row_idx, col_idx], axis=0)
        edge_index = to_undirected(torch.tensor(edge_index, dtype=torch.int64))

        label=torch.tensor(graph['label'], dtype=torch.int64)
        self.n_classes = label.max().item() + 1
        self.graph = Data(x=torch.tensor(graph['feat'], dtype=torch.float32), edge_index=edge_index, y=label)
        self.graph.train_mask, self.graph.val_mask, self.graph.test_mask = split_set(n_nodes)
        self.feat_split = [48, 54, 64]
