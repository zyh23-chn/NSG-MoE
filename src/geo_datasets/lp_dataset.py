import os
import os.path as osp
import pandas as pd
import numpy as np
import torch
import dgl
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, negative_sampling, to_undirected, contains_self_loops, remove_self_loops, contains_isolated_nodes


class LinkPredictionDataset(object):
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

        edge_split_path = os.path.join(root, f'lp-edge-split.pt')
        edge_split = torch.load(edge_split_path, map_location=device)
        feat_path = osp.join(root, f'{self.feat_name}_feat.pt')
        feat = torch.load(feat_path, map_location=device)
        self.num_nodes = num_nodes = feat.shape[0]

        train_ei = torch.vstack((edge_split['train']['source_node'], edge_split['train']['target_node']))
        val_ei = torch.vstack((edge_split['valid']['source_node'], edge_split['valid']['target_node']))
        test_ei = torch.vstack((edge_split['test']['source_node'], edge_split['test']['target_node']))
        tv_ei = coalesce(torch.cat([train_ei, val_ei], dim=1))
        full_ei = coalesce(torch.cat([train_ei, val_ei, test_ei], dim=1))

        self.edge_split = edge_split
        self.train_g = Data(x=feat, edge_index=train_ei)
        self.train_g.pos_edge_label_index = train_ei
        # self.train_g.neg_edge_label_index = negative_sampling(
        #     edge_index=train_ei,
        #     num_nodes=num_nodes,
        # )

        self.val_g = Data(x=feat, edge_index=train_ei)
        self.val_g.src = edge_split['valid']['source_node']
        self.val_g.dst_neg = edge_split['valid']['target_node_neg'][:, :50]
        self.val_g.pos_edge_label_index = val_ei
        # self.val_g.neg_edge_label_index = negative_sampling(
        #     edge_index=tv_ei,
        #     num_nodes=num_nodes,
        #     num_neg_samples=val_ei.shape[1],
        # )

        self.test_g = Data(x=feat, edge_index=tv_ei)
        self.test_g.src = edge_split['test']['source_node']
        self.test_g.dst_neg = edge_split['test']['target_node_neg'][:, :50]
        self.test_g.pos_edge_label_index = test_ei
        # self.test_g.neg_edge_label_index = negative_sampling(
        #     edge_index=full_ei,
        #     num_nodes=num_nodes,
        #     num_neg_samples=test_ei.shape[1],
        # )

        # self.train_g = to_undirected(train_g)
        # self.val_g = to_undirected(val_g)
        # self.test_g = to_undirected(test_g)
        # self.val_data = Data(x=feat, edge_index=)
        # self.test_data = Data(x=feat, edge_index=)

    def get_idx_split(self):
        return self.node_split

    def __getitem__(self, idx: int):
        assert idx == 0, 'This dataset has only one graph'
        return self.train_data

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))


# borrowed from OGB
class NodeClassificationEvaluator:
    def __init__(self, eval_metric: str):
        """
        Args:
            eval_metric (str): evaluation metric, can be "rocauc" or "acc".
        """
        self.num_tasks = 1
        self.eval_metric = eval_metric

    def _parse_and_check_input(self, input_dict):
        if self.eval_metric == 'rocauc' or self.eval_metric == 'acc':
            if not 'y_true' in input_dict:
                raise RuntimeError('Missing key of y_true')
            if not 'y_pred' in input_dict:
                raise RuntimeError('Missing key of y_pred')

            y_true, y_pred = input_dict['y_true'], input_dict['y_pred']

            '''
                y_true: numpy ndarray or torch tensor of shape (num_nodes num_tasks)
                y_pred: numpy ndarray or torch tensor of shape (num_nodes num_tasks)
            '''

            # converting to torch.Tensor to numpy on cpu
            if torch is not None and isinstance(y_true, torch.Tensor):
                y_true = y_true.detach().cpu().numpy()

            if torch is not None and isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.detach().cpu().numpy()

            ## check type
            if not (isinstance(y_true, np.ndarray) and isinstance(y_true, np.ndarray)):
                raise RuntimeError('Arguments to Evaluator need to be either numpy ndarray or torch tensor')

            if not y_true.shape == y_pred.shape:
                raise RuntimeError('Shape of y_true and y_pred must be the same')

            if not y_true.ndim == 2:
                raise RuntimeError('y_true and y_pred must to 2-dim arrray, {}-dim array given'.format(y_true.ndim))

            if not y_true.shape[1] == self.num_tasks:
                raise RuntimeError('Number of tasks should be {} but {} given'.format(self.num_tasks, y_true.shape[1]))

            return y_true, y_pred

        else:
            raise ValueError('Undefined eval metric %s ' % (self.eval_metric))

    def eval(self, input_dict):

        if self.eval_metric == 'rocauc':
            y_true, y_pred = self._parse_and_check_input(input_dict)
            return self._eval_rocauc(y_true, y_pred)
        elif self.eval_metric == 'acc':
            y_true, y_pred = self._parse_and_check_input(input_dict)
            return self._eval_acc(y_true, y_pred)
        else:
            raise ValueError('Undefined eval metric %s ' % (self.eval_metric))

    @property
    def expected_input_format(self):
        desc = '==== Expected input format of Evaluator\n'
        if self.eval_metric == 'rocauc':
            desc += '{\'y_true\': y_true, \'y_pred\': y_pred}\n'
            desc += '- y_true: numpy ndarray or torch tensor of shape (num_nodes num_tasks)\n'
            desc += '- y_pred: numpy ndarray or torch tensor of shape (num_nodes num_tasks)\n'
            desc += 'where y_pred stores score values (for computing ROC-AUC),\n'
            desc += 'num_task is {}, and '.format(self.num_tasks)
            desc += 'each row corresponds to one node.\n'
        elif self.eval_metric == 'acc':
            desc += '{\'y_true\': y_true, \'y_pred\': y_pred}\n'
            desc += '- y_true: numpy ndarray or torch tensor of shape (num_nodes num_tasks)\n'
            desc += '- y_pred: numpy ndarray or torch tensor of shape (num_nodes num_tasks)\n'
            desc += 'where y_pred stores predicted class label (integer),\n'
            desc += 'num_task is {}, and '.format(self.num_tasks)
            desc += 'each row corresponds to one node.\n'
        else:
            raise ValueError('Undefined eval metric %s ' % (self.eval_metric))

        return desc

    @property
    def expected_output_format(self):
        desc = '==== Expected output format of Evaluator\n'
        if self.eval_metric == 'rocauc':
            desc += '{\'rocauc\': rocauc}\n'
            desc += '- rocauc (float): ROC-AUC score averaged across {} task(s)\n'.format(self.num_tasks)
        elif self.eval_metric == 'acc':
            desc += '{\'acc\': acc}\n'
            desc += '- acc (float): Accuracy score averaged across {} task(s)\n'.format(self.num_tasks)
        else:
            raise ValueError('Undefined eval metric %s ' % (self.eval_metric))

        return desc

    def _eval_rocauc(self, y_true, y_pred):
        '''
        compute ROC-AUC and AP score averaged across tasks
        '''

        rocauc_list = []

        for i in range(y_true.shape[1]):
            # AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
                is_labeled = y_true[:, i] == y_true[:, i]
                rocauc_list.append(roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i]))

        if len(rocauc_list) == 0:
            raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')

        return {'rocauc': sum(rocauc_list) / len(rocauc_list)}

    def _eval_acc(self, y_true, y_pred):
        acc_list = []

        for i in range(y_true.shape[1]):
            is_labeled = y_true[:, i] == y_true[:, i]
            correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
            acc_list.append(float(np.sum(correct)) / len(correct))

        return {'acc': sum(acc_list) / len(acc_list)}
