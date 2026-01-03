import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import coalesce
from itertools import combinations
from copy import deepcopy
import numpy as np
import torch.distributions as dist
from typing import Dict, List, Union
from torch_geometric.utils import to_undirected
from tqdm import tqdm
from multiprocessing import Process
import time

# from .hgnns import SAGE_h, HAN, HGT
from .hgnns import SAGE_h, GAT_h, HAN, HGT
from g_utils import get_adj, save_with_pickle, load_with_pickle, maximum_spanning_tree, get_directed_ei, pairwise_cos_sim

DEBUG = False
N_SAMPLES = 100
SPAR = False  # apply an additional linear in front and the sparsification mechanism
TIMER=False


class BaseNSG(nn.Module):
    def __init__(self, model_name, in_channels, hidden_dim, n_layers, out_channels, fea_split: List, device):
        super().__init__()
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.out_channels = out_channels
        self.fea_split = fea_split = [0] + fea_split
        self.n_type = n_type = len(fea_split)
        # self.tp_list = tp_list = [str(i) for i in range(n_type)]
        tp_list = [str(i) for i in range(n_type)]  # use str(i) as the name of the modality
        self.in_channels = {}
        self.lin_dict = nn.ModuleDict()
        for i, tp in enumerate(tp_list):
            in_channel = (fea_split[i + 1] if i + 1 < n_type else in_channels) - fea_split[i]
            if SPAR:
                self.lin_dict[tp] = nn.Linear(in_channel, hidden_dim)
                self.in_channels[tp] = hidden_dim
            else:
                self.in_channels[tp] = in_channel

        dumb = HeteroData()
        for tp in tp_list:
            dumb[tp].x = None
            dumb[(tp, tp)].edge_index = None
        for tp0, tp1 in combinations(tp_list, 2):
            dumb[(tp0, tp1)].edge_index = None
            dumb[(tp1, tp0)].edge_index = None
        self.metadata = dumb.metadata()
        self.device = device
        self.fc = nn.LazyLinear(self.out_channels)

    def _get_hdata(self, x, edge_index, et=1):
        '''
        et
        1: self connection
        2: cross connection
        3: both type
        '''
        n_nodes = x.size(0)
        hdata = HeteroData()
        fea_split, n_type, tp_list = self.fea_split, self.n_type, self.metadata[0]
        '''self-type connection'''
        for i, tp in enumerate(tp_list):
            hdata[tp].x = x[:, fea_split[i] : (fea_split[i + 1] if i + 1 < n_type else None)]
            if et == 1 or et == 3:
                hdata[(tp, tp)].edge_index = edge_index.clone()
        intra_ei = torch.tensor([list(range(n_nodes)) for _ in range(2)]).long().to(edge_index.device)
        for tp0, tp1 in combinations(tp_list, 2):
            if et == 2 or et == 3:
                hdata[(tp0, tp1)].edge_index = coalesce(torch.hstack((edge_index, intra_ei)))
                hdata[(tp1, tp0)].edge_index = coalesce(torch.hstack((edge_index, intra_ei)))
            else:
                hdata[(tp0, tp1)].edge_index = intra_ei
                hdata[(tp1, tp0)].edge_index = intra_ei
        return hdata

    def _get_hdata_spar(self, x, edge_index, et=1, k=2):
        '''
        get sparsified NSG
        edge_index: undirected (symmetric)
        k: top-k neighbors of cross-type connections
        '''
        n_nodes = x.size(0)
        hdata = HeteroData()
        fea_split, n_type, tp_list = self.fea_split, self.n_type, self.metadata[0]
        n_mod = len(fea_split)  # num of modalities
        k = min(k, n_mod - 1)
        '''self-type connection'''
        for i, tp in enumerate(tp_list):
            hdata[tp].x = self.lin_dict[tp](x[:, fea_split[i] : (fea_split[i + 1] if i + 1 < n_type else None)]).relu_()
            if et in (1, 3):
                hdata[(tp, tp)].edge_index = edge_index.clone()
        for tp0, tp1 in combinations(tp_list, 2):
            hdata[(tp0, tp1)].edge_index = []
            if tp0 != tp1:
                hdata[(tp1, tp0)].edge_index = []

        '''intra-node connection'''
        for i in tqdm(range(n_nodes)):
            tmp = torch.stack([hdata[tp].x[i] for tp in tp_list], dim=0)
            sim_mat = pairwise_cos_sim(tmp, tmp)
            eds = maximum_spanning_tree(sim_mat.detach().cpu().numpy())
            for u, v in eds:
                hdata[(tp_list[u], tp_list[v])].edge_index.append((i, i))
                hdata[(tp_list[v], tp_list[u])].edge_index.append((i, i))

        '''cross-type connection'''
        if et in (2, 3):
            ...
        for ed_type, ei in hdata.edge_index_dict.items():
            # exlude self-type here
            if ed_type[0] != ed_type[-1]:
                hdata[ed_type].edge_index = torch.tensor(ei).t().long().to(edge_index.device)

        return hdata


class NSG(BaseNSG):
    def __init__(self, *args, et=1):
        super().__init__(*args)

        self.et = et
        if self.model_name == "SAGE-h":
            self.gnn = SAGE_h(self.hidden_dim, self.n_layers, self.hidden_dim, self.metadata)
        elif self.model_name == "GAT-h":
            self.gnn = GAT_h(self.hidden_dim, self.n_layers, self.hidden_dim, self.metadata)
        elif self.model_name == "HAN":
            self.gnn = HAN(-1, self.hidden_dim, self.n_layers, self.hidden_dim, self.metadata)
        elif self.model_name == "HGT":
            self.gnn = HGT(self.hidden_dim, self.n_layers, self.hidden_dim, self.metadata)
        else:
            raise ValueError(f'{self.model_name}')

    def forward(self, x, edge_index):
        '''Given a homogeneous graph data'''
        if SPAR:
            hdata = self._get_hdata_spar(x, edge_index, self.et)
        else:
            hdata = self._get_hdata(x, edge_index, self.et)
        
        if TIMER:
            torch.cuda.synchronize()
            start_time = time.perf_counter()

        out_dict = self.gnn(hdata.x_dict, hdata.edge_index_dict)
        # out = self.fc(sum([out_dict[tp] for tp in self.metadata[0]]))
        out = self.fc(torch.cat([out_dict[tp] for tp in self.metadata[0]], dim=1))
        if TIMER:
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            print(f"NSG forward time: {(end_time - start_time) * 1000} ms")
        return out


class NSG_MOE(BaseNSG):
    def __init__(
        self,
        *args,
        n_self_experts=2,
        n_cross_experts=2,
        k=2,
        noisy_gating=True,
        aux_coef=10**4,
    ):
        super().__init__(*args)
        self.n_self_experts = n_self_experts
        self.n_cross_experts = n_cross_experts
        self.k = k
        self.n_tot = n_tot = n_self_experts + n_cross_experts
        self.noisy_gating = noisy_gating
        self.aux_coef = aux_coef
        self.gate = nn.ParameterDict()
        self.noisy_gate = nn.ParameterDict()
        self.aux_loss = 0.0
        # self.register_buffer('tot_loss', torch.zeros(1))
        for nt in self.metadata[0]:
            self.gate[nt] = torch.zeros(self.in_channels[nt], n_tot)
            self.noisy_gate[nt] = torch.zeros(self.in_channels[nt], n_tot)

        if self.model_name == "SAGE-h":
            self.experts = nn.ModuleList([SAGE_h(self.hidden_dim, self.n_layers, self.hidden_dim, self.metadata) for _ in range(n_tot)])
        elif self.model_name == "GAT-h":
            self.experts = nn.ModuleList([GAT_h(self.hidden_dim, self.n_layers, self.hidden_dim, self.metadata) for _ in range(n_tot)])
        elif self.model_name == "HAN":
            self.experts = nn.ModuleList([HAN(-1, self.hidden_dim, self.n_layers, self.hidden_dim, self.metadata) for _ in range(n_tot)])
        elif self.model_name == "HGT":
            self.experts = nn.ModuleList([HGT(self.hidden_dim, self.n_layers, self.hidden_dim, self.metadata) for _ in range(n_tot)])
        else:
            raise ValueError(f'{self.model_name}')

    def _calc_importance_loss(self, gate_probs, eps=0.0):
        importance = gate_probs.sum(dim=0)
        importance_loss = torch.std(importance) / (torch.mean(importance) ** 2 + eps)
        return importance_loss

    def _calc_load_loss(self, ebds, noisy_ebds, gate_logits, eps=0.0):
        k = self.k
        n_tot = self.n_tot
        if k < n_tot:
            topk_values, _ = torch.topk(gate_logits, k + 1)
            is_in_topk = gate_logits >= topk_values[:, k - 1].unsqueeze(1)
            kth_exc = torch.where(
                is_in_topk,
                topk_values[:, k].unsqueeze(1).expand(-1, n_tot),
                topk_values[:, k - 1].unsqueeze(1).expand(-1, n_tot),
            )
            normal_dist = dist.Normal(loc=0, scale=1)
            load_vec = normal_dist.cdf((ebds - kth_exc) / F.softplus(noisy_ebds)).sum(0)
            load_loss = torch.std(load_vec) / (torch.mean(load_vec) ** 2 + eps)
        else:
            load_loss = 0.0
        return load_loss

    def forward(self, x, edge_index):
        '''Given a homogeneous graph data'''
        n_nodes = x.size(0)
        tp_list = self.metadata[0]
        if SPAR:
            hdata = self._get_hdata_spar(x, edge_index, et=1)
            hdata2 = self._get_hdata_spar(x, edge_index, et=2)
        else:
            hdata = self._get_hdata(x, edge_index, et=1)
            hdata2 = self._get_hdata(x, edge_index, et=2)

        if TIMER:
            torch.cuda.synchronize()
            start_time = time.perf_counter()
        out_list = [self.experts[i](hdata.x_dict, hdata.edge_index_dict) for i in range(self.n_self_experts)] + [self.experts[i](hdata2.x_dict, hdata2.edge_index_dict) for i in range(self.n_self_experts, self.n_tot)]
        # tp2list = {}
        # for tp in tp_list:
        #     tp2list[tp] = torch.stack([ebd_dict[tp] for ebd_dict in out_list], dim=1)
        '''gating'''
        out_dict = {}
        tot_loss = 0.0

        prob_dict = {}
        emb_list = {tp: [[] for _ in range(self.n_tot)] for tp in tp_list}
        for tp in tp_list:
            ebds = hdata[tp].x @ self.gate[tp]
            if self.noisy_gating:
                noisy_ebds = hdata[tp].x @ self.noisy_gate[tp]
                gate_logits = ebds + torch.randn(ebds.shape).to(ebds.device) * F.softplus(noisy_ebds)
                tot_loss += self._calc_load_loss(ebds, noisy_ebds, gate_logits)
            else:
                gate_logits = ebds
            # Get top-k experts for each input
            topk_probs, topk_indices = torch.topk(gate_logits, self.k, dim=-1)
            # topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
            gate_probs = torch.full_like(gate_logits, float('-inf'))
            gate_probs = gate_probs.scatter(1, topk_indices, topk_probs)
            gate_probs = F.softmax(gate_probs, dim=-1)
            expert_outputs = torch.stack([ebd_dict[tp] for ebd_dict in out_list], dim=1)
            output = (expert_outputs * gate_probs.unsqueeze(-1)).sum(dim=1)
            tot_loss += self._calc_importance_loss(gate_probs)

            if DEBUG:
                prob_dict[tp] = gate_probs.sum(dim=0).detach().cpu().numpy()
                # take some samples
                for i, j in gate_probs[:N_SAMPLES].nonzero():
                    emb_list[tp][j].append(hdata[tp].x[i].detach().cpu().numpy())

            # # Initialize output
            # output = torch.zeros(n_nodes, self.hidden_dim).to(x.device)

            # # Process each expert in the top-k
            # for i in range(self.k):
            #     # Create mask for expert i
            #     expert_mask = F.one_hot(topk_indices[:, i], self.n_tot).float()

            #     # Get expert outputs (all experts process the input)

            #     # Select outputs from the current expert and weight by gate probability
            #     selected_output = (expert_outputs * expert_mask.unsqueeze(-1)).sum(dim=1)
            #     weighted_output = selected_output * topk_probs[:, i].unsqueeze(-1)

            #     # Add to total output
            #     output += weighted_output
            out_dict[tp] = output
        if DEBUG:
            save_with_pickle(prob_dict, '../saved/debug_nsg_moe')
            save_with_pickle(emb_list, '../saved/debug_nsg_emb')

        self.aux_loss = tot_loss

        out = self.fc(torch.cat([out_dict[tp] for tp in tp_list], dim=1))
        if TIMER:
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            print(f"NSG_MOE forward time: {(end_time - start_time) * 1000} ms")
        return out
