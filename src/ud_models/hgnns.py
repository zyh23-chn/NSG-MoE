from typing import Dict, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HANConv, HGTConv, GraphNorm, Linear, GATConv, SAGEConv, GPSConv
from torch_geometric.nn.attention import PerformerAttention
from torch_geometric.utils import to_dense_batch
from copy import deepcopy


def tup2str(o):
    return '__'.join(o)


class HAN(nn.Module):
    def __init__(self, in_channels: Union[int, Dict[str, int]], hidden_channels, num_layers, out_channels, metadata, heads=1, dropout=0.0):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.convs = nn.ModuleList([HANConv(hidden_channels, hidden_channels, metadata, heads=heads) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.metadata = metadata
        node_types = metadata[0]
        self.norm_dict = nn.ModuleDict()
        self.lin_dict = nn.ModuleDict()
        self.lin_dict2 = nn.ModuleDict()
        for node_type in node_types:
            self.norm_dict[node_type] = GraphNorm(hidden_channels)
            self.lin_dict[node_type] = Linear(in_channels, hidden_channels)
            self.lin_dict2[node_type] = Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict):
        out = x_dict
        for node_type in x_dict:
            out[node_type] = self.lin_dict[node_type](x_dict[node_type]).relu_()
        x0 = {k: v.clone() for k, v in out.items()}
        for conv in self.convs:
            # out0 = {k: v.clone() for k, v in out.items()}
            out = conv(out, edge_index_dict)
            # for edge_type in edge_index_dict:
            #     '''residual'''
            #     src_type, _, dst_type = edge_type
            #     out[dst_type] += out0[dst_type]
            for node_type in x_dict:
                out[node_type] = F.gelu(out[node_type])
                # out[node_type] = self.dropout(out[node_type])
        for node_type in x_dict:
            out[node_type] = out[node_type] + x0[node_type]
            out[node_type] = self.norm_dict[node_type](out[node_type])
            out[node_type] = self.lin_dict2[node_type](out[node_type])
            
        return out


class HGT(nn.Module):
    def __init__(self, hidden_channels, num_layers, out_channels, metadata, num_heads=1):
        super().__init__()
        self.lin_dict = nn.ModuleDict()
        node_types = metadata[0]
        for node_type in node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata, num_heads)
            self.convs.append(conv)

        self.norm_dict = nn.ModuleDict()
        self.lin_dict2 = nn.ModuleDict()
        for node_type in node_types:
            self.norm_dict[node_type] = GraphNorm(hidden_channels)
            self.lin_dict2[node_type] = Linear(hidden_channels, out_channels)
        # self.norm_dict = {node_type: GraphNorm(hidden_channels) for node_type in node_types}
        # self.lin_dict2 = {node_type: Linear(hidden_channels, out_channels) for node_type in node_types}

    def forward(self, x_dict, edge_index_dict):
        out = {node_type: self.lin_dict[node_type](x).relu_() for node_type, x in x_dict.items()}

        x0 = {k: v.clone() for k, v in out.items()}
        for conv in self.convs:
            out = conv(out, edge_index_dict)
            for node_type in x_dict:
                out[node_type] = F.gelu(out[node_type])
                # out[node_type] = self.dropout(out[node_type])
        for node_type in x_dict:
            out[node_type] = out[node_type] + x0[node_type]
            out[node_type] = self.norm_dict[node_type](out[node_type])
            out[node_type] = self.lin_dict2[node_type](out[node_type])
        return out

        for conv in self.convs:
            out = conv(out, edge_index_dict)
        for node_type in out:
            out[node_type] = self.norm_dict[node_type](out[node_type])
            out[node_type] = self.lin_dict2[node_type](out[node_type])
        return out


class SAGE_h(nn.Module):
    def __init__(self, hidden_channels, num_layers, out_channels, metadata, dropout=0.5):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        node_types, edge_types = metadata
        self.lin_dict = nn.ModuleDict()
        for node_type in node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = nn.ModuleDict()
            for edge_type in edge_types:
                conv_dict[tup2str(edge_type)] = SAGEConv(-1, hidden_channels)
            self.convs.append(conv_dict)

        self.norm_dict = nn.ModuleDict()
        self.lin_dict2 = nn.ModuleDict()
        for node_type in node_types:
            self.norm_dict[node_type] = GraphNorm(hidden_channels)
            self.lin_dict2[node_type] = Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict):
        # out_dict = {node_type: torch.zeros((x.size(0), self.hidden_channels)).to(x.device) for node_type, x in x_dict.items()}
        out_dict = {node_type: self.lin_dict[node_type](x).relu_() for node_type, x in x_dict.items()}
        x0 = {k: v.clone() for k, v in out_dict.items()}

        for conv in self.convs:
            for edge_type, edge_index in edge_index_dict.items():
                src_type, _, dst_type = edge_type
                sage_conv = conv[tup2str(edge_type)]
                src_x = out_dict[src_type]
                dst_x = out_dict[dst_type]
                out = sage_conv((src_x, dst_x), edge_index)
                # out = self.dropout(F.gelu(out))
                # out += dst_x
                out = F.gelu(out)
                out_dict[dst_type] = out

        for node_type, out in out_dict.items():
            out_dict[node_type] = out_dict[node_type] + x0[node_type]
            out_dict[node_type] = self.norm_dict[node_type](out_dict[node_type])
            out_dict[node_type] = self.lin_dict2[node_type](out_dict[node_type])
        return out_dict


class GAT_h(nn.Module):
    def __init__(self, hidden_channels, num_layers, out_channels, metadata, num_heads=1, dropout=0):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        node_types, edge_types = metadata
        self.lin_dict = nn.ModuleDict()
        for node_type in node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels * num_heads)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = nn.ModuleDict()
            for edge_type in edge_types:
                conv_dict[tup2str(edge_type)] = GATConv(-1, hidden_channels, num_heads, dropout=dropout)
            self.convs.append(conv_dict)
        # self.gat_convs = nn.ModuleDict()
        # for edge_type in edge_types:
        #     self.gat_convs[tup2str(edge_type)] = GATConv(-1, hidden_channels , num_heads, dropout=0.6)

        self.norm_dict = nn.ModuleDict()
        self.lin_dict2 = nn.ModuleDict()
        for node_type in node_types:
            self.norm_dict[node_type] = GraphNorm(hidden_channels * num_heads)
            self.lin_dict2[node_type] = Linear(hidden_channels * num_heads, out_channels)
        # self.norm_dict = {node_type: GraphNorm(hidden_channels).to(device) for node_type in node_types}
        # self.lin_dict2 = {node_type: Linear(hidden_channels, out_channels).to(device) for node_type in node_types}

    def forward(self, x_dict, edge_index_dict):
        out_dict = {node_type: self.lin_dict[node_type](x).relu_() for node_type, x in x_dict.items()}
        x0 = {k: v.clone() for k, v in out_dict.items()}

        for conv in self.convs:
            for edge_type, edge_index in edge_index_dict.items():
                src_type, _, dst_type = edge_type
                gat_conv = conv[tup2str(edge_type)]
                src_x = out_dict[src_type]
                dst_x = out_dict[dst_type]
                out = gat_conv((src_x, dst_x), edge_index)
                # out = self.dropout(F.gelu(out))
                # out += dst_x
                out = F.gelu(out)
                out_dict[dst_type] = out

        for node_type, out in out_dict.items():
            out_dict[node_type] = out_dict[node_type] + x0[node_type]
            out_dict[node_type] = self.norm_dict[node_type](out_dict[node_type])
            out_dict[node_type] = self.lin_dict2[node_type](out_dict[node_type])
        return out_dict

        out_dict = {node_type: torch.zeros((x.size(0), self.hidden_channels)).to(x.device) for node_type, x in x_dict.items()}
        emb_dict = {node_type: self.lin_dict[node_type](x).relu_() for node_type, x in x_dict.items()}

        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            gat_conv = self.gat_convs[tup2str(edge_type)]
            src_x = emb_dict[src_type]
            dst_x = emb_dict[dst_type]
            out = gat_conv((src_x, dst_x), edge_index)
            out_dict[dst_type] += out

        for node_type, out in out_dict.items():
            out_dict[node_type] = self.norm_dict[node_type](out_dict[node_type])
            out_dict[node_type] = self.lin_dict2[node_type](out_dict[node_type])
        return out_dict


class GT_HAN(HAN):
    '''graph transformer'''

    def __init__(self, attn_type, attn_heads, *args, attn_kwargs={}, **kwargs):
        super().__init__(*args, **kwargs)
        self.f_lin_dict = nn.ModuleDict()  # transform into the same dimensionality
        node_types = self.metadata[0]
        for node_type in node_types:
            self.f_lin_dict[node_type] = nn.LazyLinear(self.hidden_channels)
        channels = self.hidden_channels
        self.attn_type = attn_type
        if attn_type == 'multihead':
            self.attn = torch.nn.MultiheadAttention(
                channels,
                attn_heads,
                batch_first=True,
                **attn_kwargs,
            )
        elif attn_type == 'performer':
            self.attn = PerformerAttention(
                channels=channels,
                heads=attn_heads,
                **attn_kwargs,
            )
        else:
            # TODO: Support BigBird
            raise ValueError(f'{attn_type} is not supported')

    def forward(self, x_dict, edge_index_dict):
        out = {}
        for nt, x in x_dict.items():
            out[nt] = self.f_lin_dict[nt](x)
        node_types = self.metadata[0]

        for conv in self.convs:
            mpnn_out = conv(out, edge_index_dict)
            '''attention'''
            # Global attention transformer-style model.
            h0 = torch.cat([out[nt] for nt in node_types], dim=0)
            h = h0.clone()

            if isinstance(self.attn, torch.nn.MultiheadAttention):
                h, _ = self.attn(h, h, h, need_weights=False)
            elif isinstance(self.attn, PerformerAttention):
                h = self.attn(h)

            # h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + h0  # Residual connection.
            cur_pos = 0
            for nt in node_types:
                cnt = out[nt].size(0)
                mpnn_out[nt] += h[cur_pos : cur_pos + cnt, :]
                cur_pos += cnt
            # if self.norm2 is not None:
            #     if self.norm_with_batch:
            #         h = self.norm2(h, batch=batch)
            #     else:
            #         h = self.norm2(h)
            out = mpnn_out
        for node_type in x_dict:
            if out[node_type] is not None:
                out[node_type] = self.norm_dict[node_type](out[node_type])
                out[node_type] = self.lin_dict[node_type](out[node_type])
        return out
