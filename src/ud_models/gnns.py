import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GraphNorm


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, dropout=0.5):
        super(GraphSAGE, self).__init__()
        self.lin = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList([SAGEConv(hidden_channels, hidden_channels) for _ in range(num_layers)])
        self.norm = GraphNorm(hidden_channels)
        self.dropout = nn.Dropout(dropout)
        self.lin_out = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.lin(x)
        # x0 = x
        x0 = x.clone()
        for conv in self.convs:
            x = conv(x, edge_index)
            # x = F.gelu(x)
            # x = self.dropout(x)
        x = x + x0
        x = self.norm(x)
        x = self.lin_out(x)
        return x


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, dropout=0):
        super(GCN, self).__init__()
        self.lin = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList([GCNConv(hidden_channels, hidden_channels) for _ in range(num_layers)])
        self.norm = GraphNorm(hidden_channels)
        self.dropout = nn.Dropout(dropout)
        self.lin_out = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.lin(x)
        x0 = x.clone()
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.gelu(x)
            x = self.dropout(x)
        x = x + x0
        x = self.norm(x)
        x = self.lin_out(x)
        return x


class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, heads=8, dropout=0.6):
        super(GAT, self).__init__()
        self.lin = nn.Linear(in_channels, hidden_channels * heads)
        self.convs = nn.ModuleList([GATConv(hidden_channels * heads, hidden_channels, heads) for _ in range(num_layers)])
        self.norm = GraphNorm(hidden_channels * heads)
        self.dropout = nn.Dropout(dropout)
        self.lin_out = nn.Linear(hidden_channels * heads, out_channels)

    def forward(self, x, edge_index):
        x = self.lin(x)
        x0 = x.clone()
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.gelu(x)
            x = self.dropout(x)
        x = x + x0
        x = self.norm(x)
        x = self.lin_out(x)
        return x
