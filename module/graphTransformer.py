import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, LayerNorm

class GraphTransformer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, heads=6, dropout=1e-5, bias=True):
        super(GraphTransformer, self).__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residuals = nn.ModuleList()

        first_channels = hidden_channels * heads
        self.layers.append(TransformerConv(in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout, bias=bias))
        self.norms.append(LayerNorm(first_channels, mode='node'))
        self.residuals.append(
            nn.Identity() if in_channels == first_channels
            else nn.Linear(in_channels, first_channels, bias=False)
        )

        for _ in range(1, num_layers - 1):
            self.layers.append(TransformerConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True, dropout=dropout, bias=bias))
            self.norms.append(LayerNorm(hidden_channels * heads, mode='node'))
            self.residuals.append(nn.Identity())

        self.layers.append(TransformerConv(hidden_channels * heads, out_channels, heads=heads, concat=False, dropout=dropout, bias=bias))
        self.norms.append(LayerNorm(out_channels, mode='node'))
        self.residuals.append(
            nn.Identity() if hidden_channels * heads == out_channels
            else nn.Linear(hidden_channels * heads, out_channels, bias=False)
        )

    def forward(self, x, edge_index):
        for layer, norm, residual in zip(self.layers, self.norms, self.residuals):
            x = layer(x, edge_index) + residual(x)
            x = torch.sigmoid(norm(x))

        return x
