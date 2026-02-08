import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, LayerNorm

class GraphTransformer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, heads=6, dropout=1e-5, bias=True):
        super(GraphTransformer, self).__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.layers.append(TransformerConv(in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout, bias=bias))
        self.norms.append(LayerNorm(hidden_channels * heads))

        for _ in range(1, num_layers - 1):
            self.layers.append(TransformerConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True, dropout=dropout, bias=bias))
            self.norms.append(LayerNorm(hidden_channels * heads))

        self.layers.append(TransformerConv(hidden_channels * heads, out_channels, heads=heads, concat=False, dropout=dropout, bias=bias))
        self.norms.append(LayerNorm(out_channels))

    def forward(self, x, edge_index):
        for layer, norm in zip(self.layers, self.norms):
            x = layer(x, edge_index)
            x = norm(x)
            x = F.relu(x, inplace=True)

        return x
