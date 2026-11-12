import torch
import torch.nn as nn
from torch_geometric.nn import SAGPooling, global_mean_pool
from torch_geometric.utils import subgraph

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq,0)


class SAGPoolReadout(nn.Module):
    def __init__(self, in_channels, ratio=0.5):
        super(SAGPoolReadout, self).__init__()
        self.sagpool = SAGPooling(in_channels, ratio)

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x, edge_index, _, batch, _, _ = self.sagpool(x, edge_index, batch=batch)
        x_pooled = global_mean_pool(x, batch)

        return x_pooled



class SAGPoolReadoutEdge(nn.Module):
    def __init__(self, in_channels, ratio=0.5):
        super(SAGPoolReadoutEdge, self).__init__()
        self.sagpool = SAGPooling(in_channels, ratio)

    def forward(self, x, edge_index, batch=None):
        edge_nodes = torch.unique(edge_index)
        edge_index, _ = subgraph(edge_nodes, edge_index, relabel_nodes=True, num_nodes=x.size(0))
        x = x[edge_nodes]
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        else:
            batch = batch[edge_nodes]

        x_full, edge_index_full = x, edge_index
        x, edge_index, _, batch, _, _ = self.sagpool(x, edge_index, batch=batch)
        if edge_index.size(1) == 0:
            edge_attr = torch.cat((x_full[edge_index_full[0]], x_full[edge_index_full[1]]), dim=1)
            return edge_attr.mean(dim=0, keepdim=True)
        edge_attr = torch.cat((x[edge_index[0, :], :], x[edge_index[1, :], :]), dim=1)
        edge_batch = batch[edge_index[0]]          
        x_pooled = global_mean_pool(edge_attr, edge_batch)
        #print(edge_attr)

        return x_pooled

