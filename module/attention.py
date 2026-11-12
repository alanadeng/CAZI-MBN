import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hid_units, nb_graphs, nb_nodes):
        super(Attention, self).__init__()
        self.hid_units = hid_units
        self.nb_graphs = nb_graphs
        self.nb_nodes = nb_nodes
        self.A = nn.ModuleList([nn.Linear(hid_units, 1) for _ in range(nb_graphs)])
        self.weight_init()

    def weight_init(self):
        for attn in self.A:
            nn.init.xavier_normal_(attn.weight)
            attn.bias.data.fill_(0.0)

    def forward(self, feat_pos, feat_neg, summary):
        feat_pos, feat_pos_attn = self.attn_feature(feat_pos)
        feat_neg, feat_neg_attn = self.attn_feature(feat_neg)
        summary, summary_attn = self.attn_summary(summary)

        return feat_pos, feat_neg, summary

    def aggregate(self, features, mask=None):
        if not isinstance(features, torch.Tensor):
            features = torch.stack(features, dim=0)
        if features.dim() != 3:
            raise ValueError(f"features must have shape (layers, nodes, channels), got {features.shape}.")
        if features.size(0) != self.nb_graphs or features.size(2) != self.hid_units:
            raise ValueError(
                f"Expected {self.nb_graphs} layers with {self.hid_units} channels, "
                f"got {features.size(0)} layers with {features.size(2)} channels."
            )
        if features.size(1) != self.nb_nodes:
            raise ValueError(
                f"Expected {self.nb_nodes} nodes per layer, got {features.size(1)}."
            )

        scores = torch.cat([
            self.A[index](features[index])
            for index in range(self.nb_graphs)
        ], dim=1)
        if mask is None:
            weights = F.softmax(scores, dim=1)
        else:
            if mask.shape != scores.shape:
                raise ValueError(
                    f"mask must have shape {tuple(scores.shape)}, got {tuple(mask.shape)}."
                )
            mask = mask.to(device=scores.device, dtype=torch.bool)
            has_layer = mask.any(dim=1, keepdim=True)
            scores = scores.masked_fill(~mask, float('-inf'))
            scores = torch.where(has_layer, scores, torch.zeros_like(scores))
            weights = F.softmax(scores, dim=1) * has_layer
        aggregated = (
            features * weights.transpose(0, 1).unsqueeze(-1)
        ).sum(dim=0)
        return aggregated, weights

    def attn_feature(self, features):
        if not isinstance(features, torch.Tensor):
            features = torch.stack(features, dim=0)
        if features.dim() == 4 and features.size(1) == 1:
            features = features.squeeze(1)
        aggregated, weights = self.aggregate(features)
        return aggregated.unsqueeze(0), weights

    def attn_summary(self, features):
        if not isinstance(features, torch.Tensor):
            features = torch.stack(features, dim=0)
        if features.dim() == 4 and features.size(1) == 1:
            features = features.squeeze(1)
        if features.dim() != 3 or features.size(0) != self.nb_graphs:
            raise ValueError(
                f"features must have shape ({self.nb_graphs}, nodes, channels), got {features.shape}."
            )

        layer_summaries = features.mean(dim=1)
        scores = torch.cat([
            self.A[index](layer_summaries[index])
            for index in range(self.nb_graphs)
        ], dim=0)
        weights = F.softmax(scores, dim=0).unsqueeze(-1)
        summary = (layer_summaries * weights).sum(dim=0, keepdim=True)
        return summary, weights
