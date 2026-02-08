import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEClassifier(nn.Module):
    def __init__(self, fused_dim: int, output_dim: int, num_experts: int = 4, hidden_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.num_experts = num_experts

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fused_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
            for _ in range(num_experts)
        ])

        self.gating_network = nn.Linear(fused_dim, num_experts)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, fused_dim)
        expert_outputs = torch.stack([expert(h) for expert in self.experts], dim=1)   # (B, E, out)
        gate_weights = F.softmax(self.gating_network(h), dim=1).unsqueeze(-1)        # (B, E, 1)
        return (expert_outputs * gate_weights).sum(dim=1)                            # (B, out)


class MoE(nn.Module):
    def __init__(self, in1_dim: int, in2_dim: int, out_dim: int, num_experts: int = 4, hidden_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.in1_dim = int(in1_dim)
        self.in2_dim = int(in2_dim)
        fused_dim = self.in1_dim + self.in2_dim

        self.classifier = MoEClassifier(
            fused_dim=fused_dim,
            output_dim=int(out_dim),
            num_experts=int(num_experts),
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if x1.dim() != 2 or x2.dim() != 2:
            raise ValueError(f"x1/x2 must be 2D (B,C). Got {x1.shape} and {x2.shape}")
        if x1.size(1) != self.in1_dim:
            raise ValueError(f"x1 feature dim mismatch: expected {self.in1_dim}, got {x1.size(1)}")
        if x2.size(1) != self.in2_dim:
            raise ValueError(f"x2 feature dim mismatch: expected {self.in2_dim}, got {x2.size(1)}")

        h = torch.cat([x1, x2], dim=1)  # (B, in1+in2)
        return self.classifier(h)
