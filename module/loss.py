import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # Exp of the negative BCE loss
        F_loss = (1 - pt) ** self.gamma * BCE_loss

        if isinstance(self.alpha, (float, int)):
            alpha_t = torch.tensor([self.alpha, 1 - self.alpha], device=inputs.device)
        else:
            alpha_t = torch.tensor(self.alpha, device=inputs.device)
        alpha_t = alpha_t[targets.long()]
        F_loss = alpha_t * F_loss

        return F_loss.mean()



class KDLoss(nn.Module):
    def __init__(self, dweight=0.5, sweight=0.5):
        super(KDLoss, self).__init__()
        self.dweight = dweight
        self.sweight = sweight
        self.mse = nn.MSELoss()

    def forward(self, student_logits, teacher_logits, student_loss, is_teacher):
        if is_teacher:
            return student_loss
        else:
            distillation_loss = self.mse(student_logits, teacher_logits)
            total_loss = self.dweight * distillation_loss + self.sweight * student_loss
            return total_loss


class MultiLabelSoftMarginLoss(nn.Module):
    def __init__(self, reduction: str = "mean", pos_weight: torch.Tensor | None = None):
        super().__init__()
        if reduction not in ("none", "mean", "sum"):
            raise ValueError("reduction must be one of: 'none', 'mean', 'sum'")

        self.reduction = reduction
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

        if self.pos_weight is None:
            self.loss_fn = nn.MultiLabelSoftMarginLoss(reduction=reduction)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction=reduction, pos_weight=self.pos_weight)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        if inputs.shape != targets.shape:
            raise ValueError(f"inputs and targets must have same shape, got {inputs.shape} vs {targets.shape}")
        return self.loss_fn(inputs, targets)
