"""
Focal Loss implementation for handling class imbalance.
Focal Loss focuses learning on hard examples and down-weights easy examples.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Reduces the relative loss for well-classified examples (p_t > 0.5),
    putting more focus on hard, misclassified examples.
    
    Paper: https://arxiv.org/abs/1708.02002
    """
    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = 'mean',
        label_smoothing: float = 0.0
    ):
        """
        Args:
            alpha: Weighting factor for rare class (can be float or tensor of size n_classes)
            gamma: Focusing parameter (gamma=0 is equivalent to CE loss)
            reduction: 'mean', 'sum', or 'none'
            label_smoothing: Label smoothing factor (0.0 = no smoothing)
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits of shape (N, C) where C is number of classes
            targets: Ground truth labels of shape (N,)
            
        Returns:
            Focal loss value
        """
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            num_classes = inputs.size(1)
            smooth_targets = torch.zeros_like(inputs)
            smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            targets_one_hot = smooth_targets
        else:
            targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        
        # Compute p_t (probability of true class)
        p_t = torch.sum(targets_one_hot * F.softmax(inputs, dim=1), dim=1)
        
        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting if provided
        if isinstance(self.alpha, (float, int)):
            alpha_t = self.alpha
        else:
            # Alpha is a tensor of class weights
            alpha_t = torch.gather(self.alpha.expand(inputs.size(0), -1), 1, targets.unsqueeze(1)).squeeze(1)
        
        # Compute focal loss
        focal_loss = alpha_t * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

