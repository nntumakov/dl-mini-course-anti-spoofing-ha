import torch
import numpy as np
from .base_metric import BaseMetric

from src.metrics.calculate_eer import compute_eer

class EERMetric(BaseMetric):
    """
    Calculates the Equal Error Rate.
    """
    def __init__(self, positive_label=1, *args, **kwargs):
        """
        Args:
            positive_label (int): The index of the class that is considered "positive"
                                  (e.g., bonafide). It should correspond to the
                                  column in the logits.
        """
        super().__init__(*args, **kwargs)
        self.positive_label = positive_label

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs) -> float:
        """
        Args:
            logits (torch.Tensor): A tensor of shape (batch, num_classes).
            labels (torch.Tensor): A tensor of shape (batch,).
        
        Returns:
            eer (float): The Equal Error Rate in percent.
        """
        scores = logits[:, self.positive_label].detach().cpu().numpy()
        
        labels = labels.detach().cpu().numpy()
        bonafide_scores = scores[labels == self.positive_label]
        other_scores = scores[labels != self.positive_label]

        if len(bonafide_scores) == 0 or len(other_scores) == 0:
            return 0.0

        eer, _ = compute_eer(bonafide_scores, other_scores)
        return eer * 100