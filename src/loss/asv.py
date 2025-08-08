import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    """
    Example of a loss function to use.
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **batch):
        return {"loss": self.loss(logits, labels)}


# class AngularSoftmaxLoss(nn.Module):
    def __init__(self, gamma=0.0, m=4, s=30.0):
        super().__init__()
        self.gamma = gamma
        self.margin = m
        self.scale = s
        self.easy_margin = True
        self.cos_m = torch.cos(torch.tensor(m))
        self.sin_m = torch.sin(torch.tensor(m))
        self.th = torch.cos(torch.pi - m)
        self.mm = torch.sin(torch.pi - m) * m

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **batch):
        cosine = F.normalize(logits, p=2, dim=1)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        cos_theta_m = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            final = torch.where(cosine > 0, cos_theta_m, cosine)
        else:
            final = torch.where(cosine > self.th, cos_theta_m, cosine - self.mm)

        logits = self.scale * final

        return {"loss": F.cross_entropy(logits, labels)}
