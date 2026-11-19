"""Masked attention pointer over completed device records for the wire arm."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class DevicePointer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.scale = math.sqrt(width)

    def forward(self, queries, devices, allowed, targets=None):
        """B×W×D queries, B×N×D devices, B×W×N legal endpoint mask.

        Callers must exclude nonexistent wire decisions, including padded ones.
        Empty candidate sets are compile failures, not a softmax over infinities.
        """
        if allowed.dtype != torch.bool:
            raise ValueError("pointer mask must be boolean")
        if not allowed.any(dim=-1).all():
            raise ValueError("wire has no compatible endpoint")
        logits = self.query(queries) @ self.key(devices).transpose(-1, -2) / self.scale
        if logits.shape != allowed.shape:
            raise ValueError("pointer mask shape does not match decisions")
        logits = logits.masked_fill(~allowed, -torch.inf)
        loss = None
        if targets is not None:
            if not allowed.gather(-1, targets.unsqueeze(-1)).all():
                raise ValueError("training target is an incompatible endpoint")
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss
