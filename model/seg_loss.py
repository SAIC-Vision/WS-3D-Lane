import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import numpy as np
import torch


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class BevSegLoss(nn.Module):
    """
    Compute the loss
    """
    def __init__(self):
        super(BevSegLoss, self).__init__()
        self.focal_loss = FocalLoss(reduce=False)

    def forward(self, bev_seg, bev_seg_label):
        bev_seg = bev_seg.softmax(dim=1)
        bev_seg = bev_seg.permute(0, 2, 3, 1)
        #print(bev_seg.shape)
        bev_seg = bev_seg.reshape(-1, 2)
        #bev_seg_label = bev_seg_label.to(torch.int64)
        #print(bev_seg_label)
        bev_seg_label = bev_seg_label.reshape(-1,)
        bev_seg_label = F.one_hot(bev_seg_label)
        #print(bev_seg_label.shape)
        bev_seg_label = bev_seg_label.reshape(-1, 2)
        bev_seg_label = bev_seg_label.to(torch.float32)
        loss = self.focal_loss(bev_seg, bev_seg_label)
        #print(loss)
        return torch.sum(loss)














