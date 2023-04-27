"""
Loss functions: revised on the base of Genlanenet:
 "Gen-laneNet: a generalized and scalable approach for 3D lane detection", Y.Guo, etal., arxiv 2020
"""

import numpy as np
import torch
import torch.nn as nn

class Laneline_loss_gflat_3D(nn.Module):
    """
    Compute the loss between predicted lanelines and ground-truth laneline in anchor representation.
    The anchor representation is in flat ground space X', Y' and real 3D Z. Visibility estimation is also included.
    The X' Y' and Z estimation will be transformed to real X, Y to compare with ground truth. An additional loss in
    X, Y space is expected to guide the learning of features to satisfy the geometry constraints between two spaces

    loss = loss0 + loss1 + loss2
    loss0: cross entropy loss for lane point visibility
    loss1: cross entropy loss for lane type classification
    loss2: sum of geometric distance betwen 3D lane anchor points in X and Z offsets
    """
    def __init__(self, config, args, anchor_x_steps, x_off_std, z_std, y_off_std):
        super(Laneline_loss_gflat_3D, self).__init__()
        self.batch_size = config.batch_size
        self.args = args
        if config.no_centerline:
            self.num_types = 1
        else:
            self.num_types = 3
        self.num_x_steps = anchor_x_steps.shape[0]
        self.num_y_steps = np.array(config.anchor_y_steps).shape[0]
        self.anchor_dim = 3*self.num_y_steps + 1

        # prepare broadcast anchor_x_tensor, anchor_y_tensor, std_X, std_Y, std_Z
        tmp_zeros = torch.zeros(self.batch_size, self.num_x_steps, self.num_types, self.num_y_steps)
        self.x_off_std = torch.tensor(x_off_std.astype(np.float32)).reshape(1, 1, 1, self.num_y_steps) + tmp_zeros
        self.y_off_std = torch.tensor(y_off_std.astype(np.float32)).reshape(1, 1, 1, self.num_y_steps) + tmp_zeros
        self.z_std = torch.tensor(z_std.astype(np.float32)).reshape(1, 1, 1, self.num_y_steps) + tmp_zeros
        self.anchor_y_steps = torch.tensor(np.array(config.anchor_y_steps).astype(np.float32)).reshape(1, 1, 1, self.num_y_steps) + tmp_zeros
        self.anchor_x_tensor = torch.tensor(anchor_x_steps.astype(np.float32)).reshape(1, self.num_x_steps, 1, 1) + tmp_zeros
        self.anchor_y_tensor = torch.tensor(np.array(config.anchor_y_steps).astype(np.float32)).reshape(1, 1, 1, self.num_y_steps) + tmp_zeros
        self.anchor_x_tensor = self.anchor_x_tensor/self.x_off_std
        self.anchor_y_tensor = self.anchor_y_tensor/self.y_off_std

        try:
            self.device = torch.device(f'cuda:{args.local_rank}')
        except:
            self.device = config.device


        self.z_std = self.z_std.to(self.device)
        self.anchor_x_tensor = self.anchor_x_tensor.to(self.device)
        self.anchor_y_tensor = self.anchor_y_tensor.to(self.device)
        self.x_off_std = self.x_off_std.to(self.device)
        self.eps = torch.tensor(1e-9).to(self.device)
        self.one = torch.tensor(1.0).to(self.device)
        self.anchor_y_steps = self.anchor_y_steps.to(self.device)


    def forward(self, pred_3D_lanes, gt_3D_lanes, gt_hcam):
        """

        :param pred_3D_lanes: predicted tensor with size N x (ipm_w/8) x 3*(2*K+1)
        :param gt_3D_lanes: ground-truth tensor with size N x (ipm_w/8) x 3*(2*K+1)
        :param gt_hcam: ground-truth camera height with size N
        :return:
        """
        sizes = pred_3D_lanes.shape
        # reshape to N x ipm_w/8 x 3 x (3K+1)
        pred_3D_lanes = pred_3D_lanes.reshape(sizes[0], sizes[1], self.num_types, self.anchor_dim)
        gt_3D_lanes = gt_3D_lanes.reshape(sizes[0], sizes[1], self.num_types, self.anchor_dim)
        # class prob N x ipm_w/8 x 3 x 1, anchor values N x ipm_w/8 x 3 x 2K, visibility N x ipm_w/8 x 3 x K
        pred_class = pred_3D_lanes[:, :, :, -1].unsqueeze(-1)
        pred_anchors = pred_3D_lanes[:, :, :, :2*self.num_y_steps]
        pred_visibility = pred_3D_lanes[:, :, :, 2*self.num_y_steps:3*self.num_y_steps]
        gt_class = gt_3D_lanes[:, :, :, -1].unsqueeze(-1)
        gt_anchors = gt_3D_lanes[:, :, :, :2*self.num_y_steps]
        gt_visibility = gt_3D_lanes[:, :, :, 2*self.num_y_steps:3*self.num_y_steps]

        # cross-entropy loss for visibility
        loss0 = -torch.sum(
            gt_visibility*torch.log(pred_visibility + torch.tensor(1e-9)) +
            (torch.ones_like(gt_visibility) - gt_visibility + torch.tensor(1e-9)) *
            torch.log(torch.ones_like(pred_visibility) - pred_visibility + torch.tensor(1e-9)))/self.num_y_steps
        # cross-entropy loss for lane probability
        loss1 = -torch.sum(
            gt_class*torch.log(pred_class + torch.tensor(1e-9)) +
            (torch.ones_like(gt_class) - gt_class) *
            torch.log(torch.ones_like(pred_class) - pred_class + torch.tensor(1e-9)))
        # applying L1 norm does not need to separate X and Z
        loss2 = torch.sum(
            torch.norm(gt_class*torch.cat((gt_visibility, gt_visibility), 3)*(pred_anchors-gt_anchors), p=1, dim=3))

        # compute loss in real 3D X, Y space, the transformation considers offset to anchor and normalization by std
        pred_Xoff_g = pred_anchors[:, :, :, :self.num_y_steps]
        pred_Z = pred_anchors[:, :, :, self.num_y_steps:2*self.num_y_steps]
        gt_Xoff_g = gt_anchors[:, :, :, :self.num_y_steps]
        gt_Z = gt_anchors[:, :, :, self.num_y_steps:2*self.num_y_steps]
        gt_hcam = gt_hcam.reshape(gt_hcam.shape[0], 1, 1, 1)
        pred_Xoff = (1 - pred_Z * self.z_std / gt_hcam) * pred_Xoff_g - pred_Z * self.z_std / gt_hcam * self.anchor_x_tensor
        pred_Yoff = -pred_Z * self.z_std / gt_hcam * self.anchor_y_tensor
        gt_Xoff = (1 - gt_Z * self.z_std / gt_hcam) * gt_Xoff_g - gt_Z * self.z_std / gt_hcam * self.anchor_x_tensor
        gt_Yoff = -gt_Z * self.z_std / gt_hcam * self.anchor_y_tensor
        loss3 = torch.sum(
            torch.norm(
                gt_class * torch.cat((gt_visibility, gt_visibility), 3) *
                (torch.cat((pred_Xoff, pred_Yoff), 3) - torch.cat((gt_Xoff, gt_Yoff), 3)), p=1, dim=3))

        return loss0+loss1+loss2+loss3


