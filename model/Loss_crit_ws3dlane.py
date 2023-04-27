"""
Loss functions for ws3dlane on the base of Genlanenet:
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

    loss = loss0 + loss1 + loss2 + loss4
    loss0: cross entropy loss for lane point visibility
    loss1: cross entropy loss for lane type classification
    loss2: sum of geometric distance betwen 3D lane anchor points in X offsets
    loss4: error in lane_width
    """
    def __init__(self, config, args, anchor_x_steps, x_off_std, z_std):
        super(Laneline_loss_gflat_3D, self).__init__()
        self.batch_size = config.batch_size
        self.args = args
        if config.no_centerline:
            self.num_types = 1
        else:
            self.num_types = 3
        anchor_x_steps = np.array(anchor_x_steps)
        x_off_std = np.array(x_off_std)
        z_std = np.array(z_std)
        self.num_x_steps = anchor_x_steps.shape[0]
        self.num_y_steps = np.array(config.anchor_y_steps).shape[0]
        self.anchor_dim = 3*self.num_y_steps + 1

        # prepare broadcast anchor_x_tensor, anchor_y_tensor, std_X, std_Y, std_Z
        tmp_zeros = torch.zeros(self.batch_size, self.num_x_steps, self.num_types, self.num_y_steps)
        self.x_off_std = torch.tensor(x_off_std.astype(np.float32)).reshape(1, 1, 1, self.num_y_steps) + tmp_zeros
        self.z_std = torch.tensor(z_std.astype(np.float32)).reshape(1, 1, 1, self.num_y_steps) + tmp_zeros
        self.anchor_y_steps = torch.tensor(np.array(config.anchor_y_steps).astype(np.float32)).reshape(1, 1, 1, self.num_y_steps) + tmp_zeros
        self.anchor_x_tensor = torch.tensor(anchor_x_steps.astype(np.float32)).reshape(1, self.num_x_steps, 1, 1) + tmp_zeros
        try:
            self.device = torch.device(f'cuda:{args.local_rank}')
        except:
            self.device = config.device

        self.z_std = self.z_std.to(self.device)
        self.anchor_x_tensor = self.anchor_x_tensor.to(self.device)
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
        # applying L1 norm for x_offset
        pred_Xoff_g = pred_anchors[:, :, :, :self.num_y_steps]
        pred_Z = pred_anchors[:, :, :, self.num_y_steps:2 * self.num_y_steps]
        gt_Xoff_g = gt_anchors[:, :, :, :self.num_y_steps]
        gt_hcam = gt_hcam.reshape(self.batch_size, 1, 1, 1)
        loss2 = torch.sum(
            torch.norm(gt_class * gt_visibility * (pred_Xoff_g - gt_Xoff_g), p=1,
                       dim=3))

        # compute loss in lane_width, the transformation considers offset to anchor and normalization by std
        pred_Z = pred_Z * self.z_std  # N x ipm_w/8 x K
        gt_X_g = gt_Xoff_g * self.x_off_std + self.anchor_x_tensor
        pred_X_gt_Xg = (1 - pred_Z / gt_hcam) * gt_X_g
        pred_Y = (1 - pred_Z / gt_hcam) * self.anchor_y_steps

        loss_width = []
        for i in range(pred_X_gt_Xg.shape[0]):
            if torch.sum(gt_class[i]) == 0:
                continue
            lane_i = torch.cat([pred_X_gt_Xg[i, j, :, :] for j in range(pred_X_gt_Xg.shape[1]) if gt_class[i, j, 0, 0] > 0], dim=0)  # num_lane x k
            lane0_i = torch.cat([gt_X_g[i, j, :, 0] for j in range(gt_X_g.shape[1]) if gt_class[i, j, 0, 0] > 0],
                               dim=0)
            pred_Y_i  = torch.cat([pred_Y[i, j, :, :] for j in range(pred_Y.shape[1]) if gt_class[i, j, 0, 0] > 0], dim=0)
            gt_visibility_i = torch.cat(
                [gt_visibility[i, j, :, :] for j in range(pred_X_gt_Xg.shape[1]) if gt_class[i, j, 0, 0] > 0], dim=0)
            lane_i_vis = torch.cat([lane_i[:, j].unsqueeze(1) for j in range(lane_i.shape[1]) if
                                    (torch.sum(gt_visibility_i[:, j]) >= (lane_i.shape[0]) or j < 5)], dim=1)
            pred_Y_i_vis = torch.cat([pred_Y_i[:, j].unsqueeze(1) for j in range(lane_i.shape[1]) if
                                    (torch.sum(gt_visibility_i[:, j]) >= (lane_i.shape[0]) or j < 5)], dim=1)
            pred_lY_i_vis = torch.cat([torch.abs(pred_Y_i_vis[:, j].unsqueeze(1) - pred_Y_i_vis[:, j-1].unsqueeze(1)) for j in range(1, pred_Y_i_vis.shape[1])], dim=1)
            pred_lY_i_vis = torch.cat([pred_lY_i_vis[:, 0].unsqueeze(1), pred_lY_i_vis], dim=1)
            pred_XY_i_vis = torch.cat([lane_i_vis.unsqueeze(2), pred_Y_i_vis.unsqueeze(2)], dim=2)
            pred_l_i_vis = torch.cat([torch.norm(pred_XY_i_vis[:, j, :] - pred_XY_i_vis[:, j-1, :], dim=1).unsqueeze(1) for j in range(1, pred_XY_i_vis.shape[1])], dim=1)
            pred_l_i_vis = torch.cat([pred_lY_i_vis[:, 0].unsqueeze(1), pred_l_i_vis], dim=1)
            pred_lY_i_vis = pred_lY_i_vis.detach()
            pred_l_i_vis = pred_l_i_vis.detach()
            if lane_i_vis.shape[0] < 2:
                continue
            width_i = torch.cat([abs(lane_i_vis[j, :].unsqueeze(0) - lane_i_vis[j - 1, :].unsqueeze(0)) *
                                 pred_lY_i_vis[j-1, :].unsqueeze(0) / (pred_l_i_vis[j-1, :].unsqueeze(0) + self.eps)
                                 for j in range(1, lane_i_vis.shape[0])], dim=0)
            width0_i = torch.cat([abs(lane0_i[j].unsqueeze(0) - lane0_i[j - 1].unsqueeze(0)) for j in
                                 range(1, lane0_i.shape[0])],
                                dim=0)
            width0_i = width0_i.unsqueeze(1)
            ori_width_i = torch.cat([width0_i, width_i], dim=1)
            width_error_i = torch.cat(
                [torch.abs(ori_width_i[:, j].unsqueeze(1) - ori_width_i[:, j - 1].unsqueeze(1)) for j in
                 range(1, ori_width_i.shape[1])], dim=1)
            loss_width_i = torch.sum(width_error_i)
            loss_width.append(loss_width_i)

        if len(loss_width) == 0:
            loss3 = 0
        else:
            loss3 = torch.sum(torch.stack(loss_width, dim=0))

        loss_height = []
        for i in range(pred_Z.shape[0]):
            if torch.sum(gt_class[i]) == 0:
                continue
            pred_Z_i = torch.cat([pred_Z[i, j, :, :] for j in range(pred_Z.shape[1]) if gt_class[i, j, 0, 0] > 0],
                                dim=0)
            gt_visibility_i = torch.cat(
                [gt_visibility[i, j, :, :] for j in range(pred_Z.shape[1]) if gt_class[i, j, 0, 0] > 0], dim=0)
            pred_Z_i_vis = torch.cat([pred_Z_i[:, j].unsqueeze(1) for j in range(pred_Z_i.shape[1]) if
                                    (torch.sum(gt_visibility_i[:, j]) >= (pred_Z_i.shape[0]) or j < 5)], dim=1)

            if pred_Z_i_vis.shape[0] < 2:
                continue

            loss_height_i = torch.cat([abs(pred_Z_i_vis[j, :].unsqueeze(0) - pred_Z_i_vis[j - 1, :].unsqueeze(0)) for j in
                                 range(1, pred_Z_i_vis.shape[0])],
                                dim=0)
            loss_height.append(torch.sum(loss_height_i))


        if len(loss_height) == 0:
            loss4 = 0
        else:
            loss4 = torch.sum(torch.stack(loss_height, dim=0))


        return loss0+loss1+loss2 + 5*loss3 +5*loss4
