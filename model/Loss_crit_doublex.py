"""
Loss functions

Author: Yuliang Guo (33yuliangguo@gmail.com)
Date: March, 2020
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy



class Laneline_loss_3D(nn.Module):
    """
    Compute the loss between predicted lanelines and ground-truth laneline in anchor representation.
    The anchor representation is based on real 3D X, Y, Z.

    loss = loss1 + loss2 + loss2
    loss1: cross entropy loss for lane type classification
    loss2: sum of geometric distance betwen 3D lane anchor points in X and Z offsets
    loss3: error in estimating pitch and camera heights
    """
    def __init__(self, num_types, anchor_dim, pred_cam):
        super(Laneline_loss_3D, self).__init__()
        self.num_types = num_types
        self.anchor_dim = anchor_dim
        self.pred_cam = pred_cam

    def forward(self, pred_3D_lanes, gt_3D_lanes, pred_hcam, gt_hcam, pred_pitch, gt_pitch):
        """

        :param pred_3D_lanes: predicted tensor with size N x (ipm_w/8) x 3*(2*K+1)
        :param gt_3D_lanes: ground-truth tensor with size N x (ipm_w/8) x 3*(2*K+1)
        :param pred_pitch: predicted pitch with size N
        :param gt_pitch: ground-truth pitch with size N
        :param pred_hcam: predicted camera height with size N
        :param gt_hcam: ground-truth camera height with size N
        :return:
        """
        sizes = pred_3D_lanes.shape
        # reshape to N x ipm_w/8 x 3 x (2K+1)
        pred_3D_lanes = pred_3D_lanes.reshape(sizes[0], sizes[1], self.num_types, self.anchor_dim)
        gt_3D_lanes = gt_3D_lanes.reshape(sizes[0], sizes[1], self.num_types, self.anchor_dim)
        # class prob N x ipm_w/8 x 3 x 1, anchor value N x ipm_w/8 x 3 x 2K
        pred_class = pred_3D_lanes[:, :, :, -1].unsqueeze(-1)
        pred_anchors = pred_3D_lanes[:, :, :, :-1]
        gt_class = gt_3D_lanes[:, :, :, -1].unsqueeze(-1)
        gt_anchors = gt_3D_lanes[:, :, :, :-1]

        loss1 = -torch.sum(gt_class*torch.log(pred_class + torch.tensor(1e-9)) +
                           (torch.ones_like(gt_class)-gt_class) *
                           torch.log(torch.ones_like(pred_class)-pred_class + torch.tensor(1e-9)))
        # applying L1 norm does not need to separate X and Z
        loss2 = torch.sum(torch.norm(gt_class*(pred_anchors-gt_anchors), p=1, dim=3))
        if not self.pred_cam:
            return loss1+loss2
        loss3 = torch.sum(torch.abs(gt_pitch-pred_pitch))+torch.sum(torch.abs(gt_hcam-pred_hcam))
        return loss1+loss2+loss3


class Laneline_loss_gflat(nn.Module):
    """
    Compute the loss between predicted lanelines and ground-truth laneline in anchor representation.
    The anchor representation is in flat ground space X', Y' and real 3D Z. Visibility estimation is also included.

    loss = loss0 + loss1 + loss2 + loss2
    loss0: cross entropy loss for lane point visibility
    loss1: cross entropy loss for lane type classification
    loss2: sum of geometric distance betwen 3D lane anchor points in X and Z offsets
    loss3: error in estimating pitch and camera heights
    """
    def __init__(self, num_types, num_y_steps, pred_cam):
        super(Laneline_loss_gflat, self).__init__()
        self.num_types = num_types
        self.num_y_steps = num_y_steps
        self.anchor_dim = 3*self.num_y_steps + 1
        self.pred_cam = pred_cam

    def forward(self, pred_3D_lanes, gt_3D_lanes, pred_hcam, gt_hcam, pred_pitch, gt_pitch):
        """

        :param pred_3D_lanes: predicted tensor with size N x (ipm_w/8) x 3*(2*K+1)
        :param gt_3D_lanes: ground-truth tensor with size N x (ipm_w/8) x 3*(2*K+1)
        :param pred_pitch: predicted pitch with size N
        :param gt_pitch: ground-truth pitch with size N
        :param pred_hcam: predicted camera height with size N
        :param gt_hcam: ground-truth camera height with size N
        :return:
        """
        sizes = pred_3D_lanes.shape
        # reshape to N x ipm_w/8 x 3 x (3K+1)
        pred_3D_lanes = pred_3D_lanes.reshape(sizes[0], sizes[1], self.num_types, self.anchor_dim)
        gt_3D_lanes = gt_3D_lanes.reshape(sizes[0], sizes[1], self.num_types, self.anchor_dim)
        # class prob N x ipm_w/8 x 3 x 1, anchor value N x ipm_w/8 x 3 x 2K
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
            (torch.ones_like(gt_class)-gt_class) *
            torch.log(torch.ones_like(pred_class) - pred_class + torch.tensor(1e-9)))
        # applying L1 norm does not need to separate X and Z
        loss2 = torch.sum(torch.norm(gt_class*torch.cat((gt_visibility, gt_visibility), 3) *
                                     (pred_anchors-gt_anchors), p=1, dim=3))
        if not self.pred_cam:
            return loss0+loss1+loss2
        loss3 = torch.sum(torch.abs(gt_pitch-pred_pitch))+torch.sum(torch.abs(gt_hcam-pred_hcam))
        return loss0+loss1+loss2+loss3


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
    def __init__(self, config, args, anchor_x_steps, x_off_std, z_std, y_off_std = None):
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
        self.anchor_dim = 3*self.num_y_steps + 1 + 12

        # prepare broadcast anchor_x_tensor, anchor_y_tensor, std_X, std_Y, std_Z
        tmp_zeros = torch.zeros(self.batch_size, self.num_x_steps, self.num_types, self.num_y_steps)
        self.x_off_std = torch.tensor(x_off_std.astype(np.float32)).reshape(1, 1, 1, self.num_y_steps) + tmp_zeros
        #self.y_off_std = torch.tensor(y_off_std.astype(np.float32)).reshape(1, 1, 1, self.num_y_steps) + tmp_zeros
        self.z_std = torch.tensor(z_std.astype(np.float32)).reshape(1, 1, 1, self.num_y_steps) + tmp_zeros
        self.anchor_y_steps = torch.tensor(np.array(config.anchor_y_steps).astype(np.float32)).reshape(1, 1, 1, self.num_y_steps) + tmp_zeros
        self.anchor_x_tensor = torch.tensor(anchor_x_steps.astype(np.float32)).reshape(1, self.num_x_steps, 1, 1) + tmp_zeros
        #self.anchor_y_tensor = torch.tensor(anchor_y_steps.astype(np.float32)).reshape(1, 1, 1, self.num_y_steps) + tmp_zeros
        #self.anchor_x_tensor = self.anchor_x_tensor/self.x_off_std
        #self.anchor_y_tensor = self.anchor_y_tensor/self.y_off_std
        self.anchor_x_tensor = torch.cat((self.anchor_x_tensor.unsqueeze(2), self.anchor_x_tensor.unsqueeze(2)), dim=2)
        self.anchor_x_tensor = self.anchor_x_tensor.reshape(self.batch_size, 2 * self.num_x_steps, self.num_types, self.num_y_steps)
        # self.anchor_y_tensor = torch.cat((self.anchor_y_tensor.unsqueeze(2), self.anchor_y_tensor.unsqueeze(2)), dim=2)
        # self.anchor_y_tensor = self.anchor_y_tensor.reshape(1, 2 * self.num_x_steps, 1, 1)
        #self.x_off_std = torch.cat((self.x_off_std, self.x_off_std), dim=1)
        #self.z_std = torch.cat((self.z_std, self.z_std), dim=1)
        self.anchor_y_steps = torch.cat((self.anchor_y_steps, self.anchor_y_steps), dim=1)
        self.x_off_std = torch.cat((self.x_off_std, self.x_off_std), dim=1)
        #self.y_off_std = torch.cat((self.y_off_std, self.y_off_std), dim=1)
        self.z_std = torch.cat((self.z_std, self.z_std), dim=1)

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


    def forward(self, pred_3D_lanes, gt_3D_lanes, gt_hcam, idx, type_weight, color_weight):
        """

        :param pred_3D_lanes: predicted tensor with size N x (ipm_w/8) x 3*(2*K+1)
        :param gt_3D_lanes: ground-truth tensor with size N x (ipm_w/8) x 3*(2*K+1)
        :param pred_pitch: predicted pitch with size N
        :param gt_pitch: ground-truth pitch with size N
        :param pred_hcam: predicted camera height with size N
        :param gt_hcam: ground-truth camera height with size N
        :return:
        """
        sizes = pred_3D_lanes.shape
        # reshape to N x ipm_w/8 x 3 x (3K+1)
        pred_3D_lanes = pred_3D_lanes.reshape(sizes[0], sizes[1], self.num_types, self.anchor_dim)
        pred_3D_lanes = torch.cat([pred_3D_lanes[:, :16, :, :].unsqueeze(2), pred_3D_lanes[:, 16:, :, :].unsqueeze(2)], dim=2)  # N*16*2*1*(3K+1)
        pred_3D_lanes = pred_3D_lanes.reshape(sizes[0], sizes[1], self.num_types, self.anchor_dim)
        gt_3D_lanes = gt_3D_lanes.reshape(sizes[0], sizes[1], self.num_types, self.anchor_dim)
        gt_3D_lanes = torch.cat([gt_3D_lanes[:, :16, :, :].unsqueeze(2), gt_3D_lanes[:, 16:, :, :].unsqueeze(2)],
                                  dim=2)  # N*16*2*1*(3K+1)
        gt_3D_lanes = gt_3D_lanes.reshape(sizes[0], sizes[1], self.num_types, self.anchor_dim)
        # class prob N x ipm_w/8 x 3 x 1, anchor values N x ipm_w/8 x 3 x 2K, visibility N x ipm_w/8 x 3 x K
        pred_class = pred_3D_lanes[:, :, :, -1].unsqueeze(-1)
        pred_anchors = pred_3D_lanes[:, :, :, :2*self.num_y_steps]
        pred_visibility = pred_3D_lanes[:, :, :, 2*self.num_y_steps:3*self.num_y_steps]
        pred_type = pred_3D_lanes[:, :, :, 3*self.num_y_steps:3*self.num_y_steps+9]
        #pred_type = F.softmax(pred_type, dim=-1)
        pred_color = pred_3D_lanes[:, :, :, 3 * self.num_y_steps + 9:3 * self.num_y_steps + 12]
        #pred_color = F.softmax(pred_color, dim=-1)
        gt_class = gt_3D_lanes[:, :, :, -1].unsqueeze(-1)
        gt_anchors = gt_3D_lanes[:, :, :, :2*self.num_y_steps]
        gt_visibility = gt_3D_lanes[:, :, :, 2*self.num_y_steps:3*self.num_y_steps]
        gt_type = gt_3D_lanes[:, :, :, 3*self.num_y_steps:3*self.num_y_steps+9]
        #gt_type_mask = torch.tensor(1) - gt_type[:, :, :, 0]  # 不考虑其他类型的线型
        #type_weight[0] = 0
        type_weight = torch.tensor(copy.deepcopy(type_weight).astype(np.float32)).reshape(1, 1, 1, 9).to(self.device)
        color_weight = torch.tensor(copy.deepcopy(color_weight).astype(np.float32)).reshape(1, 1, 1, 3).to(self.device)
        weight_type = torch.sum((gt_type == 1) * type_weight, dim=-1)


        gt_color = gt_3D_lanes[:, :, :, 3 * self.num_y_steps + 9:3 * self.num_y_steps + 12]
        weight_color = torch.sum((gt_color == 1) * color_weight, dim=-1)

        #print('color', gt_color.reshape(-1, 3))
        #print('weight', weight_color.reshape(-1, 1))



        #print(F.cross_entropy(pred_type.reshape(-1, 9), gt_type.reshape(-1, 9), reduce=False).shape)
        # print(gt_class.shape)

        loss_type = torch.sum(gt_class.reshape(-1, 1) * weight_type.reshape(-1, 1)*F.cross_entropy(pred_type.reshape(-1, 9), gt_type.reshape(-1, 9), reduce=False))
        loss_color = torch.sum(gt_class.reshape(-1, 1) * weight_color.reshape(-1,1)*F.cross_entropy(pred_color.reshape(-1, 3), gt_color.reshape(-1, 3), reduce=False))

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
        # loss2 = torch.sum(
        #     torch.norm(gt_class*torch.cat((gt_visibility, gt_visibility), 3)*(pred_anchors-gt_anchors), p=1, dim=3))

        # compute loss in real 3D X, Y space, the transformation considers offset to anchor and normalization by std
        pred_Xoff_g = pred_anchors[:, :, :, :self.num_y_steps]
        pred_Z = pred_anchors[:, :, :, self.num_y_steps:2*self.num_y_steps]
        gt_Xoff_g = gt_anchors[:, :, :, :self.num_y_steps]
        #gt_Z = gt_anchors[:, :, :, self.num_y_steps:2*self.num_y_steps]
        gt_hcam = gt_hcam.reshape(self.batch_size, 1, 1, 1)

        loss2 = torch.sum(
            torch.norm(gt_class * gt_visibility* (pred_Xoff_g - gt_Xoff_g), p=1,
                       dim=3))


        #gt_Z = gt_Z * self.z_std  # N x ipm_w/8 x1 x K
        pred_Z = pred_Z * self.z_std  # N x ipm_w/8 x K

        #print('x loss:', loss2.item())
        #pred_X_g = pred_Xoff_g * self.x_off_std + self.anchor_x_tensor
        gt_X_g = gt_Xoff_g * self.x_off_std + self.anchor_x_tensor  # bev_x

        pred_X_gt_Xg = (1 - pred_Z / gt_hcam) * gt_X_g  # 3d_x

        pred_Y = (1 - pred_Z / gt_hcam) * self.anchor_y_steps  # 3d_y

        #gt_Y = (1 - gt_Z / gt_hcam) * self.anchor_y_steps


        loss_width = []
        loss_height = []
        #print(pred_X_gt_Xg.shape)
        for i in range(pred_X_gt_Xg.shape[0]):
            if torch.sum(gt_class[i]) == 0:
                continue
            #lane_i = torch.cat([pred_X_gt_Xg[i, j, :, :] for j in range(pred_X_gt_Xg.shape[1]) if gt_class[i, j, 0, 0] > 0], dim=0)  # num_lane x k
            number = [j for j in range(pred_X_gt_Xg.shape[1]) if gt_class[i, j, 0, 0] > 0]
            if len(number) < 2:
                continue

            mask = torch.ones(len(number) - 1)
            mask_num = [j - 1 for j in range(1, len(number)) if (number[j] % 2 == 1 and number[j] - number[j - 1] == 1)]  # 拿到存在双线的anchor_ID
            mask[mask_num] = 0
            mask_num2 = [j - 1 for j in range(1, len(number)) if (gt_X_g[i, number[j], 0, 5] - gt_X_g[i, number[j-1], 0, 5]) > 5] # 拿到相距过远的两条线的anchor_ID
            mask[mask_num2] = 0
            mask_num3 = [j - 1 for j in range(1, len(number)) if
                         (gt_X_g[i, number[j], 0, 5] - gt_X_g[i, number[j-1], 0, 5]) < 1]  # 拿到相距过近的两条线的anchor_ID
            mask_num4 = [j - 1 for j in range(1, len(number)) if
                         (gt_type[i, number[j], 0, -1] > 0.5 or gt_type[i, number[j-1], 0, -1] > 0.5)]  # 不为路沿
            mask[mask_num3] = 0
            mask[mask_num4] = 0
            mask = mask.reshape((-1, 1)).to(self.device)
            #print(number)
            #print(mask)
            pred_X_i = pred_X_gt_Xg[i][number].squeeze()
            #print(pred_X_i.shape)
            pred_X0_g_i = gt_X_g[i][number].squeeze()[:, :1]
            pred_Y_i = pred_Y[i][number].squeeze()
            pred_Z_i = pred_Z[i][number].squeeze()
            gt_visibility_i = gt_visibility[i][number].squeeze()

            num_vis = [j for j in range(pred_X_i.shape[1]) if
                       (torch.sum(gt_visibility_i[:, j]) >= (pred_X_i.shape[0] - 0.5) or j < 5)]
            pred_X_i_vis = pred_X_i[:, num_vis]
            pred_Y_i_vis = pred_Y_i[:, num_vis]
            pred_Z_i_vis = pred_Z_i[:, num_vis]
            pred_Y_i_vis = pred_Y_i_vis.detach()

            pred_YZ_i_vis = torch.cat([pred_Y_i_vis.unsqueeze(2), pred_Z_i_vis.unsqueeze(2)], dim=2)
            pred_lYZ_i_vis = torch.norm(pred_YZ_i_vis[:, 1:, :] - pred_YZ_i_vis[:, :-1, :], dim=-1)
            pred_lYZ_i_vis = torch.cat([pred_lYZ_i_vis[:, :1], pred_lYZ_i_vis], dim=1)

            pred_XYZ_i_vis = torch.cat(
                [pred_X_i_vis.unsqueeze(2), pred_Y_i_vis.unsqueeze(2), pred_Z_i_vis.unsqueeze(2)], dim=2)
            pred_l_i_vis = torch.norm(pred_XYZ_i_vis[:, 1:, :] - pred_XYZ_i_vis[:, :-1, :], dim=-1)
            pred_l_i_vis = torch.cat([pred_l_i_vis[:, :1], pred_l_i_vis], dim=1)

            pred_W_i_vis = torch.abs(pred_X_i_vis[1:, :] - pred_X_i_vis[:-1, :])  # shape(lane_num-1, y_vis_num)

            pred_lYZ_i_vis = pred_lYZ_i_vis.detach()
            pred_l_i_vis = pred_l_i_vis.detach()

            gt_Xg_i = gt_X_g[i][number].squeeze()
            gt_Yg_i = self.anchor_y_steps[i][number].squeeze()
            gt_Xg_i_vis = gt_Xg_i[:, num_vis]
            gt_Yg_i_vis = gt_Yg_i[:, num_vis]
            gt_XYg_i_vis = torch.cat([gt_Xg_i_vis.unsqueeze(2), gt_Yg_i_vis.unsqueeze(2)], dim=2)
            gt_lg_i_vis = torch.norm(gt_XYg_i_vis[:, 1:, :] - gt_XYg_i_vis[:, :-1, :],
                                     dim=-1)  # shape(lane_num, y_vis_num-1)
            gt_lXg_i_vis = gt_Xg_i_vis[:, 1:] - gt_Xg_i_vis[:, :-1]  # shape(lane_num, y_vis_num-1)
            gt_lYg_i_vis = torch.abs(gt_Yg_i_vis[:, 1:] - gt_Yg_i_vis[:, :-1])  # shape(lane_num, y_vis_num-1)

            sin_alpha = (gt_lXg_i_vis / gt_lg_i_vis)[:-1, :]  # shape(lane_num-1, y_vis_num-1)
            cos_alpha = (gt_lYg_i_vis / gt_lg_i_vis)[:-1, :]
            sin_belta = (gt_lXg_i_vis / gt_lg_i_vis)[1:, :]
            cos_belta = (gt_lYg_i_vis / gt_lg_i_vis)[1:, :]
            xishu_i = 2 / ((1 / (cos_alpha + self.eps) + 1 / (cos_belta + self.eps)) *
                           torch.sqrt((self.one + cos_alpha * cos_belta + sin_alpha * sin_belta) / 2))

            width_i = pred_W_i_vis * pred_lYZ_i_vis[:-1, :] / (pred_l_i_vis[:-1, :] + self.eps)
            width0_i = torch.abs(pred_X0_g_i[1:, :] - pred_X0_g_i[:-1, :]) * xishu_i[:, :1]  # shape(lane_num-1, 1)

            ori_width_i = torch.cat([width0_i, width_i], dim=1)

            width_error_i = torch.abs(ori_width_i[:, 1:] - ori_width_i[:, :-1]) * mask

            loss_width_i = torch.sum(width_error_i)
            loss_width.append(loss_width_i)

            pred_Z_i_vis_diff = torch.abs(pred_Z_i_vis[1:, :] - pred_Z_i_vis[:-1, :])

            if pred_Z_i_vis.shape[0] < 2:
                continue

            loss_height_i = pred_Z_i_vis_diff
            loss_height.append(torch.sum(loss_height_i))



        if len(loss_width) == 0:
            loss_width = 0
        else:
            #print(loss_width)
            loss_width = torch.sum(torch.stack(loss_width, dim=0))

        if len(loss_height) == 0:
            loss_height = 0
        else:
            loss_height = torch.sum(torch.stack(loss_height, dim=0))

        #print(loss_type, loss_color)


        return 10*loss0+loss1+loss2 + 10*loss_width + 10*loss_height + 0.01*loss_type + 0.01*loss_color, loss_width



# unit test
if __name__ == '__main__':
    num_types = 3

    # for Laneline_loss_3D
    print('Test Laneline_loss_3D')
    anchor_dim = 2*6 + 1
    pred_cam = True
    criterion = Laneline_loss_3D(num_types, anchor_dim, pred_cam)
    criterion = criterion.cuda()

    pred_3D_lanes = torch.rand(8, 26, num_types*anchor_dim).cuda()
    gt_3D_lanes = torch.rand(8, 26, num_types*anchor_dim).cuda()
    pred_pitch = torch.ones(8).float().cuda()
    gt_pitch = torch.ones(8).float().cuda()
    pred_hcam = torch.ones(8).float().cuda()
    gt_hcam = torch.ones(8).float().cuda()

    loss = criterion(pred_3D_lanes, gt_3D_lanes, pred_pitch, gt_pitch, pred_hcam, gt_hcam)
    print(loss)

    # for Laneline_loss_gflat
    print('Test Laneline_loss_gflat')
    num_y_steps = 6
    anchor_dim = 3*num_y_steps + 1
    pred_cam = True
    criterion = Laneline_loss_gflat(num_types, num_y_steps, pred_cam)
    criterion = criterion.cuda()

    pred_3D_lanes = torch.rand(8, 26, num_types*anchor_dim).cuda()
    gt_3D_lanes = torch.rand(8, 26, num_types*anchor_dim).cuda()
    pred_pitch = torch.ones(8).float().cuda()
    gt_pitch = torch.ones(8).float().cuda()
    pred_hcam = torch.ones(8).float().cuda()
    gt_hcam = torch.ones(8).float().cuda()

    loss = criterion(pred_3D_lanes, gt_3D_lanes, pred_pitch, gt_pitch, pred_hcam, gt_hcam)

    print(loss)

    # for Laneline_loss_gflat_3D
    print('Test Laneline_loss_gflat_3D')
    batch_size = 8
    anchor_x_steps = np.linspace(-10, 10, 26, endpoint=True)
    anchor_y_steps = np.array([3, 5, 10, 20, 30, 40, 50, 60, 80, 100])
    num_y_steps = anchor_y_steps.shape[0]
    x_off_std = np.ones(num_y_steps)
    y_off_std = np.ones(num_y_steps)
    z_std = np.ones(num_y_steps)
    pred_cam = True
    criterion = Laneline_loss_gflat_3D(batch_size, num_types, anchor_x_steps, anchor_y_steps, x_off_std, y_off_std, z_std, pred_cam, no_cuda=False)
    # criterion = criterion.cuda()

    anchor_dim = 3*num_y_steps + 1
    pred_3D_lanes = torch.rand(batch_size, 26, num_types*anchor_dim).cuda()
    gt_3D_lanes = torch.rand(batch_size, 26, num_types*anchor_dim).cuda()
    pred_pitch = torch.ones(batch_size).float().cuda()
    gt_pitch = torch.ones(batch_size).float().cuda()
    pred_hcam = torch.ones(batch_size).float().cuda()*1.5
    gt_hcam = torch.ones(batch_size).float().cuda()*1.5

    loss = criterion(pred_3D_lanes, gt_3D_lanes, pred_pitch, gt_pitch, pred_hcam, gt_hcam)

    print(loss)