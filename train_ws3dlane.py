import argparse
import torch
import time
import json
from torch import distributed, optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from data.Load_Data_3DLane_ext_pitch import LaneDataset, unormalize_lane_anchor, compute_3d_lanes_all_prob
from model.LaneNet3D_ext import Net
from model.Loss_crit_ws3dlane import Laneline_loss_gflat_3D
from model.seg_loss import BevSegLoss
from utils.logger import get_logger
from utils.base_config import get_config
from utils.initializer import initializer
from eval_3D_lane import LaneEval
from utils.metrics import AngleMAE
from torch.nn import L1Loss
import datetime
import numpy as np

def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    rt = tensor.detach()
    distributed.all_reduce(rt, op=distributed.reduce_op.SUM)
    rt /= distributed.get_world_size()
    return rt

class Trainer(object):
    def __init__(self, config, args):

        super(Trainer, self).__init__()
        self.config = config
        self.args = args
        self.logger = get_logger(config, config.train.num_workers)

        torch.cuda.set_device(args.local_rank)
        distributed.init_process_group(
            'nccl',
            init_method='env://',
            timeout=datetime.timedelta(seconds=3600)
        )
        self.device = torch.device(f'cuda:{args.local_rank}')
        self.epochs = config.train.epochs
        self.batch_size = config.batch_size
        self.lr = config.train.lr
        self.weight_decay = config.train.wd
        self.train_dataset = None
        self.val_dataset = None
        self.train_sampler = None
        self.val_sampler = None
        self.train_loader = None
        self.val_loader = None
        self.scheduler = None
        self.optimizer = None
        self.network = None
        self.criterion = None
        self.weight_save_path = config.utils.output_prefix
        self.epoch_idx = 0
        self.set_network_optimizer()
        self.set_dataset()
        self.set_dataloader()
        self.set_scheduler()
        self.set_criterion()
        self.set_metrics()

    def set_dataset(self):
        with open('ws3dlane_anchor_std_20_ext.json', 'r') as f:
            anchor_std = json.load(f)
        self.train_dataset = LaneDataset(self.config, is_train=True, args = self.args)
        self.train_dataset.set_x_off_std(np.array(anchor_std['x_off_std']))
        self.train_dataset.set_z_std(np.array(anchor_std['z_std']))
        #self.train_dataset.normalize_lane_label()
        self.val_dataset = LaneDataset(self.config, is_train=False,  args = self.args)
        self.val_dataset.set_x_off_std(np.array(anchor_std['x_off_std']))
        self.val_dataset.set_z_std(np.array(anchor_std['z_std']))
        self.val_dataset.normalize_lane_label()

    def set_dataloader(self):
        self.train_sampler = DistributedSampler(self.train_dataset)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, num_workers=4,
                                  pin_memory=True, sampler=self.train_sampler, drop_last=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.config.batch_size, num_workers=4,
                                  pin_memory=True, drop_last=True)

    def set_network_optimizer(self):
        self.network = Net(self.config, self.args).to(self.device)

        self.init_type = self.config.train.init.initializer
        self.init_params = self.config.train.init.params
        if self.config.train.pretrain.active:
            self.logger.info('Init param form pretrain model: {}'.format(
                self.config.train.pretrain.param_path))
            self.network.load_state_dict(
                torch.load(self.config.train.pretrain.param_path), strict = False)
        else:
            for m in self.network.modules():
                if isinstance(m, torch.nn.Conv2d):
                    initializer(self.init_type, m.weight, self.init_params)
                    if m.bias is not None:
                        torch.nn.init.constant_(m.bias, 0)

            self.network.load_pretrained_vgg(batch_norm=True)

        if self.config.train.optimizer == 'adam':
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr, weight_decay=self.config.train.wd)

    def set_scheduler(self):
        warmup_epoches = self.config.train.warmup_epoches
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.1 * (
                    epoch + 1) if epoch + 1 <= warmup_epoches else 1.0001 - (epoch + 1 - warmup_epoches)/(self.epochs - warmup_epoches))

    def set_criterion(self):
        self.lane_criterion = Laneline_loss_gflat_3D(self.config,
                                                self.args,
                                                self.train_dataset.anchor_x_steps,
                                                self.train_dataset._x_off_std,
                                                self.train_dataset._z_std,
                                                ).to(self.device)
        self.lane_val_criterion = Laneline_loss_gflat_3D(self.config,
                                                self.args,
                                                self.train_dataset.anchor_x_steps,
                                                self.train_dataset._x_off_std,
                                                self.train_dataset._z_std,
                                                ).to(self.device)

        self.pose_l1_criterion = L1Loss().to(self.device)

        self.seg_criterion = BevSegLoss().to(self.device)


    def set_metrics(self):
        # 定义评测函数
        self.evaluator = LaneEval(self.config)
        self.train_metric = AngleMAE(config=self.config, name='pitch_error', is_train=True)
        self.val_metric = AngleMAE(config=self.config, name='pitch_error', is_train=False)

    def evaluation(self):
        val_gt_file = self.config.dataset.val.json_path[0]
        valid_set_labels = [json.loads(line) for line in open(val_gt_file).readlines()]
        lane_pred_file = 'ws3dlanenet_pred.json'
        with torch.no_grad():
            self.network.eval()
            val_loss1 = 0.0
            val_loss2 = 0.0
            with open(lane_pred_file, 'w') as jsonFile:
                for batch_idx, batch in enumerate(self.val_loader):
                    img, gt, idx, gt_hcam, gt_pitch, imgname, bev_seg_label = batch
                    img = img.to(self.device, dtype=torch.float32)
                    gt_hcam = gt_hcam.to(self.device)
                    gt_pitch = gt_pitch.to(self.device)
                    gt_pitch = gt_pitch.unsqueeze(1)
                    if not self.config.fix_cam and not self.config.pred_cam:
                        self.network.update_projection(self.config, gt_hcam, gt_pitch)
                    output, _, output_pitch, _ = self.network(img)
                    gt = gt.to(self.device)
                    batch_loss1 = self.lane_val_criterion(output, gt, gt_hcam)
                    batch_loss2 = self.pose_l1_criterion(output_pitch, gt_pitch)
                    reduced_loss1 = reduce_tensor(batch_loss1.data)
                    val_loss1 += reduced_loss1.item()  # 统计损失
                    reduced_loss2 = reduce_tensor(batch_loss2.data)
                    val_loss2 += reduced_loss2.item()  # 统计损失
                    gt_hcam = gt_hcam.data.cpu().numpy().flatten()

                    output = output.data.cpu().numpy()
                    gt = gt.data.cpu().numpy()
                    num_el = img.size(0)
                    for j in range(num_el):
                        unormalize_lane_anchor(output[j], self.val_dataset)
                        unormalize_lane_anchor(gt[j], self.val_dataset)

                    for j in range(num_el):
                        im_id = idx[j]
                        json_line = valid_set_labels[im_id]
                        lane_anchors = output[j]
                        lanelines_pred, centerlines_pred, lanelines_prob, centerlines_prob = \
                            compute_3d_lanes_all_prob(lane_anchors, self.val_dataset.anchor_dim,
                                                      self.val_dataset.anchor_x_steps, self.val_dataset.anchor_y_steps, gt_hcam[j])
                        json_line["laneLines"] = lanelines_pred
                        json_line["centerLines"] = centerlines_pred
                        json_line["laneLines_prob"] = lanelines_prob
                        json_line["centerLines_prob"] = centerlines_prob
                        json.dump(json_line, jsonFile)
                        jsonFile.write('\n')

            eval_stats = self.evaluator.bench_one_submit(lane_pred_file, val_gt_file)
            eval_stats_pr = self.evaluator.bench_one_submit_varying_probs(lane_pred_file, val_gt_file)

            if self.args.local_rank == 0:
                names, values = self.val_metric.get()
                val_s = "Epoch[{}] Evaluation:".format(self.epoch_idx + 1)  # noqa
                val_s += "val_loss1={:.6f},".format(val_loss1/batch_idx)
                val_s += "val_loss2={:.6f}\n".format(val_loss2 / batch_idx)
                val_s += "{}={:.3f}\n".format(names, values)
                val_s += "laneline AP {:.8} \n".format(eval_stats_pr['laneline_AP'])
                val_s += "laneline F-measure {:.8} \n".format(eval_stats[0])
                val_s += "laneline Recall  {:.8} \n".format(eval_stats[1])
                val_s += "laneline Precision  {:.8} \n".format(eval_stats[2])
                val_s += "laneline x error (close)  {:.8} m\n".format(eval_stats[3])
                val_s += "laneline x error (far)  {:.8} m\n".format(eval_stats[4])
                val_s += "laneline z error (close)  {:.8} m\n".format(eval_stats[5])
                val_s += "laneline z error (far)  {:.8} m\n".format(eval_stats[6])
                self.logger.info(val_s)


    def train(self):

        for self.epoch_idx in range(self.epochs):
            train_epoch_loss1 = 0.0
            train_epoch_loss2 = 0.0
            train_epoch_loss3 = 0.0
            self.network.train()
            batch_tic = time.time()
            self.train_dataset.pitch_aug()
            self.train_dataset.normalize_lane_label()
            self.train_sampler.set_epoch(self.epoch_idx)
            self.optimizer.zero_grad()

            for batch_idx, batch in enumerate(self.train_loader):
                img, gt, idx, gt_hcam, gt_pitch, imgname, bev_seg_label = batch
                img = img.to(self.device, dtype=torch.float32)
                gt_hcam = gt_hcam.to(self.device)
                gt_pitch = gt_pitch.to(self.device)
                if not self.config.fix_cam and not self.config.pred_cam:
                    self.network.update_projection(self.config, gt_hcam, gt_pitch)
                output, _, output_pitch, bev_seg = self.network(img)
                bev_seg_label = bev_seg_label.to(self.device)
                gt = gt.to(self.device)
                gt_pitch = gt_pitch.unsqueeze(1)
                batch_loss1 = self.lane_criterion(output, gt, gt_hcam)
                #batch_loss2 = self.pose_l1_criterion(output_pitch, gt_pitch)
                batch_loss3 = self.seg_criterion(bev_seg, bev_seg_label)
                batch_loss = 10*batch_loss1 + batch_loss3
                reduced_loss1 = reduce_tensor(batch_loss1.data)
                train_epoch_loss1 += reduced_loss1.item()
                # reduced_loss2 = reduce_tensor(batch_loss2.data)
                # train_epoch_loss2 += reduced_loss2.item()
                reduced_loss3 = reduce_tensor(batch_loss3.data)
                train_epoch_loss3 += reduced_loss3.item()
                if np.isnan(train_epoch_loss3) or np.isnan(train_epoch_loss1):
                    continue
                batch_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.args.local_rank == 0:
                    if (batch_idx + 1) % self.config.train.frequent == 0:
                        speed = self.config.batch_size * \
                                (batch_idx + 1) / (time.time() - batch_tic)
                        lr = self.optimizer.param_groups[0]['lr']
                        loss1 = train_epoch_loss1 / self.config.train.frequent   # noqa
                        loss2 = train_epoch_loss2 / self.config.train.frequent
                        loss3 = train_epoch_loss3 / self.config.train.frequent # noqa
                        s = "Epoch[{}]".format(self.epoch_idx + 1) + " Batch[{}] Speed: {:.2f} samples/sec lr: {:.8f} loss1: {:.6f} loss2: {:.6f} loss3: {:.6f}".format(  # noqa
                                batch_idx + 1, speed, lr, loss1, loss2, loss3)
                        self.logger.info(s)
                        train_epoch_loss1 = 0
                        train_epoch_loss2 = 0
                        train_epoch_loss3 = 0

            self.scheduler.step()

            if self.args.local_rank == 0:
                self.logger.info('saving weights to' + self.weight_save_path + '/self_3dlanenet_%03d.pth', (self.epoch_idx + 1))
                torch.save(self.network.state_dict(), self.weight_save_path + '/self_3dlanenet_%03d.pth' % (self.epoch_idx + 1))

            if (self.epoch_idx + 1) % 10 == 0:
                self.evaluation()




def main():
    torch.multiprocessing.set_start_method('spawn')
    args = argparse.ArgumentParser(description='Train')
    args.add_argument('--local_rank', help='node rank for distributed training', default=-1, type=int)
    args.add_argument('--cfg', default='scripts/config_3dlane_apollo.yaml', type=str)
    args = args.parse_args()
    config = get_config(args)
    trainer = Trainer(config, args)
    trainer.train()


if __name__ == '__main__':
    main()

