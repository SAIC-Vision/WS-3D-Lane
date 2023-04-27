import argparse
import torch
import json
from torch.utils.data import DataLoader
from data.Load_Data_3DLane_ext_pitch import LaneDataset, unormalize_lane_anchor, compute_3d_lanes_all_prob
from model.LaneNet3D_ext import Net
from utils.logger import get_logger
from utils.base_config import get_config
from utils.initializer import initializer
from eval_3D_lane import LaneEval
import os
import numpy as np
from utils.tools import Visualizer

os.environ['CUDA_VISIBLE_DEVICES']='0'

class Trainer(object):
    def __init__(self, config, args):
        super(Trainer, self).__init__()
        self.config = config
        self.args = args
        self.logger = get_logger(config, config.train.num_workers)
        self.device = torch.device(f'cuda:0')
        self.config.device = self.device
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
        self.set_metrics()

    def set_dataset(self):
        with open('ws3dlane_anchor_std_20_ext.json', 'r') as f:
            anchor_std = json.load(f)
        self.val_dataset = LaneDataset(self.config, is_train=False)
        self.val_dataset.set_x_off_std(np.array(anchor_std['x_off_std']))
        self.val_dataset.set_z_std(np.array(anchor_std['z_std']))
        self.val_dataset.normalize_lane_label()
        self.vs_saver = Visualizer(self.config, vis_folder='val_vis_ws3dlane')

    def set_dataloader(self):
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
                torch.load(self.config.train.pretrain.param_path), strict=False)
        else:
            for m in self.network.modules():
                if isinstance(m, torch.nn.Conv2d):
                    initializer(self.init_type, m.weight, self.init_params)
                    if m.bias is not None:
                        torch.nn.init.constant_(m.bias, 0)

            self.network.load_pretrained_vgg(batch_norm=True)

    def set_metrics(self):
        self.evaluator = LaneEval(self.config)

    def evaluation(self):
        val_gt_file = self.config.dataset.val.json_path[0]
        valid_set_labels = [json.loads(line) for line in open(val_gt_file).readlines()]
        lane_pred_file = '3dlanenet_pred_file.json'
        with torch.no_grad():
            self.network.eval()
            with open(lane_pred_file, 'w') as jsonFile:
                for batch_idx, batch in enumerate(self.val_loader):
                    img, _, gt, idx, gt_hcam, gt_pitch, imgname, _ = batch
                    img = img.to(self.device, dtype=torch.float32)
                    gt_hcam = gt_hcam.to(self.device)
                    gt_pitch = gt_pitch.to(self.device)
                    if not self.config.fix_cam and not self.config.pred_cam:
                        self.network.update_projection(self.config, gt_hcam, gt_pitch)
                    output, _, _, _ = self.network(img)
                    gt = gt.to(self.device)
                    gt_hcam = gt_hcam.data.cpu().numpy().flatten()
                    output = output.data.cpu().numpy()
                    gt = gt.data.cpu().numpy()
                    num_el = img.size(0)
                    for j in range(num_el):
                        unormalize_lane_anchor(output[j], self.val_dataset)
                        unormalize_lane_anchor(gt[j], self.val_dataset)

                    self.vs_saver.save_result_new(self.val_dataset, 'val', 300, batch_idx, idx,
                                                  img, gt, output, gt_pitch, gt_hcam, evaluate=True)

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

            val_s = "Epoch[{}] Evaluation:".format(self.epoch_idx + 1)  # noqa
            val_s += "laneline AP {:.8} \n".format(eval_stats_pr['laneline_AP'])
            val_s += "laneline F-measure {:.8} \n".format(eval_stats[0])
            val_s += "laneline Recall  {:.8} \n".format(eval_stats[1])
            val_s += "laneline Precision  {:.8} \n".format(eval_stats[2])
            val_s += "laneline x error (close)  {:.8} m\n".format(eval_stats[3])
            val_s += "laneline x error (far)  {:.8} m\n".format(eval_stats[4])
            val_s += "laneline z error (close)  {:.8} m\n".format(eval_stats[5])
            val_s += "laneline z error (far)  {:.8} m\n".format(eval_stats[6])
            self.logger.info(val_s)


def main():
    args = argparse.ArgumentParser(description='Train')
    args.add_argument('--cfg', default='scripts/config_3dlanenet_apollo_test.yaml', type=str)
    args = args.parse_args()
    config = get_config(args)
    trainer = Trainer(config, args)
    trainer.evaluation()

if __name__ == '__main__':
    main()

