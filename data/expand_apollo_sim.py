import json
import numpy as np
import math
import random
import copy


def projection_g2c(cam_pitch, cam_height, gt_cam_yaw = 0):
    gt_cam_yaw = - gt_cam_yaw
    R_yaw = np.array([[math.cos(gt_cam_yaw), -math.sin(gt_cam_yaw), 0],
                    [math.sin(gt_cam_yaw), math.cos(gt_cam_yaw), 0],
                    [0, 0, 1]])

    R_pitch = np.array([[1, 0, 0],
                      [0, np.cos(np.pi / 2 + cam_pitch), -np.sin(np.pi / 2 + cam_pitch)],
                      [0, np.sin(np.pi / 2 + cam_pitch), np.cos(np.pi / 2 + cam_pitch)]])
    R_g2c = np.matmul(R_pitch, R_yaw)
    T0 = np.array([[0], [0], [cam_height]])
    T = -np.matmul(R_g2c, T0)

    return R_g2c, T

def projection_c2g(cam_pitch, cam_height):
    R_g2c = np.array([[1,                             0,                              0],
                      [0, np.cos(np.pi / 2 + cam_pitch), -np.sin(np.pi / 2 + cam_pitch)],
                      [0, np.sin(np.pi / 2 + cam_pitch),  np.cos(np.pi / 2 + cam_pitch)]])
    R_c2g = np.linalg.inv(R_g2c)
    return R_c2g, [[0], [0], [cam_height]]

camara_intrin = np.array([
        [2015, 0, 960],
        [0, 2015, 540],
        [0, 0, 1]], dtype='double')

jsonpath = './data_splits/standard/train.json'
new_jsonpath = './data_splits/standard/train_ext.json'
with open(new_jsonpath, 'w') as json_file:
    with open(jsonpath, 'r') as f:
        for line in f:
            dic = json.loads(line)
            cam_height = dic['cam_height']
            cam_pitch = dic['cam_pitch']
            laneLines = dic['laneLines']
            new_dict = dic
            json.dump(dic, json_file)
            json_file.write('\n')
            R_g2c, T_g2c = projection_g2c(cam_pitch, cam_height)

            new_dict = copy.deepcopy(dic)
            new_cam_pitch = copy.deepcopy(cam_pitch) + 0.017 * random.random()
            R_c2g, T_c2g = projection_c2g(new_cam_pitch, cam_height)
            new_laneLines = []
            laneLines_cp = copy.deepcopy(laneLines)
            for idx, line in enumerate(laneLines_cp):  # N * 3 * p
                line_g = np.array(line).T  #3*p
                line_c = np.dot(R_g2c, line_g) + T_g2c
                new_line_g = (np.dot(R_c2g, line_c) + T_c2g).T
                new_line_g = new_line_g.tolist()
                new_laneLines.append(new_line_g)
            new_dict['laneLines'] = new_laneLines
            new_dict['cam_pitch'] = new_cam_pitch
            json.dump(new_dict, json_file)
            json_file.write('\n')

            new_dict = copy.deepcopy(dic)
            new_cam_pitch = copy.deepcopy(cam_pitch) - 0.017 * random.random()
            R_c2g, T_c2g = projection_c2g(new_cam_pitch, cam_height)
            new_laneLines = []
            laneLines_cp = copy.deepcopy(laneLines)
            for idx, line in enumerate(laneLines_cp):  # N * 3 * p
                line_g = np.array(line).T  # 3*p
                line_c = np.dot(R_g2c, line_g) + T_g2c
                new_line_g = (np.dot(R_c2g, line_c) + T_c2g).T
                new_line_g = new_line_g.tolist()
                new_laneLines.append(new_line_g)
            new_dict['laneLines'] = new_laneLines
            new_dict['cam_pitch'] = new_cam_pitch
            json.dump(new_dict, json_file)
            json_file.write('\n')

















