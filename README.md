# (Official Repo) WS-3D-Lane: Weakly Supervised 3D Lane Detection With 2D Lane Labels

## Introduction

This is the official pytorch implementation of [WS-3D-Lane, which supervises 3D lane detection network with 2D lane labels.  **[ICRA 2023 paper]**](https://arxiv.org/pdf/2209.11523.pdf)


Key feature: A weakly supervised 3D lane detection method with only 2D lane labels.


## Baseline
This repo is based on the open source code of ['Gen-LaneNet'](https://github.com/yuliangguo/Pytorch_Generalized_3D_Lane_Detection) in pytorch. 

## Requirements
If you have Anaconda installed, you can directly import the provided environment file.

    conda env update --file environment.yaml

Those important packages includes:
* opencv-python             4.1.0.25
* pytorch                   1.4.0
* torchvision               0.5.0
* tensorboard               1.15.0
* tensorboardx              1.7
* py3-ortools               5.1.4041

## Data preparation

The 3D lane detection method is trained and tested on the 
[3D lane synthetic dataset](https://github.com/yuliangguo/3D_Lane_Synthetic_Dataset). Running the demo
code on a single image should directly work. However, repeating the training, testing and evaluation requires to prepare the dataset:
* Download the [raw datasets](https://drive.google.com/open?id=1Kisxoj7mYl1YyA_4xBKTE8GGWiNZVain). 
* Download the prepared [data splits and pretrained models](https://drive.google.com/open?id=1GDgiAmJdP_BEluAZDgMaclNwb34OenCn). 
* Put 'data_splits' in current directory.

If you prefer to build your own data splits using the dataset, please follow the steps described in the 3D lane 
synthetic dataset repository. All necessary codes are included here already. 

## How to train the model

Step 1: Revise your data path and save path in './scripts/config_3dlanenet_apollo_ws.yaml'.

Step 2: Train the WS-3D-Lane network.

    CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 9999 train_ws3dlane.py --cfg scripts/config_3dlanenet_apollo_ws.yaml

Change the training hyper-parameters may get better results.
## Evaluation

Step 1: Revise your data path and save path in 'scripts/config_3dlanenet_apollo_test.yaml'.

Step 2: Evaluate your network.

    python test.py

Note: our model weights are under *pth/* folder

## Citation

Please cite the paper in your publications if it helps your research: 

    @article{2023ws3dlane,
          title={WS-3D-Lane: Weakly Supervised 3D Lane Detection With 2D Lane Labels}, 
          author={Jianyong Ai and Wenbo Ding and Jiuhua Zhao and Jiachen Zhong},
          year={2023},
          eprint={2209.11523},
          archivePrefix={arXiv},
    }

## Copyright

The copyright of this work belongs to SAIC AILAB.


