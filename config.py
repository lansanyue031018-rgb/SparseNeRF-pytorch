import argparse
import torch
from os import path


def get_config():
    config = argparse.ArgumentParser()

    # basic hyperparams to specify where to load/save data from/to
    config.add_argument("--log_dir", type=str, default="log")
    config.add_argument("--dataset_name", type=str, default="llff")
    config.add_argument("--scene", type=str, default="my6")
    # model hyperparams
    config.add_argument("--use_viewdirs", action="store_false")
    config.add_argument("--randomized", action="store_false")
    config.add_argument("--ray_shape", type=str, default="cone")  # should be "cylinder" if llff
    config.add_argument("--white_bkgd", action="store_false")  # should be False if using llff
    config.add_argument("--override_defaults", action="store_true")
    config.add_argument("--num_levels", type=int, default=2)
    config.add_argument("--num_samples", type=int, default=128)
    config.add_argument("--hidden", type=int, default=256)
    config.add_argument("--density_noise", type=float, default=0.0)
    config.add_argument("--density_bias", type=float, default=-1.0)
    config.add_argument("--rgb_padding", type=float, default=0.001)
    config.add_argument("--resample_padding", type=float, default=0.01)
    config.add_argument("--min_deg", type=int, default=0)
    config.add_argument("--max_deg", type=int, default=16)
    config.add_argument("--viewdirs_min_deg", type=int, default=0)
    config.add_argument("--viewdirs_max_deg", type=int, default=4)
    # loss and optimizer hyperparams
    config.add_argument("--coarse_weight_decay", type=float, default=0.1)
    config.add_argument("--lr_init", type=float, default=1e-3)
    config.add_argument("--lr_final", type=float, default=5e-5)
    config.add_argument("--lr_delay_steps", type=int, default=2500)
    config.add_argument("--lr_delay_mult", type=float, default=0.1)
    config.add_argument("--weight_decay", type=float, default=1e-5)

    config.add_argument("--depth_rank_weight", type=float, default=0.5)
    config.add_argument("--corner_nearest_pair_ratio", type=float, default=0)
    config.add_argument("--edge_rgb_only_ratio", type=float, default=0.05)
    # training/test hyperparams
    #对于真实世界数据集如llff数据集或者拍摄照片(建议也放llff文件夹)，往往分辨率很高，需要缩减分辨率，数据集是llff时会自动设为4将原图分辨率除4
    #对于blender数据集，渲染时可以改为1，评估时的输出图片和训练时输入图，均采用输入数据集图片分辨率的2倍下采样
    config.add_argument("--factor", type=int, default=2)
    #对于评估时，测试集视图叫多时，对测试集视图数量进行采样，取原测试数据集的默认1/8进行评估以减少评估时间，对应llff后面自动设置为1
    config.add_argument("--test_skip", type=int, default=8)
    config.add_argument("--max_steps", type=int, default=200_000)# 迭代次数设置
    config.add_argument("--batch_size", type=int, default=1024)# 批次大小(3060的6g显存实测1024稳定训练，2048甚至更大会爆显存)
    config.add_argument("--do_eval", action="store_true")#默认关闭训练过程中的自动评估
    config.add_argument("--continue_training", action="store_true")
    config.add_argument("--save_every", type=int, default=33000)#评估和保存权重文件和优化器配置文件的频率
    config.add_argument("--device", type=str, default="cuda")
    # visualization hyperparams
    config.add_argument("--chunks", type=int, default=4096)#光线分批渲染批次:3060的6g显存实测4096可渲染，8192甚至更大则不行
    config.add_argument("--model_weight_path", default="log/model.pt")
    config.add_argument("--visualize_depth", action="store_true")
    config.add_argument("--visualize_normals", action="store_true")
    # extracting mesh hyperparams
    config.add_argument("--x_range", nargs="+", type=float, default=[-1.2, 1.2])
    config.add_argument("--y_range", nargs="+", type=float, default=[-1.2, 1.2])
    config.add_argument("--z_range", nargs="+", type=float, default=[-1.2, 1.2])
    config.add_argument("--grid_size", type=int, default=256)
    config.add_argument("--sigma_threshold", type=float, default=50.0)
    config.add_argument("--occ_threshold", type=float, default=0.2)

    #训练时，是否使用稀疏数据集，默认不使用
    config.add_argument("--use_sparse", action='store_false', help='sampling linearly in disparity rather than depth')
    # 针对blender的稀疏文件名称（可改8/16/32个视角版本），官方的blender训练train数据集是360均分为100个视角图
    config.add_argument("--sparse_json_name", type=str, default="transforms_sparse_32.json",help='稀疏数据集的JSON文件名（如transforms_sparse_8/16/32.json）')
    # 针对llff稀疏文件名称（可改0.1/0.2/0.4/0.6等）
    config.add_argument("--sparse_ratio", type=float, default=0.2, help='稀疏比例')

    config = config.parse_args()

    # default configs for llff, automatically set if dataset is llff and not override_defaults
    if config.dataset_name == "llff" and not config.override_defaults:
        # 对应llff数据集，渲染时可以根据情况改为1/4/8，评估时的输出图片和训练时输入图，均采用输入数据集图片分辨率的4倍下采样
        config.factor = 4
        config.ray_shape = "cylinder"
        config.white_bkgd = False
        config.density_noise = 1.0
        config.test_skip = 1

    if config.scene == "my" and not config.override_defaults:
        config.factor = 1
        config.ray_shape = "cylinder"
        config.white_bkgd = False
        config.density_noise = 1.0
        config.test_skip = 1

    if config.scene == "my6" and not config.override_defaults:
        config.factor = 1
        config.ray_shape = "cylinder"
        config.white_bkgd = False
        config.density_noise = 1.0
        config.test_skip = 1

    if config.scene == "my4" and not config.override_defaults:
        config.factor = 1
        config.ray_shape = "cylinder"
        config.white_bkgd = False
        config.density_noise = 1.0
        config.test_skip = 1

    if config.scene == "my5" and not config.override_defaults:
        config.factor = 1
        config.ray_shape = "cylinder"
        config.white_bkgd = False
        config.density_noise = 1.0
        config.test_skip = 1

    config.device = torch.device(config.device)
    base_data_path = "data/nerf_llff_data/"
    if config.dataset_name == "blender":
        base_data_path = "data/nerf_synthetic/"
    elif config.dataset_name == "multicam":
        base_data_path = "data/nerf_multiscale/"
    config.base_dir = path.join(base_data_path, config.scene)

    return config
