import numpy as np
import torch
import os
from os import path
from config import get_config
from model import MipNeRF
import imageio
from datasets import get_dataloader
from tqdm import tqdm
from pose_utils import visualize_depth, visualize_normals, to8b


# 本文件专门用于渲染出图和出视频(不用于评估模型性能)，不是使用test数据集划分，而使用特定render数据集，经过一定算法生成有序的视角光线

def visualize(config):
    # 1. 加载数据集
    data = get_dataloader(
        config.dataset_name,
        config.base_dir,
        split="render",
        factor=config.factor,
        shuffle=False
    )
    # 强制指定设备（避免CPU/GPU不匹配）
    device = config.device  # 直接用 config 里的设备
    # 2. 加载模型（测试阶段参数适配）
    model = MipNeRF(
        use_viewdirs=config.use_viewdirs,
        randomized=False,  # 推理必须关随机采样，开启是为了训练时学习更加泛化
        ray_shape=config.ray_shape,
        white_bkgd=config.white_bkgd,
        num_levels=config.num_levels,
        num_samples=config.num_samples,
        hidden=config.hidden,
        density_noise=config.density_noise,
        density_bias=config.density_bias,
        rgb_padding=config.rgb_padding,
        resample_padding=config.resample_padding,
        min_deg=config.min_deg,
        max_deg=config.max_deg,
        viewdirs_min_deg=config.viewdirs_min_deg,
        viewdirs_max_deg=config.viewdirs_max_deg,
        device=device,
    )
    # 加载权重并迁移到指定设备
    state_dict = torch.load(config.model_weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"[INFO] 生成视频，共{len(data)}个视角 | 设备：{device}")

    # 准备容器
    rgb_frames = []
    depth_frames = [] if config.visualize_depth else None
    normal_frames = [] if config.visualize_normals else None

    # 创建单张图保存目录
    img_save_dir = path.join(config.log_dir, "single_frames")
    os.makedirs(img_save_dir, exist_ok=True)
    frame_idx = 0

    # 测试阶段关闭梯度
    with torch.no_grad():
        for frame_idx, ray in enumerate(tqdm(data)):
            # 直接渲染，不需要处理 ray
            img, dist, acc = model.render_image(ray, data.h, data.w, chunks=config.chunks)

            # 在裁剪代码前新增打印
            img_np = img.cpu().numpy() if isinstance(img, torch.Tensor) else img
            print(f"第{frame_idx}张图原始数值：min={img_np.min():.4f}, max={img_np.max():.4f}, mean={img_np.mean():.4f}")

            # ========== 保存单张图（解决白图） ==========
            # 转为NumPy
            if img_np.dtype != np.uint8:
                img_np = img_np.astype(np.uint8)
            img_8b = img_np
            # 保存图片
            img_path = path.join(img_save_dir, f"frame_{frame_idx:03d}.png")
            imageio.imwrite(img_path, img_8b)
            frame_idx += 1

            # 收集视频帧
            rgb_frames.append(img_8b)
            if config.visualize_depth:
                depth_frames.append(to8b(visualize_depth(dist, acc, data.near, data.far)))
            if config.visualize_normals:
                normal_frames.append(to8b(visualize_normals(dist, acc)))

            # 限制数量
            if frame_idx >= 25:  # 从0开始，到25就是26张
                break

    # 生成视频（兼容8位格式帧）//fps就是1秒拼接多少图，一般设为24或30
    imageio.mimwrite(path.join(config.log_dir, "video.mp4"), rgb_frames, fps=1, quality=10, codec="libx265")#作者给的是codecs="hvec"，而不是codec="libx265"，但好像作者的不行
    if config.visualize_depth:
        imageio.mimwrite(path.join(config.log_dir, "depth.mp4"), depth_frames, fps=24, quality=10, codec="libx265")
    if config.visualize_normals:
        imageio.mimwrite(path.join(config.log_dir, "normals.mp4"), normal_frames, fps=24, quality=10, codec="libx265")

    print(f"\n[完成] 渲染{frame_idx}张图，保存至：{img_save_dir}")
    print(f"视频保存至：{config.log_dir}/video.mp4")


if __name__ == "__main__":
    config = get_config()
    visualize(config)