"""
关键设计思路：
最小化 GPU 负载：GPU 只做 “模型前向推理” 这一件事，且分批进行，避免显存压力；
CPU 中心化评估：所有非渲染操作（张量处理、指标计算、图片保存）都在 CPU 完成，设备无冲突；
贴合数据集原始格式：直接操作 Dataset 而非 DataLoader，避免 “强行适配” 带来的逻辑混乱；
标准化流程：渲染→格式转换→指标计算→结果保存，每一步职责单一，易维护。
"""
# 导入系统/文件操作模块
import os
from os import path
# 导入图像读写模块
import imageio
# 导入科学计算库
import numpy as np
# 导入进度条显示模块
from tqdm import tqdm
import torch
# 导入torchmetrics\PIPS指标库
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lpips

# 导入项目自定义配置模块
from config import get_config
# 导入数据集加载模块
from datasets import get_dataset
# 导入MipNeRF模型模块
from model import MipNeRF
# 导入工具函数（图像格式转换）
from pose_utils import to8b, visualize_depth
# 导入光线数据处理工具函数
from ray_utils import namedtuple_map


def render_flat_rays(model, rays, chunks=8192):
    """
    分批渲染展平光线束，每批结果立即转回CPU，避免显存峰值。
    Args:
        model: 加载好权重的MipNeRF模型
        rays: 展平的光线束（namedtuple格式，包含origins/directions等）
        chunks: 每批渲染的光线数量（控制显存占用）
    Returns:
        tuple(torch.Tensor, torch.Tensor, torch.Tensor):
            pred_rgb[num_rays, 3], pred_dist[num_rays], pred_acc[num_rays]

    """
    num_rays = rays.origins.shape[0]
    rgb_chunks, dist_chunks, acc_chunks = [], [], []
    with torch.no_grad():
        for i in range(0, num_rays, chunks):
            chunk_rays = namedtuple_map(lambda r: r[i:i + chunks].to(model.device), rays)
            rgb, distance, acc = model(chunk_rays)
            rgb_chunks.append(rgb[-1].cpu())
            dist_chunks.append(distance[-1].cpu())
            acc_chunks.append(acc[-1].cpu())
    return torch.cat(rgb_chunks, dim=0), torch.cat(dist_chunks, dim=0), torch.cat(acc_chunks, dim=0)


def _resolve_image_shapes(dataset):
    """Return list of (h, w) for each image in the dataset split."""
    heights = dataset.h
    widths = dataset.w

    if np.isscalar(heights) and np.isscalar(widths):
        return [(int(heights), int(widths)) for _ in range(dataset.n_poses)]

    heights = np.asarray(heights).astype(int).tolist()
    widths = np.asarray(widths).astype(int).tolist()
    return list(zip(heights, widths))


def evaluate_test_set(config):
    # 设备配置：GPU仅用于模型推理，CPU处理所有非推理操作
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    cpu_device = torch.device("cpu")  # 统一CPU设备标识
    print(f"[INFO] 使用设备: {device} (推理) | {cpu_device} (评估/保存)")

    dataset = get_dataset(
        config.dataset_name,
        config.base_dir,
        split="test",
        factor=config.factor,
        device=device,
    )

    config.white_bkgd = dataset.white_bkgd

    model = MipNeRF(
        use_viewdirs=config.use_viewdirs,
        randomized=False,  # 评估时关闭随机采样
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
    # 加载模型权重，自动映射到指定设备
    state_dict = torch.load(config.model_weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()# 评估模式

    image_shapes = _resolve_image_shapes(dataset)
    # ===== 测试集（非深度和法线图部分）进行 1/8 采样，减小显存压力 =====
    test_skip = getattr(config, "test_skip", 8)  # 优先从配置读取，默认8
    # 隔test_skip取1，生成采样后的图像索引（如[0,8,16,...]）
    sampled_indices = list(range(0, len(image_shapes), test_skip))
    print(f"[INFO] 测试图像数量: {len(sampled_indices)}")

    # 预计算每张图在展平ray/pixel中的区间，支持按sampled_indices精确索引
    flat_offsets = []
    start = 0
    for h, w in image_shapes:
        end = start + h * w
        flat_offsets.append((start, end, h, w))
        start = end

    # 初始化评估指标：固定在CPU上，避免设备冲突
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(cpu_device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0, channel=3).to(cpu_device)
    lpips_metric = lpips.LPIPS(net="alex").to(cpu_device)

    save_dir = path.join(config.log_dir, "eval_results")
    pred_dir = path.join(save_dir, "pred")  # 预测图目录
    gt_dir = path.join(save_dir, "gt")  # GT图目录，真实原图片ground true
    depth_dir = path.join(save_dir, "depth")  # 深度可视化目录
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    # 初始化指标列表
    psnr_values, ssim_values, lpips_values = [], [], []

    for image_idx, dataset_img_idx in enumerate(tqdm(sampled_indices, desc="Evaluating")):
        start, end, h, w = flat_offsets[dataset_img_idx]
        rays = namedtuple_map(lambda r: r[start:end], dataset.rays)
        target_rgb = dataset.images[start:end].reshape(h, w, 3).float().cpu()
        pred_rgb_flat, pred_dist_flat, pred_acc_flat = render_flat_rays(model, rays, chunks=config.chunks)
        pred_rgb = pred_rgb_flat.reshape(h, w, 3)
        pred_dist = pred_dist_flat.reshape(h, w)
        pred_acc = pred_acc_flat.reshape(h, w)

        pred_rgb = torch.clamp(pred_rgb, 0.0, 1.0)
        target_rgb = torch.clamp(target_rgb, 0.0, 1.0)

        pred_nchw = pred_rgb.permute(2, 0, 1).unsqueeze(0)
        target_nchw = target_rgb.permute(2, 0, 1).unsqueeze(0)

        psnr_val = psnr_metric(pred_nchw, target_nchw).item()
        ssim_val = ssim_metric(pred_nchw, target_nchw).item()
        lpips_val = lpips_metric(pred_nchw * 2 - 1, target_nchw * 2 - 1).mean().item()

        psnr_values.append(psnr_val)
        ssim_values.append(ssim_val)
        lpips_values.append(lpips_val)

        # 8.8 保存当前图像的预测图和GT图（核心修改：每轮都保存）
        img_name = f"{image_idx:03d}"  # 统一命名格式：000.png, 001.png...
        # 保存预测图
        pred_img = to8b(pred_rgb.numpy())  # 转换为8位图像（0-255）
        imageio.imwrite(path.join(pred_dir, f"{img_name}_pred.png"), pred_img)
        # 保存GT图
        gt_img = to8b(target_rgb.numpy())
        imageio.imwrite(path.join(gt_dir, f"{img_name}_gt.png"), gt_img)
        # 保存深度图（颜色可视化）
        depth_vis = to8b(visualize_depth(pred_dist.numpy(), pred_acc.numpy(), dataset.near, dataset.far))
        imageio.imwrite(path.join(depth_dir, f"{img_name}_depth.png"), depth_vis)

        # 8.9 打印单张图像的指标
        print(f"\n[Image {image_idx:03d}] PSNR={psnr_val:.4f} SSIM={ssim_val:.4f} LPIPS={lpips_val:.4f}")

    print("\n[INFO] ===== Test Set Metrics =====")
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_lpips = np.mean(lpips_values)
    print(f"[INFO] Mean PSNR : {avg_psnr:.4f}")
    print(f"[INFO] Mean SSIM : {avg_ssim:.4f}")
    print(f"[INFO] Mean LPIPS: {avg_lpips:.4f}")

    # 保存指标到文件（新增：参考NeRF评估文件，保存详细指标）
    with open(path.join(save_dir, "eval_metrics.txt"), "w") as f:
        f.write(f"Mean PSNR : {avg_psnr:.4f}\n")
        f.write(f"Mean SSIM : {avg_ssim:.4f}\n")
        f.write(f"Mean LPIPS: {avg_lpips:.4f}\n\n")
        f.write("=== Per-Image Metrics ===\n")
        for idx, (p, s, l) in enumerate(zip(psnr_values, ssim_values, lpips_values)):
            f.write(f"Image {idx:03d}: PSNR={p:.4f}, SSIM={s:.4f}, LPIPS={l:.4f}\n")

    print(f"[INFO] 评估完成！所有结果已保存到 {save_dir}")


if __name__ == "__main__":
    config = get_config()
    evaluate_test_set(config)