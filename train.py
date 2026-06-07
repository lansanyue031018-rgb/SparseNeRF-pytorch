import os.path
import shutil
from os import path
import signal
import sys

import lpips
import numpy as np
import torch
import torch.optim as optim
import torch.utils.tensorboard as tb
from torchmetrics import StructuralSimilarityIndexMeasure
from tqdm import tqdm

from config import get_config
from datasets import cycle, get_dataloader, get_dataset
from loss import NeRFLoss, mse_to_psnr
from model import MipNeRF
from ray_utils import namedtuple_map
from scheduler import MipLRDecay

# ===== 全局变量+紧急保存函数 =====
last_step = 0
model_ref = None
optimizer_ref = None
config_ref = None
def emergency_save(signal_num, frame):
    global last_step, model_ref, optimizer_ref, config_ref
    if last_step > 0 and model_ref is not None and optimizer_ref is not None:
        # 记录最后步数
        step_record_path = path.join(config_ref.log_dir, "last_step.txt")
        with open(step_record_path, "w") as f:
            f.write(str(last_step))
        # 更新并覆盖到最新版模型
        torch.save(model_ref.state_dict(), path.join(config_ref.log_dir, "model.pt"))
        torch.save(optimizer_ref.state_dict(), path.join(config_ref.log_dir, "optim.pt"))

        print(f"\n===== 紧急保存步数 {last_step} 的模型，已覆盖最新model.pt和optim.pt文件 =====")
    sys.exit(0)
# 注册Ctrl+C信号捕获
signal.signal(signal.SIGINT, emergency_save)
# ======================================

# ===== 分批渲染函数(防止一次性处理整图光线) =====
def render_flat_rays(model, rays, chunks=8192):
    """分批渲染光线，每批结果立即转回CPU，降低GPU显存占用"""
    num_rays = rays.origins.shape[0]
    rgb_chunks = []
    with torch.no_grad():
        for i in range(0, num_rays, chunks):
            chunk_rays = namedtuple_map(lambda r: r[i:i + chunks].to(model.device), rays)
            rgb, _, _ = model(chunk_rays)
            rgb_chunks.append(rgb[-1].cpu())  # 立即转回CPU，释放GPU显存
    return torch.cat(rgb_chunks, dim=0)
# ==============================================

class FullImageEvalSampler:
    def __init__(self, dataset):
        self.dataset = dataset
        self.image_shapes = self._resolve_image_shapes(dataset)
        self.image_index = 0
        self.flat_offsets = []
        offset = 0
        for h, w in self.image_shapes:
            self.flat_offsets.append((offset, offset + h * w, h, w))
            offset += h * w

    @staticmethod
    def _resolve_image_shapes(dataset):
        heights = dataset.h
        widths = dataset.w
        if np.isscalar(heights) and np.isscalar(widths):
            return [(int(heights), int(widths)) for _ in range(dataset.n_poses)]
        heights = np.asarray(heights).astype(int).tolist()
        widths = np.asarray(widths).astype(int).tolist()
        return list(zip(heights, widths))

    def next(self):
        # 取出当前图的区间
        start, end, h, w = self.flat_offsets[self.image_index]

        rays = namedtuple_map(lambda r: r[start:start + h * w], self.dataset.rays)
        pixels = self.dataset.images[start:start + h * w]
        self.image_index = (self.image_index + 1) % len(self.flat_offsets)
        return rays, pixels, h, w

def train_model(config):
    # ============续训相关================
    global last_step, model_ref, optimizer_ref, config_ref
    config_ref = config  # 绑定全局配置
    step_record_path = path.join(config.log_dir, "last_step.txt")  # 步数记录文件
    # ======================================
    model_save_path = path.join(config.log_dir, "model.pt")
    optimizer_save_path = path.join(config.log_dir, "optim.pt")

    # === 获取训练数据集实例，设置batch_size ===
    train_dataset = get_dataset(
        dataset_name=config.dataset_name,
        base_dir=config.base_dir,
        split="train",
        factor=config.factor,
        device=config.device,
        use_sparse=config.use_sparse,
        sparse_json_name=config.sparse_json_name,
        sparse_ratio=config.sparse_ratio,
        corner_nearest_pair_ratio=config.corner_nearest_pair_ratio,
        edge_rgb_only_ratio=config.edge_rgb_only_ratio,
    )  # 给LLFF数据集设置batch_size，用于__getitem__生成有序索引
    if config.dataset_name == "llff":
        train_dataset.batch_size = config.batch_size

    data = iter(cycle(
        get_dataloader(dataset_name=config.dataset_name, base_dir=config.base_dir, split="train", factor=config.factor,
                       batch_size=config.batch_size, shuffle=True, device=config.device, use_sparse=config.use_sparse,
                       sparse_json_name=config.sparse_json_name, sparse_ratio=config.sparse_ratio,
                       corner_nearest_pair_ratio=config.corner_nearest_pair_ratio,
                       edge_rgb_only_ratio=config.edge_rgb_only_ratio)))
    eval_sampler = None
    eval_metrics = None
    if config.do_eval:
        eval_dataset = get_dataset(config.dataset_name, config.base_dir, split="test", factor=config.factor, device=config.device, use_sparse=False, sparse_json_name=None, sparse_ratio=None)
        eval_sampler = FullImageEvalSampler(eval_dataset)
        cpu_device = torch.device("cpu")
        eval_metrics = {
            "ssim": StructuralSimilarityIndexMeasure(data_range=1.0, channel=3).to(cpu_device),
            "lpips": lpips.LPIPS(net="alex").to(cpu_device),
        }

    model = MipNeRF(
        use_viewdirs=config.use_viewdirs,
        randomized=config.randomized,
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
        device=config.device,
    )
    optimizer = optim.AdamW(model.parameters(), lr=config.lr_init, weight_decay=config.weight_decay)
    # ============续训相关================
    model_ref = model  #绑定全局模型
    optimizer_ref = optimizer  #绑定全局优化器
    start_step = 0
    if config.continue_training:
        # 优先读取步数记录文件
        if path.exists(step_record_path):
            with open(step_record_path, "r") as f:
                start_step = int(f.read().strip())
            print(f"续训起始步数：{start_step}")
        # 加载模型
        if path.exists(model_save_path):
            model.load_state_dict(torch.load(model_save_path))
        if path.exists(optimizer_save_path):
            optimizer.load_state_dict(torch.load(optimizer_save_path))
    else:
        # 全新训练：清空日志 + 删除旧步数记录
        shutil.rmtree(path.join(config.log_dir, 'train'), ignore_errors=True)
        if path.exists(step_record_path):
            os.remove(step_record_path)
    # ======================================
    # 调度器设置
    scheduler = MipLRDecay(optimizer, lr_init=config.lr_init, lr_final=config.lr_final, max_steps=config.max_steps, lr_delay_steps=config.lr_delay_steps, lr_delay_mult=config.lr_delay_mult)
    # ========== 新增：续训时同步调度器步数 ==========
    if config.continue_training and start_step > 0:
        # 让调度器走start_step步，同步学习率
        for _ in range(start_step):
            scheduler.step()
        print(f"调度器同步至步数 {start_step}，当前学习率：{scheduler.get_last_lr()[-1]}")
    # ==============================================

    loss_func = NeRFLoss(coarse_weight_decay=config.coarse_weight_decay,depth_rank_weight=getattr(config, "depth_rank_weight", 0.3))
    os.makedirs(config.log_dir, exist_ok=True)
    logger = tb.SummaryWriter(path.join(config.log_dir, 'train'), flush_secs=1)

    # ===== 续训相关：循环从之前得到的start_step开始，并不断存储更新到last_step =====
    global last_step
    last_step = start_step
    for step in tqdm(range(start_step + 1, config.max_steps)):
        last_step = step
        # === 修改：适配数据集返回的三元组(ray, pixel, depth_dpt) ===
        batch_data = next(data)
        depth_rank_count = None
        if len(batch_data) == 3:
            rays, pixels, extra = batch_data  # llff下extra可能是排序样本数量
            if config.dataset_name == "llff" and torch.is_tensor(extra):
                depth_rank_count = int(extra.item())
        else:
            rays, pixels = batch_data
        pixels = pixels.to(config.device)

        # === 关键修复：接入model返回的distances ===
        comp_rgb, distances, _ = model(rays)
        # 复用同批采样光线，取细采样的深度，用于深度排序损失
        fine_depth = distances[-1]
        if depth_rank_count is not None:
            fine_depth = fine_depth[:depth_rank_count]

        # === Compute loss   ===
        loss_val, psnr ,satisfiedratio= loss_func(
            comp_rgb,
            pixels,
            rays.lossmult.to(config.device),
            depth_render=fine_depth if config.dataset_name == "llff" else None
        )

        # update model weights.
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        scheduler.step()

        psnr = psnr.detach().cpu().numpy()
        logger.add_scalar('train/satisfiedratio', float(satisfiedratio), global_step=step)
        logger.add_scalar('train/loss', float(loss_val.detach().cpu().numpy()), global_step=step)
        logger.add_scalar('train/coarse_psnr', float(np.mean(psnr[:-1])), global_step=step)
        logger.add_scalar('train/fine_psnr', float(psnr[-1]), global_step=step)
        logger.add_scalar('train/avg_psnr', float(np.mean(psnr)), global_step=step)
        logger.add_scalar('train/lr', float(scheduler.get_last_lr()[-1]), global_step=step)

        if step % config.save_every == 0 and step != 0 and step != 1:
            if eval_sampler is not None:
                del rays
                del pixels
                eval_result = eval_model(config, model, eval_sampler, eval_metrics)
                eval_psnr = eval_result["psnr"].detach().cpu().numpy()
                logger.add_scalar('eval/avg_psnr', float(np.mean(eval_psnr)), global_step=step)
                logger.add_scalar('eval/ssim', float(eval_result["ssim"]), global_step=step)
                logger.add_scalar('eval/lpips', float(eval_result["lpips"]), global_step=step)

        if step % config.save_every == 0 and step != 0 and step != 1:
                # 2. 保存带步数后缀的模型和优化器
                model_step_path = path.join(config.log_dir, f"model_{last_step}.pt")
                optim_step_path = path.join(config.log_dir, f"optim_{last_step}.pt")
                torch.save(model.state_dict(), model_step_path)
                torch.save(optimizer.state_dict(), optim_step_path)
                print(f"\n已保存步数 {last_step} 的模型：{model_step_path}")

                # 3.记录当前步数到文件
                with open(step_record_path, "w") as f:
                    f.write(str(step))

    torch.save(model.state_dict(), model_save_path)
    torch.save(optimizer.state_dict(), optimizer_save_path)
    # 训练完成后记录最终步数到文件
    with open(step_record_path, "w") as f:
        f.write(str(config.max_steps))

def eval_model(config, model, sampler, metrics):
    model.eval() # 评估模式，会自动关闭随机采样设置
    rays, pixels, h, w = sampler.next()
    # ===== 分批渲染 =====
    # 不再将整束光线转GPU，而是在render_flat_rays中分批转
    pred_rgb_flat = render_flat_rays(model, rays, chunks=getattr(config, "chunks", 8192))
    # =====================================================

    # ===== 所有张量操作在CPU完成 =====
    pred_rgb = torch.clamp(pred_rgb_flat.reshape(h, w, 3), 0.0, 1.0)
    target_rgb = torch.clamp(pixels.reshape(h, w, 3).float().cpu(), 0.0, 1.0)
    # =====================================================

    # 计算PSNR
    mse = torch.mean((pred_rgb - target_rgb) ** 2)
    psnr = torch.tensor([mse_to_psnr(mse)])

    # 指标计算（CPU上计算）
    pred_nchw = pred_rgb.permute(2, 0, 1).unsqueeze(0)
    target_nchw = target_rgb.permute(2, 0, 1).unsqueeze(0)
    ssim_val = metrics["ssim"](pred_nchw, target_nchw).item()
    lpips_val = metrics["lpips"](pred_nchw * 2 - 1, target_nchw * 2 - 1).mean().item()

    model.train()
    return {"psnr": psnr, "ssim": ssim_val, "lpips": lpips_val}


if __name__ == "__main__":
    config = get_config()
    train_model(config)
