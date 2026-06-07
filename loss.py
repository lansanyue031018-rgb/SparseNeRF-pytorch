
import torch

def mse_to_psnr(mse):
    """将MSE损失转换为PSNR（峰值信噪比），加1e-8避免log(0)"""
    return -10.0 * torch.log10(mse + 1e-8)


def depth_ranking_loss(distance_mean, margin_pair=1e-4, neighbor_pair_weight=0.1):
    """
    4-ray深度排序损失
    输入要求：distance_mean顺序严格为[far, near, far, near, ...]。
    约束：NeRF渲染深度满足 far_depth > near_depth。
    """
    # 4-ray分组（主点对+邻近点对）优先；否则回退到普通2-ray分组。
    depth_groups = distance_mean.reshape(-1, 4)  # [far, near, far_nbr, near_nbr]
    main_loss = torch.maximum(
        depth_groups[:, 1] - depth_groups[:, 0] + margin_pair,
        torch.zeros(1, device=distance_mean.device, dtype=distance_mean.dtype)
    )
    neighbor_loss1 = torch.maximum(
        torch.abs(depth_groups[:, 0] - depth_groups[:, 2]) - margin_pair,
        torch.zeros(1, device=distance_mean.device, dtype=distance_mean.dtype)
    )
    neighbor_loss2 = torch.maximum(
        torch.abs(depth_groups[:, 1] - depth_groups[:, 3]) - margin_pair,
        torch.zeros(1, device=distance_mean.device, dtype=distance_mean.dtype)
    )
    loss = (main_loss + neighbor_pair_weight * (neighbor_loss1+neighbor_loss2)).mean()
    # ===================== 打印 0 所占比例 =====================
    # 只统计主排序满足率
    satisfy_main = (main_loss <= 1e-12).float().mean()
    return loss, satisfy_main

class NeRFLoss(torch.nn.modules.loss._Loss):
    def __init__(self, coarse_weight_decay=0.1, depth_rank_weight=0.5):
        super(NeRFLoss, self).__init__()
        self.coarse_weight_decay = coarse_weight_decay
        self.depth_rank_weight = depth_rank_weight  # 深度排序损失权重

    def forward(self, input, target, mask, depth_render=None):
        """
        前向计算：RGB损失 + 深度排序损失（可选）
        Args:
            input: NeRF输出的RGB（粗+细采样），list of [B,H,W,3]
            target: 真实RGB图，[B,H,W,3]
            mask: 掩码，[B,H,W]
            depth_render: NeRF渲染深度图，[B,H,W]（不传则不计算深度损失）
            depth_dpt: DPT参考深度图，[B,H,W]
        Returns:
            total_loss: 总损失（RGB损失 + 深度排序损失）
            psnrs: 各层PSNR（仅由RGB损失计算）
        """
        # 1. 原有RGB损失计算（保留所有逻辑）
        rgb_losses, psnrs = [], []
        for rgb in input:
            mse = (mask * ((rgb - target[..., :3]) ** 2)).sum() / mask.sum()
            rgb_losses.append(mse)
            with torch.no_grad():
                psnrs.append(mse_to_psnr(mse))
        rgb_losses = torch.stack(rgb_losses)
        total_rgb_loss = self.coarse_weight_decay * rgb_losses[:-1].sum() + rgb_losses[-1]

        # 2. 深度排序损失（仅当传入渲染深度时计算）
        total_depth_loss = torch.zeros(1, device=total_rgb_loss.device, dtype=total_rgb_loss.dtype).squeeze(0)
        zero_ratio=0
        if depth_render is not None:
            total_depth_loss, zero_ratio = depth_ranking_loss(depth_render)
        # 3. 总损失 = RGB损失 + 权重×深度排序损失
        total_loss = total_rgb_loss + self.depth_rank_weight * total_depth_loss

        return total_loss, torch.Tensor(psnrs), zero_ratio
