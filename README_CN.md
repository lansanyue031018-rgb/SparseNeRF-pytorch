# SparseNeRF-PyTorch / Dep-NeRF

[English](README.md)

这是一个受
[SparseNeRF](https://github.com/Wanggcong/SparseNeRF)
启发、面向稀疏视图新视图合成任务的非官方 PyTorch 研究实现。本仓库将该改进方法命名为
**Dep-NeRF**。项目保留 Mip-NeRF 主干，在训练阶段加入局部相对深度排序约束、
边缘 RGB 副采样和可配置的数据集稀疏化功能。

> [!IMPORTANT]
> 本项目不是 SparseNeRF 官方实现，也不是逐行移植版本。与原始 SparseNeRF
> 不同，本实现**只使用深度排序损失，不包含深度连续性损失**。项目主要用于科研复现、
> 稀疏视图新视图合成实验和相关研究。

## 项目特点

- 基于 PyTorch 和 Mip-NeRF 多尺度主干实现。
- 使用单目深度先验提供局部相对远近关系。
- 使用 `[far, near, far_neighbor, near_neighbor]` 四射线排序分组。
- 边缘和四角副采样射线只参与 RGB 重建，不参与深度排序监督。
- 支持通过配置对 LLFF 和 Blender 风格数据集进行稀疏化。
- 支持 PSNR、SSIM、LPIPS、RGB 渲染图和深度图评估。
- 已在 NVIDIA GeForce RTX 3060 6 GB 显卡上完成实验。

## 方法概述

在稀疏视图条件下，仅使用 RGB 重建损失很难充分约束场景几何。Dep-NeRF
保留 Mip-NeRF 的锥体采样、积分位置编码和体渲染主干，并加入相对深度排序损失：

```text
L_total = L_rgb + lambda_rank * L_rank
```

排序损失不直接回归单目深度图中的绝对数值，而是利用局部区域内较稳定的前后关系，
从而降低单目深度尺度不确定性对训练的影响。

训练采样由两部分组成：

1. **局部排序样本**：同时参与 RGB 重建和深度排序监督。
2. **边缘/四角副采样样本**：提高边界区域训练覆盖率，只参与 RGB 重建。

本方法没有改变 Mip-NeRF 的主要网络结构和前向渲染流程，改动主要集中在
`loss.py`、`datasets.py`、`train.py` 和 `config.py`。

## 与原始 SparseNeRF 的差异

| 对比项 | 本仓库实现 |
| --- | --- |
| 深度学习框架 | PyTorch |
| 主干网络 | Mip-NeRF 风格锥体/圆柱体采样与积分位置编码 |
| 几何正则化 | 局部深度排序损失 |
| 深度连续性损失 | 未实现 |
| 排序采样 | 局部四射线分组和邻域辅助点 |
| 边界补偿 | 边缘/四角 RGB-only 副采样 |
| 数据集稀疏化 | LLFF 稀疏比例和 Blender 稀疏 JSON 文件 |

## 实验结果

以下结果来自统一条件下的对比实验。PSNR、SSIM 越高越好，LPIPS
越低越好。

| 数据集 | 训练视图 | 模型 | PSNR | SSIM | LPIPS |
| --- | ---: | --- | ---: | ---: | ---: |
| LLFF horns | 5 | NeRF | 17.6062 | 0.5417 | 0.5128 |
| LLFF horns | 5 | Mip-NeRF | 19.9849 | 0.6132 | 0.4286 |
| LLFF horns | 5 | Dep-NeRF | **20.7247** | **0.6301** | **0.4164** |
| 自制铁路机车轮对场景 | 6 | NeRF | 16.4132 | 0.5344 | 0.6816 |
| 自制铁路机车轮对场景 | 6 | Mip-NeRF | 18.7422 | 0.6336 | 0.5612 |
| 自制铁路机车轮对场景 | 6 | Dep-NeRF | **19.7217** | **0.6847** | **0.5501** |
| 自制室内东方红机车场景 | 4 | NeRF | 12.3590 | 0.4210 | 0.7088 |
| 自制室内东方红机车场景 | 4 | Mip-NeRF | 16.0914 | 0.5654 | 0.5536 |
| 自制室内东方红机车场景 | 4 | Dep-NeRF | **16.4870** | **0.6042** | **0.5298** |

实验结果表明，该方法主要适合**少视图训练场景**。密集视图实验没有显示出
相对 Mip-NeRF 的稳定提升。当多视图监督已经较充分时，建议减小
`depth_rank_weight` 或关闭深度排序正则项。

## 已测试环境

| 组件 | 版本 |
| --- | --- |
| Python | 3.9.7 |
| PyTorch | 1.11.0 |
| CUDA | 11.3 |
| GPU | NVIDIA GeForce RTX 3060 6 GB |

Python 依赖已记录在 `requirements.txt` 中。其版本对应测试环境；如果使用更新的
CUDA、显卡或操作系统，可能需要调整个别依赖版本。

## 安装

```bash
git clone https://github.com/lansanyue031018-rgb/SparseNeRF-pytorch.git
cd SparseNeRF-pytorch

conda create -n depnerf python=3.9.7
conda activate depnerf

pip install -r requirements.txt
```

对于测试使用的 CUDA 11.3 环境，在安装依赖文件后重新安装对应 CUDA 版本的
PyTorch：

```bash
pip install --force-reinstall \
  torch==1.11.0+cu113 \
  torchvision==0.12.0+cu113 \
  torchaudio==0.11.0 \
  --index-url https://download.pytorch.org/whl/cu113
```

当前 `requirements.txt` 未固定网格提取可能使用的 `plyfile`、`PyMCubes` 和
`open3d` 等可选依赖。

## 数据集准备

### LLFF 目录结构

将 LLFF 数据集放入 `data/nerf_llff_data/`：

```text
data/
  nerf_llff_data/
    horns/
      images/
      images_4/
      poses_bounds.npy
      depth_maps/
        depth_000.png
        depth_001.png
        ...
```

深度图文件名需要与 RGB 图像对应，并增加 `depth_` 前缀。例如：

```text
images_4/000.png
depth_maps/depth_000.png
```

单目相对深度先验图属于数据集准备环节，不在 NeRF 训练过程中在线生成。可以使用
[MiDaS](https://github.com/isl-org/MiDaS)、
[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
或其他相对深度模型自行生成。将每张结果转换为单通道图像，推荐保存为 16 位 PNG，
文件名前增加 `depth_`，然后放入对应场景的 `depth_maps/` 目录。

当前排序采样器采用**逆深度约定：数值越大表示越近，数值越小表示越远**。如果所用模型
输出的是度量深度或远近顺序相反，需要先进行取反或等价转换，再用于训练。

仓库中的辅助脚本 `get_depth_map_for_llff_dtu.py` 使用 MiDaS/DPT。脚本目前保留了
原实验机器的绝对路径。运行前需要将两处 `repo_or_dir` 修改为本机 MiDaS 仓库路径，
并将 `weight_path` 修改为已下载的模型权重路径：

```bash
python get_depth_map_for_llff_dtu.py \
  --benchmark llff \
  --dataset_id horns \
  --root_path data/nerf_llff_data
```

### 数据集稀疏化

- LLFF 使用 `sparse_ratio` 按比例均匀保留训练视图。
- Blender 风格数据集使用 `sparse_json_name`，例如
  `transforms_sparse_8.json`。
- 当前历史版本的 `--use_sparse` 使用了 `action="store_false"`：
  稀疏模式默认开启，传入 `--use_sparse` 反而会关闭稀疏模式。

## 训练

接近稀疏视图对比实验设置的训练命令：

```bash
python train.py \
  --dataset_name llff \
  --scene horns \
  --sparse_ratio 0.1 \
  --depth_rank_weight 0.5 \
  --corner_nearest_pair_ratio 0.1 \
  --edge_rgb_only_ratio 0.05 \
  --batch_size 1024 \
  --max_steps 33000 \
  --save_every 11000 \
  --log_dir log/horns_depnerf
```

LLFF 默认配置会自动设置 `factor=4`、`ray_shape=cylinder`、
`white_bkgd=False` 和 `test_skip=1`。如果希望命令行中的参数覆盖这些默认设置，
需要同时传入 `--override_defaults`。

继续训练：

```bash
python train.py \
  --dataset_name llff \
  --scene horns \
  --log_dir log/horns_depnerf \
  --continue_training
```

### 批量实验（实验性功能）

`run_experiments.py` 用于按顺序连续训练多个数据集或配置。一个训练任务达到预设里程碑
或退出后，脚本会读取 `exp_plan.json` 并自动开始下一个任务；如果检测到已有权重，
也可以自动继续训练。

仓库中保留的默认值来自原 Windows 实验环境，不能直接用于其他机器。使用前需要修改：

- `run_experiments.py` 中 `--plan`、`--python`、`--logs-dir` 的默认绝对路径，
  或在命令行中显式传入。
- `exp_plan.json` 中的 `global.log_dir` 和每个任务的 `cwd`。
- 适用于自己数据集的场景名、稀疏比例、随机种子、重复次数和 `extra_args`。

```bash
python run_experiments.py \
  --plan /path/to/exp_plan.json \
  --python /path/to/python \
  --logs-dir /path/to/scheduler_logs
```

### 主要配置参数

| 参数 | 代码默认值 | 作用 |
| --- | ---: | --- |
| `depth_rank_weight` | `0.5` | 深度排序正则项权重 |
| `corner_nearest_pair_ratio` | `0.0` | 将局部排序框移动到最近四角区域的概率 |
| `edge_rgb_only_ratio` | `0.05` | 每个批次中边缘/四角 RGB-only 射线比例 |
| `sparse_ratio` | `0.2` | LLFF 训练视图保留比例 |
| `sparse_json_name` | `transforms_sparse_32.json` | Blender 稀疏训练划分文件 |
| `batch_size` | `1024` | 每批训练射线数 |
| `max_steps` | `200000` | 最大训练步数 |
| `save_every` | `33000` | 权重保存和自动评估间隔 |
| `chunks` | `4096` | 分块渲染时每块射线数 |

当前 `loss.py` 中的 `margin_pair` 和 `neighbor_pair_weight` 还是函数默认参数，
尚未作为命令行参数暴露。对比实验使用了 `margin_pair=1e-4`。复现实验前应核对
当前实现中的实际参数，不应只根据配置表推断。

## 评估

```bash
python evaluate.py \
  --dataset_name llff \
  --scene horns \
  --model_weight_path log/horns_depnerf/model.pt \
  --log_dir log/horns_depnerf
```

评估结果保存在：

```text
log/horns_depnerf/eval_results/
  pred/
  gt/
  eval_metrics.txt
```

## 渲染

```bash
python visualize.py \
  --dataset_name llff \
  --scene horns \
  --model_weight_path log/horns_depnerf/model.pt \
  --log_dir log/horns_depnerf \
  --visualize_depth
```

渲染结果和视频会写入指定的日志目录。视频导出需要 `imageio` 可调用的 FFmpeg。

## 项目结构

```text
config.py                       # 命令行配置
datasets.py                     # 数据读取、稀疏化和训练采样
loss.py                         # RGB 损失和局部深度排序损失
model.py                        # Mip-NeRF 模型
train.py                        # 训练、日志和权重保存
evaluate.py                     # PSNR、SSIM、LPIPS 评估
visualize.py                    # RGB 和深度视频渲染
get_depth_map_for_llff_dtu.py   # MiDaS/DPT 深度先验生成
run_experiments.py              # 顺序批量实验调度
extract_mesh.py                 # 可选网格提取
```

## 已知限制

- 本方法面向稀疏视图，不保证在密集视图训练下优于 Mip-NeRF。
- 结果依赖单目深度质量、COLMAP 位姿精度、视角覆盖范围和场景本身。
- 反光、透明、弱纹理和重复纹理区域可能产生错误排序先验。
- 当前仓库未包含自制数据集和预训练权重。
- MiDaS 辅助脚本和实验性批量训练脚本仍包含示例 Windows 绝对路径，必须按实际机器
  修改或通过命令行覆盖。
- 部分历史布尔命令行参数使用 `store_false`，行为不够直观。

## 后续计划

- 补充跨平台 Conda 环境或容器环境定义。
- 移除深度生成与批量实验脚本中的本机绝对路径。
- 发布可复现配置、预训练权重和可视化对比结果。
- 补充深度排序和边缘采样机制的消融实验。
- 统一稀疏模式命令行语义，并开放全部排序损失超参数。

## 致谢

本项目参考和使用了以下研究工作中的思想：

- [SparseNeRF: Distilling Depth Ranking for Few-shot Novel View Synthesis](https://github.com/Wanggcong/SparseNeRF)
- [Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields](https://jonbarron.info/mipnerf/)
- [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://www.matthewtancik.com/nerf)
- [MiDaS](https://github.com/isl-org/MiDaS)
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)

## 引用

如果本仓库对你的研究有帮助，请引用原始 SparseNeRF 和 Mip-NeRF 论文：

```bibtex
@inproceedings{wang2023sparsenerf,
  title     = {SparseNeRF: Distilling Depth Ranking for Few-shot Novel View Synthesis},
  author    = {Wang, Guangcong and Chen, Zhaoxi and Loy, Chen Change and Liu, Ziwei},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year      = {2023}
}

@inproceedings{barron2021mipnerf,
  title     = {Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields},
  author    = {Barron, Jonathan T. and Mildenhall, Ben and Tancik, Matthew and
               Hedman, Peter and Martin-Brualla, Ricardo and Srinivasan, Pratul P.},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year      = {2021}
}
```

## 许可证

本仓库中的原创贡献采用 [MIT License](LICENSE) 发布。

第三方代码、上游衍生代码、模型、权重、数据集和素材仍受各自许可证约束。特别是原始
SparseNeRF 仓库采用带有非商业限制的
[S-Lab License 1.0](https://github.com/Wanggcong/SparseNeRF/blob/main/LICENSE)；
本仓库的 MIT 许可证不会覆盖或取消这些限制。
