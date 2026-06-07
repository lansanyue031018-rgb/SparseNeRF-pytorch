import cv2
import torch
import numpy as np
import os
# 设置缓存路径（指向你手动创建的torch_cache），将现在好的别人已经训练的权重模型存在此处
os.environ['TORCH_HOME'] = r'E:\PycharmProject\mipnerf-pytorch-main\torch_cache'
# 禁用PyTorch Hub联网检查
os.environ['TORCH_HUB_DISABLE_EXTERNAL_LOAD'] = '1'
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--benchmark', type=str, default="llff")  # 默认LLFF
parser.add_argument('-d', '--dataset_id', type=str, default="horns")  # 场景名，如fern/leaves
parser.add_argument('-r', '--root_path', type=str, default="data/nerf_llff_data")  # 数据集根目录，如data/nerf_llff_data
args = parser.parse_args()

# ================== 配置部分 ==================
# 模型选择：DPT_Hybrid 平衡精度和速度，推荐LLFF使用
model_type = "DPT_Hybrid"  # MiDaS v3 - Hybrid
# model_type = "MiDaS_small"  # 若显存不足可用这个

# 加载MiDaS模型
#midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas = torch.hub.load(
    repo_or_dir=r'E:\PycharmProject\mipnerf-pytorch-main\torch_cache\hub\intel-isl_MiDaS_master',
    model=model_type,
    source='local'  # 强制本地加载，不联网
)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# ========== 手动加载权重 ==========
weight_path = r'E:\PycharmProject\mipnerf-pytorch-main\torch_cache\hub\checkpoints\dpt_hybrid-midas-501f0c75.pt'
state_dict = torch.load(weight_path, map_location=device)
midas.load_state_dict(state_dict)

midas.to(device)
midas.eval()

# 加载图像预处理transform
#midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
# 本地加载transforms
midas_transforms = torch.hub.load(
    repo_or_dir=r'E:\PycharmProject\mipnerf-pytorch-main\torch_cache\hub\intel-isl_MiDaS_master',
    model="transforms",
    source='local'
)
if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

# ================== 路径处理（适配LLFF） ==================
# 规范化根路径
root_path = args.root_path.rstrip('/') + '/'

if args.benchmark == "DTU":
    # DTU数据集逻辑（保持原样）
    root_path = root_path + args.dataset_id + '/*3_r5000*'
    output_path = os.path.join('depth_midas_temp_DPT_Hybrid', args.benchmark, args.dataset_id)
else:
    # LLFF数据集逻辑：直接保存到场景目录下的 depth_maps 文件夹
    scene_dir = os.path.join(root_path, args.dataset_id)
    output_path = os.path.join(scene_dir, 'depth_maps')  # 和训练代码预期的路径一致
    # 读取 images_4 或 images_8 文件夹（根据你的factor设置）
    # 优先找 images_4，没有则找 images_8，再没有找 images
    img_folder = None
    for candidate in ['images_4', 'images_8', 'images']:
        candidate_path = os.path.join(scene_dir, candidate)
        if os.path.exists(candidate_path):
            img_folder = candidate_path
            break
    if img_folder is None:
        raise ValueError(f"找不到图像文件夹！请检查 {scene_dir} 下是否有 images_4/images_8/images 文件夹")
    root_path = os.path.join(img_folder, '*png')

# 创建输出目录
os.makedirs(output_path, exist_ok=True)

# 获取所有图像路径
image_paths = sorted(glob.glob(root_path))
print(f'找到 {len(image_paths)} 张图像，保存深度图到：{output_path}')

# ================== 深度图生成与保存（核心修改） ==================
downsampling = 1  # 不需要下采样，训练代码会自己处理

for k, filename in enumerate(image_paths):
    print(f'[{k + 1}/{len(image_paths)}] 处理：{filename}')

    # 1. 读取图像
    img = cv2.imread(filename)
    if img is None:
        print(f"警告：无法读取图像 {filename}，跳过")
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # 2. MiDaS推理
    input_batch = transform(img).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        # 插值回原始分辨率
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(h // downsampling, w // downsampling),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    # 3. 转换为numpy数组
    output = prediction.cpu().numpy()

    # 4. 保存深度图（替代 utils.io.write_depth）
    # 生成文件名：depth_xxx.png
    base_name = os.path.basename(filename)
    depth_name = 'depth_' + base_name
    output_file = os.path.join(output_path, depth_name)

    # ================== 关键：用OpenCV保存16位PNG ==================
    # MiDaS输出是相对深度，我们直接保存为16位单通道PNG
    # 训练代码用 cv2.IMREAD_ANYDEPTH 读取，完美兼容
    # 注意：将深度值缩放到16位范围（0-65535），保留精度
    output_normalized = (output - output.min()) / (output.max() - output.min() + 1e-8)  # 归一化到0-1
    output_16bit = (output_normalized * 65535).astype(np.uint16)  # 转16位

    # 保存！
    cv2.imwrite(output_file, output_16bit)
    print(f'  已保存深度图：{output_file}')

print('所有深度图生成完成！')