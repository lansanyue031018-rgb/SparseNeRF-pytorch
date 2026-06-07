import os
from os import path
import json
import numpy as np
import cv2
from PIL import Image
import torch
from ray_utils import Rays, convert_to_ndc, namedtuple_map
from pose_utils import normalize, look_at, poses_avg, recenter_poses, to_float, generate_spiral_cam_to_world, generate_spherical_cam_to_world, flatten
from torch.utils.data import Dataset, DataLoader


# 深度采样函数：保证输出的索引按 [far, near, far, near,...] 成对排列
def sample_depth_ranking_ray_indices(
    depth_image,
    batch_size,
    llff_scan="",
    corner_nearest_pair_ratio=0.3,
):
    """Sample ray indices in 4-ray groups [far, near, far_nbr, near_nbr]."""
    if isinstance(depth_image, torch.Tensor):
        depth_image = depth_image.detach().cpu().numpy()
    depth_image = np.asarray(depth_image)
    h, w = depth_image.shape
    ray_indices = []
    folds = 4  # 固定4-ray分组

    # 场景适配的采样框大小
    if llff_scan == "leaves":
        box_h, box_w = 180, 180
    elif llff_scan == "horns":
        box_h, box_w = 60, 60
    elif llff_scan in ["room", "fortress"]:
        box_h, box_w = h, w
    else:
        box_h, box_w = 30, 30

    box_h = min(box_h, h)
    box_w = min(box_w, w)

    n_groups = batch_size // folds
    max_trials = 20
    box_nested_h, box_nested_w = 3, 3
    top_percent = 0.3
    eps = 1e-6

    corner_nearest_pair_ratio = float(np.clip(corner_nearest_pair_ratio, 0.0, 1.0))
    corner_tops = np.array([0, 0, h - 170, h - 170], dtype=np.int32)
    corner_lefts = np.array([0, w - 320, 0, w - 320], dtype=np.int32)

    def random_neighbor(ph, pw, label):
        h_min = max(ph - box_nested_h, 0)
        h_max = min(ph + box_nested_h, h - 1)
        w_min = max(pw - box_nested_w, 0)
        w_max = min(pw + box_nested_w, w - 1)

        patch = depth_image[h_min:h_max + 1, w_min:w_max + 1]
        vec = patch.reshape(-1).astype(np.float32)
        if vec.size == 0:
            return ph, pw

        sorted_ind = np.argsort(np.abs(vec - float(label)))
        k = max(1, int(vec.size * top_percent))
        pick = sorted_ind[np.random.randint(0, k)]
        patch_w = patch.shape[1]
        nh = h_min + int(pick // patch_w)
        nw = w_min + int(pick % patch_w)
        return nh, nw

    for _ in range(n_groups):
        picked = False

        for _trial in range(max_trials):
            box_h1 = box_h
            box_w1 = box_w
            top = np.random.randint(0, h - box_h + 1)
            left = np.random.randint(0, w - box_w + 1)

            # 按比例将部分点对切换到“最近角同尺寸区域”采样
            if np.random.rand() < corner_nearest_pair_ratio:
                d2 = (corner_tops - top) ** 2 + (corner_lefts - left) ** 2
                nearest_corner = int(np.argmin(d2))
                top = int(corner_tops[nearest_corner])
                left = int(corner_lefts[nearest_corner])
                box_h1 = 170
                box_w1 = 320

            h0 = top + np.random.randint(0, box_h1)
            w0 = left + np.random.randint(0, box_w1)
            h1 = top + np.random.randint(0, box_h1)
            w1 = left + np.random.randint(0, box_w1)


            if h0 == h1 and w0 == w1:
                continue

            d0 = float(depth_image[h0, w0])
            d1 = float(depth_image[h1, w1])

            # 过滤无效深度和近似相等深度，避免无意义/噪声排序对
            if d0 <= 0.0 or d1 <= 0.0 or abs(d0 - d1) <= eps:
                continue

            n0h, n0w = random_neighbor(h0, w0, d0)
            n1h, n1w = random_neighbor(h1, w1, d1)

            idx0 = h0 * w + w0
            idx1 = h1 * w + w1
            nidx0 = n0h * w + n0w
            nidx1 = n1h * w + n1w

            # MiDaS/DPT输出是逆深度：值越大越近 => far应是值更小的那个
            if d0 <= d1:
                ray_indices.extend([idx0, idx1, nidx0, nidx1])# [far, near]
            else:

                ray_indices.extend([idx1, idx0, nidx1, nidx0]) # [far, near]

            picked = True
            break

        if not picked:
            # 回退：即便找不到理想pair，也保证不同像素并按逆深度排序
            idx = np.random.choice(h * w, size=4, replace=False)
            d = [float(depth_image[ii // w, ii % w]) for ii in idx]
            pair1 = sorted([0, 1], key=lambda t: d[t])
            pair2 = sorted([2, 3], key=lambda t: d[t])
            ray_indices.extend([int(idx[pair1[0]]), int(idx[pair1[1]]), int(idx[pair2[0]]), int(idx[pair2[1]])])

    ray_indices = np.array(ray_indices[:batch_size], dtype=np.int64).reshape(-1)
    return ray_indices


def sample_edge_only_rgb_ray_indices(depth_image, num_rays, edge_band_ratio=0.1, corner_patch_ratio=0.2):
    """仅从图像边缘/四角采样，用于RGB补样，不参与深度排序监督。"""
    if num_rays <= 0:
        return np.zeros((0,), dtype=np.int64)

    if isinstance(depth_image, torch.Tensor):
        depth_image = depth_image.detach().cpu().numpy()
    h, w = np.asarray(depth_image).shape

    band_h = max(1, int(h * edge_band_ratio))
    band_w = max(1, int(w * edge_band_ratio))
    corner_h = max(1, int(h * corner_patch_ratio))
    corner_w = max(1, int(w * corner_patch_ratio))

    edge_mask = np.zeros((h, w), dtype=bool)
    edge_mask[:band_h, :] = True
    edge_mask[-band_h:, :] = True
    edge_mask[:, :band_w] = True
    edge_mask[:, -band_w:] = True

    corner_mask = np.zeros((h, w), dtype=bool)
    corner_mask[:corner_h, :corner_w] = True
    corner_mask[:corner_h, -corner_w:] = True
    corner_mask[-corner_h:, :corner_w] = True
    corner_mask[-corner_h:, -corner_w:] = True

    candidate_mask = edge_mask | corner_mask
    candidate_indices = np.flatnonzero(candidate_mask.reshape(-1))
    if candidate_indices.size == 0:
        candidate_indices = np.arange(h * w, dtype=np.int64)

    replace = candidate_indices.size < num_rays
    sampled = np.random.choice(candidate_indices, size=num_rays, replace=replace)
    return sampled.astype(np.int64).reshape(-1)

def get_dataset(dataset_name, base_dir, split, factor=4, device=torch.device("cpu"), use_sparse=False, sparse_json_name="transforms_sparse_32.json", sparse_ratio=None,
                corner_nearest_pair_ratio=0.3, edge_rgb_only_ratio=0.25):   # 如果是blender，则传输稀疏参数
    if dataset_name == "blender":
        d = dataset_dict[dataset_name](base_dir, split, factor=factor, device=device, use_sparse=use_sparse, sparse_json_name=sparse_json_name)
    elif dataset_name == "llff":
        d = dataset_dict[dataset_name](
            base_dir,
            split,
            factor=factor,
            device=device,
            use_sparse=use_sparse,
            sparse_ratio=sparse_ratio,
            corner_nearest_pair_ratio=corner_nearest_pair_ratio,
            edge_rgb_only_ratio=edge_rgb_only_ratio,
        )
    else:
        d = dataset_dict[dataset_name](base_dir, split, factor=factor, device=device)
    return d

def get_dataloader(dataset_name, base_dir, split, factor=4, batch_size=None, shuffle=True, device=torch.device("cpu"), use_sparse=False, sparse_json_name="transforms_sparse_32.json", sparse_ratio=None,
                   corner_nearest_pair_ratio=0.3, edge_rgb_only_ratio=0.25):
    # 传递稀疏参数给get_dataset
    d = get_dataset(dataset_name=dataset_name, base_dir=base_dir, split=split, factor=factor, device=device,use_sparse=use_sparse,
                    sparse_json_name=sparse_json_name, sparse_ratio=sparse_ratio,
                    corner_nearest_pair_ratio=corner_nearest_pair_ratio, edge_rgb_only_ratio=edge_rgb_only_ratio)
    collate_fn = None
    if dataset_name == "llff" and split == "train":
        if batch_size is None:
            raise ValueError("LLFF train dataloader requires batch_size for 2-ray depth ranking sampling")
        d.batch_size = batch_size
        batch_size = 1
        collate_fn = lambda batch: batch[0]

    # make the batchsize height*width, so that one "batch" from the dataloader corresponds to one
    # image used to render a video, and don't shuffle dataset
    if split == "render":
        batch_size = d.w * d.h
        shuffle = False
    loader = DataLoader(d, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    loader.h = d.h
    loader.w = d.w
    loader.near = d.near
    loader.far = d.far
    return loader

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

#spherify是渲染视频的旋转方式的调整，如果是Multicam、Blender、LLFF文件夹命名的数据集，还需要从本py对应函数位置修改spherify：False为前后摇晃旋转，True是斜俯视绕物体中心轴旋转
#n_poses是视角数，也就是渲染一次生成的图数，此函数NeRFDataset中修改n_poses即可
class NeRFDataset(Dataset):
    def __init__(self, base_dir, split, spherify=True, near=2, far=6, white_bkgd=False, factor=1, n_poses=60, radius=None, radii=None, h=None, w=None, device=torch.device("cpu")):
        super(Dataset, self).__init__()
        self.base_dir = base_dir
        self.split = split
        self.spherify = spherify
        self.near = near
        self.far = far
        self.white_bkgd = white_bkgd
        self.factor = factor
        self.n_poses = n_poses
        self.n_poses_copy = n_poses
        self.radius = radius
        self.radii = radii
        self.h = h
        self.w = w
        self.device = device
        self.rays = None
        self.images = None
        self.depth_images = None
        self.load()

    def load(self):
        if self.split == "render":
            self.generate_render_rays()
        else:
            self.generate_training_rays()

        self.flatten_to_pytorch()
        print('Done')
        print()

    def generate_training_poses(self):
        """
        Generate training poses, datasets should implement this function to load the proper data from disk.
        Should initialize self.h, self.w, self.focal, self.cam_to_world, and self.images
        """
        raise ValueError('no generate_training_poses(self).')

    def generate_render_poses(self):
        """
        Generate arbitrary poses (views)
        """
        self.focal = 1200
        self.n_poses = self.n_poses_copy
        if self.spherify:
            self.generate_spherical_poses(self.n_poses)
        else:
            self.generate_spiral_poses(self.n_poses)

    def generate_spherical_poses(self, n_poses=120):
        self.poses = generate_spherical_cam_to_world(self.radius, n_poses)
        self.cam_to_world = self.poses[:, :3, :4]

    def generate_spiral_poses(self, n_poses=120):
        self.cam_to_world = generate_spiral_cam_to_world(self.radii, self.focal, n_poses)

    def generate_training_rays(self):
        """
        Generates rays to train mip-NeRF
        """
        print("Loading Training Poses")
        self.generate_training_poses()
        print("Generating rays")
        self.generate_rays()

    def generate_render_rays(self):
        """
        Generates rays used to render a video using a trained mip-NeRF
        """
        print("Generating Render Poses")
        self.generate_render_poses()
        print("Generating rays")
        self.generate_rays()

    def generate_rays(self):
        """Computes rays using a General Pinhole Camera Model
        Assumes self.h, self.w, self.focal, and self.cam_to_world exist
        """
        x, y = np.meshgrid(
            np.arange(self.w, dtype=np.float32),  # X-Axis (columns)
            np.arange(self.h, dtype=np.float32),  # Y-Axis (rows)
            indexing='xy')
        camera_directions = np.stack(
            [(x - self.w * 0.5 + 0.5) / self.focal,
             -(y - self.h * 0.5 + 0.5) / self.focal,
             -np.ones_like(x)],
            axis=-1)
        # Rotate ray directions from camera frame to the world frame
        directions = ((camera_directions[None, ..., None, :] * self.cam_to_world[:, None, None, :3, :3]).sum(axis=-1))  # Translate camera frame's origin to the world frame
        origins = np.broadcast_to(self.cam_to_world[:, None, None, :3, -1], directions.shape)
        viewdirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

        # Distance from each unit-norm direction vector to its x-axis neighbor
        dx = np.sqrt(np.sum((directions[:, :-1, :, :] - directions[:, 1:, :, :]) ** 2, -1))
        dx = np.concatenate([dx, dx[:, -2:-1, :]], 1)

        # Cut the distance in half, and then round it out so that it's
        # halfway between inscribed by / circumscribed about the pixel.
        radii = dx[..., None] * 2 / np.sqrt(12)

        ones = np.ones_like(origins[..., :1])

        self.rays = Rays(
            origins=origins,
            directions=directions,
            viewdirs=viewdirs,
            radii=radii,
            lossmult=ones,
            near=ones * self.near,
            far=ones * self.far)

    def flatten_to_pytorch(self):
        if self.rays is not None:
            self.rays = namedtuple_map(lambda r: torch.tensor(r).float().reshape([-1, r.shape[-1]]), self.rays)
        if self.images is not None:
            self.images = torch.from_numpy(self.images.reshape([-1, 3]))
        # 深度图展平，和rays、images索引完全对齐
        if self.depth_images is not None:
            self.depth_images = torch.from_numpy(self.depth_images.reshape([-1]))

    def ray_to_device(self, rays):
        return namedtuple_map(lambda r: r.to(self.device), rays)

    def __getitem__(self, i):
        ray = namedtuple_map(lambda r: r[i], self.rays)
        if self.split == "render":
            # render rays
            return ray  # Don't put on device, will batch it using config.chunks in mipNeRF.render_image() function
        else:
            # training rays
            pixel = self.images[i]  # Don't put pixel on device yet, waste of space
            # 非LLFF数据集/非训练集，深度返回None
            depth = self.depth_images[i] if self.depth_images is not None else None
            return self.ray_to_device(ray), pixel ,depth

    def __len__(self):
        if self.split == "render":
            return self.rays[0].shape[0]
        else:
            return len(self.images)

#spherify是渲染视频的旋转方式的调整，False为前后摇晃旋转，True是斜俯视绕物体中心轴旋转
class Multicam(NeRFDataset):
    """Multicam Dataset."""
    def __init__(self, base_dir, split, factor=1, spherify=True, white_bkgd=True, near=2, far=6, radius=4, radii=1, h=800, w=800, device=torch.device("cpu")):
        super(Multicam, self).__init__(base_dir, split, factor=factor, spherify=spherify, near=near, far=far, white_bkgd=white_bkgd, radius=radius, radii=radii, h=h, w=w, device=device)

    def generate_training_poses(self):
        """Load data from disk"""
        with open(path.join(self.base_dir, 'metadata.json'), 'r') as fp:
            split_dir = self.split
            self.meta = json.load(fp)[split_dir]
        # should now have ['pix2cam', 'cam2world', 'width', 'height'] in self.meta
        images = []
        for fbase in self.meta['file_path']:
                fname = os.path.join(self.base_dir, fbase)
                with open(fname, 'rb') as imgin:
                    image = np.array(Image.open(imgin), dtype=np.float32) / 255.
                if self.white_bkgd:
                    image = image[..., :3] * image[..., -1:] + (1. - image[..., -1:])
                images.append(image[..., :3])
        self.pix2cam = self.meta['pix2cam']
        self.cam_to_world = self.meta['cam2world']
        self.w = self.meta['width']
        self.h = self.meta['height']
        self.n_poses = len(images)
        self.images = flatten(images)

    def generate_rays(self):
        """Generating rays for all images"""
        if self.split == "render":
            super().generate_rays()
        else:
            def res2grid(w, h):
                return np.meshgrid(
                    np.arange(w, dtype=np.float32) + .5,  # X-Axis (columns)
                    np.arange(h, dtype=np.float32) + .5,  # Y-Axis (rows)
                    indexing='xy')

            xy = [res2grid(w, h) for w, h in zip(self.w, self.h)]
            pixel_directions = [np.stack([x, y, np.ones_like(x)], axis=-1) for x, y in xy]
            camera_directions = [v @ p2c[:3, :3].T for v, p2c in zip(pixel_directions, self.pix2cam)]
            directions = [v @ c2w[:3, :3].T for v, c2w in zip(camera_directions, self.cam_to_world)]
            origins = [
                np.broadcast_to(c2w[:3, -1], v.shape)
                for v, c2w in zip(directions, self.cam_to_world)
            ]
            viewdirs = [
                v / np.linalg.norm(v, axis=-1, keepdims=True) for v in directions
            ]

            def broadcast_scalar_attribute(x):
                return [
                    np.broadcast_to(x[i], origins[i][..., :1].shape)
                    for i in range(self.n_poses)
                ]

            lossmult = broadcast_scalar_attribute(self.meta['lossmult'])
            near = broadcast_scalar_attribute(self.meta['near'])
            far = broadcast_scalar_attribute(self.meta['far'])

            # Distance from each unit-norm direction vector to its x-axis neighbor.
            dx = [
                np.sqrt(np.sum((v[:-1, :, :] - v[1:, :, :]) ** 2, -1)) for v in directions
            ]
            dx = [np.concatenate([v, v[-2:-1, :]], 0) for v in dx]
            # Cut the distance in half, and then round it out so that it's
            # halfway between inscribed by / circumscribed about the pixel.
            radii = [v[..., None] * 2 / np.sqrt(12) for v in dx]

            self.rays = Rays(
                origins=origins,
                directions=directions,
                viewdirs=viewdirs,
                radii=radii,
                lossmult=lossmult,
                near=near,
                far=far)
            self.rays = namedtuple_map(flatten, self.rays)

#spherify是渲染视频的旋转方式的调整，False为前后摇晃旋转，True是斜俯视绕物体中心轴旋转
class Blender(NeRFDataset):
    """Blender Dataset."""
    def __init__(self, base_dir, split, factor=1, spherify=True, white_bkgd=True, near=2, far=6, radius=4, radii=1, h=800, w=800, device=torch.device("cpu"),use_sparse=False,sparse_json_name="transforms_sparse_32.json"):
        # 保存稀疏配置参数
        self.use_sparse = use_sparse
        self.sparse_json_name = sparse_json_name
        super(Blender, self).__init__(base_dir, split, factor=factor, spherify=spherify, near=near, far=far, white_bkgd=white_bkgd, radius=radius, radii=radii, h=h, w=w, device=device)

    def generate_training_poses(self):
        """Load data from disk"""
        split_dir = self.split

        # 确定要读取的transforms文件路径
        if split_dir == "train" and self.use_sparse:
            # 训练集 + 开启稀疏 → 读取稀疏文件
            json_path = path.join(self.base_dir, self.sparse_json_name)
        else:
            # 验证/测试集 或 未开启稀疏的训练时 → 读取官方文件
            json_path = path.join(self.base_dir, f'transforms_{split_dir}.json')
        # 打印加载日志（验证是否加载正确文件）
        print(f"[Blender Dataset] 加载 {split_dir} 集 transforms 文件：{json_path}")
        # 读取JSON文件
        with open(json_path, 'r') as fp:
            meta = json.load(fp)

        images = []
        cams = []
        for i in range(len(meta['frames'])):
            frame = meta['frames'][i]
            fname = os.path.join(self.base_dir, frame['file_path'] + '.png')
            with open(fname, 'rb') as imgin:
                image = np.array(Image.open(imgin), dtype=np.float32) / 255.
                if self.factor >= 2:
                    [halfres_h, halfres_w] = [hw // 2 for hw in image.shape[:2]]
                    image = cv2.resize(
                        image, (halfres_w, halfres_h), interpolation=cv2.INTER_AREA)
            cams.append(np.array(frame['transform_matrix'], dtype=np.float32))
            images.append(image)
        self.images = np.stack(np.array(images), axis=0)
        if self.white_bkgd:
            self.images = (
                    self.images[..., :3] * self.images[..., -1:] +
                    (1. - self.images[..., -1:]))
        else:
            self.images = self.images[..., :3]
        self.h, self.w = self.images.shape[1:3]
        self.cam_to_world = np.stack(cams, axis=0)
        camera_angle_x = float(meta['camera_angle_x'])
        self.focal = .5 * self.w / np.tan(.5 * camera_angle_x)
        self.n_poses = self.images.shape[0]

#spherify是渲染视频的旋转方式的调整，False为前后摇晃旋转，True是斜俯视绕物体中心轴旋转
class LLFF(NeRFDataset):
    def __init__(self, base_dir, split, factor=4, spherify=False, near=0, far=1, white_bkgd=False, device=torch.device("cpu"),use_sparse=False,sparse_ratio=None,
                 corner_nearest_pair_ratio=0.3, edge_rgb_only_ratio=0.25):
        self.use_sparse = use_sparse
        self.sparse_ratio = sparse_ratio
        self.llff_scan = os.path.basename(base_dir)  # 场景名，用于采样框适配
        self.batch_size = None  # 在get_dataloader(llff/train)中注入
        self.depth_images_for_sampling = None
        self.corner_nearest_pair_ratio = corner_nearest_pair_ratio
        self.edge_rgb_only_ratio = edge_rgb_only_ratio
        super(LLFF, self).__init__(base_dir, split, spherify=spherify, near=near, far=far, white_bkgd=white_bkgd, factor=factor, device=device)

    def generate_training_poses(self):
        """Load data from disk"""
        img_dir = 'images'
        if self.factor != 1:
            img_dir = 'images_' + str(self.factor)
        img_dir = path.join(self.base_dir, img_dir)
        img_files = [
            path.join(img_dir, f)
            for f in sorted(os.listdir(img_dir))
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')
        ]
        images = []
        depth_images = []  # 新增：存储深度图
        for img_file in img_files:
            with open(img_file, 'rb') as img_in:
                image = to_float(np.array(Image.open(img_in)))
                images.append(image)
            # 加载DPT深度图
            root_path = os.path.dirname(os.path.dirname(img_file))
            base = os.path.splitext(os.path.basename(img_file))[0]  # 不带后缀文件名
            candidates = [
                os.path.join(root_path, "depth_maps", f"depth_{base}.png"),
                os.path.join(root_path, "depth_maps", f"depth_{base}.jpg"),
                os.path.join(root_path, "depth_maps", f"depth_{base}.jpeg"),
            ]
            depth_file = next((p for p in candidates if os.path.exists(p)), None)

            if depth_file is not None:
                depth_img = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
                depth_img = depth_img.astype(np.float32)
                depth_img = np.nan_to_num(depth_img, nan=0.0, posinf=0.0, neginf=0.0)
                target_h, target_w = image.shape[:2]
                depth_img = cv2.resize(depth_img, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                depth_images.append(depth_img)
            else:
                raise ValueError(f"深度图不存在，尝试过: {candidates}")

        images = np.stack(images, -1)
        self.depth_images = np.stack(depth_images, -1)  # 新增：保存深度图

        # Load poses
        with open(path.join(self.base_dir, 'poses_bounds.npy'), 'rb') as fp:
            poses_arr = np.load(fp)
        poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
        bds = poses_arr[:, -2:].transpose([1, 0])
        # Update poses according to downsampling.
        poses[:2, 4, :] = np.array(images.shape[:2]).reshape([2, 1])
        poses[2, 4, :] = poses[2, 4, :] * 1. / self.factor
        # Correct rotation matrix ordering and move variable dim to axis 0.
        poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
        poses = np.moveaxis(poses, -1, 0).astype(np.float32)
        images = np.moveaxis(images, -1, 0)
        self.images = images
        bds = np.moveaxis(bds, -1, 0).astype(np.float32)
        # Rescale according to a default bd factor.
        scale = 1. / (bds.min() * .75)
        poses[:, :3, 3] *= scale
        bds *= scale
        self.bds = bds
        # Recenter poses.
        poses = recenter_poses(poses)
        self.poses = poses
        self.images = images
        self.h, self.w = images.shape[1:3]
        self.n_poses = images.shape[0]
        # 深度图维度调整：[H,W,N] → [N,H,W]，和RGB图维度完全对齐
        self.depth_images = np.moveaxis(self.depth_images, -1, 0)

    def generate_render_poses(self):
        self.generate_training_poses()
        self.n_poses = self.n_poses_copy  # get overwritten in generate_training_poses, change back to original
        if self.spherify:
            self.generate_spherical_poses(self.n_poses)
        else:
           self.generate_spiral_poses(self.n_poses)
        self.cam_to_world = self.poses[:, :3, :4]
        self.focal = self.poses[0, -1, -1]

    def generate_training_rays(self):
        self.generate_training_poses()
        # 核心划分：按步长8切分train/test
        if self.split == "train":
            # train训练集：排除所有索引为8的倍数的图像（取其余所有）
            indices = [i for i in np.arange(self.images.shape[0]) if i not in np.arange(self.images.shape[0])[::8]]
            print("Loading Training Poses")
        else:
            # test测试集（val集复用）：只取索引为8的倍数的图像
            indices = np.arange(self.images.shape[0])[::8]
            print("Loading Test Poses")

        # 稀疏采样（仅对train集生效）
        if self.split == "train" and self.use_sparse and self.sparse_ratio is not None:
          # 按稀疏比例均匀采样
          n_sparse = int(len(indices) * self.sparse_ratio)
          n_sparse = max(1, n_sparse)
          # 均匀选索引（保证视角分布）
          sparse_step = len(indices) // n_sparse
          sparse_indices = indices[::sparse_step]
          # 确保数量准确（避免余数问题）
          if len(sparse_indices) > n_sparse:
              sparse_indices = sparse_indices[:n_sparse]
          print(f"[LLFF Dataset] 稀疏采样：原始训练视角数 {len(indices)} → 稀疏后 {len(sparse_indices)}")
          indices = sparse_indices

        # ========== 核心修复：所有数据（RGB/位姿/深度图）用同一套稀疏索引筛选 ==========
        self.images = self.images[indices]
        self.poses = self.poses[indices]
        # 关键：深度图必须和RGB图用完全相同的稀疏索引筛选
        self.depth_images = self.depth_images[indices]
        self.depth_images_for_sampling = self.depth_images.copy()

        # 更新关键参数（确保n_poses是稀疏后的数量）
        self.n_poses = len(self.images)  # 核心：n_poses同步为稀疏后的数量
        self.cam_to_world = self.poses[:, :3, :4]
        self.focal = self.poses[0, -1, -1]
        print("Generating rays")
        self.generate_rays()

    def __len__(self):
        """LLFF训练集：按batch_size动态生成2-ray有序索引，长度为总射线数//batch_size"""
        if self.split == "train":
            if self.batch_size is None:
                raise ValueError("LLFF train dataset batch_size is not set. Please use get_dataloader(..., batch_size=...).")
            return len(self.rays[0]) // self.batch_size
        elif self.split == "render":
            return self.rays[0].shape[0]
        else:
            return len(self.images)

    def __getitem__(self, i):
        """LLFF训练集：混合采样（局部排序主样本 + 边缘RGB补样），并显式返回排序样本数量。"""
        if self.split == "train":
            if self.batch_size is None:
                raise ValueError("LLFF train dataset batch_size is not set. Please use get_dataloader(..., batch_size=...).")

            # A Hybrid Sampling Strategy:
            # 1) 主局部框采样：用于深度排序监督 + RGB损失
            # 2) 四角/边缘补充采样：仅用于RGB覆盖增强，不参与深度排序监督
            edge_ratio = float(np.clip(self.edge_rgb_only_ratio, 0.0, 0.95))
            edge_rgb_count = int(self.batch_size * edge_ratio)
            rank_count = self.batch_size - edge_rgb_count
            rank_count = (rank_count // 4) * 4
            if rank_count <= 0:
                rank_count = min(self.batch_size, 4)
            if rank_count > self.batch_size:
                rank_count = (self.batch_size // 4) * 4
            if rank_count <= 0:
                rank_count = self.batch_size
            edge_rgb_count = self.batch_size - rank_count

            # 随机选一张图像，生成4-ray有序索引
            img_idx = np.random.randint(0, self.n_poses)
            # 单张图像的射线起始偏移
            img_start_idx = img_idx * self.h * self.w
            depth_img = self.depth_images_for_sampling[img_idx]
            # 主采样：生成有序4-ray索引（相对于单张图像的局部索引）
            local_rank_indices = sample_depth_ranking_ray_indices(
            depth_image = depth_img,
            batch_size = rank_count,
            llff_scan = self.llff_scan,
            corner_nearest_pair_ratio = self.corner_nearest_pair_ratio,
            )
            # 补样：四角/边缘补充，仅供RGB监督
            local_edge_indices = sample_edge_only_rgb_ray_indices(depth_img, edge_rgb_count)
            local_indices = np.concatenate([local_rank_indices, local_edge_indices], axis=0)
            # 转换为全局索引
            global_indices = img_start_idx + local_indices
            # 按全局索引取对齐的ray/rgb/depth
            ray = namedtuple_map(lambda r: r[global_indices], self.rays)
            pixel = self.images[global_indices]
            ray = self.ray_to_device(ray)
            # 返回排序样本数量，供训练时切分depth ranking监督范围
            return ray, pixel, torch.tensor(rank_count, dtype=torch.int64)
        elif self.split == "render":
            ray = namedtuple_map(lambda r: r[i], self.rays)
            return ray
        else:
            # test集保持原有逻辑
            ray = namedtuple_map(lambda r: r[i], self.rays)
            pixel = self.images[i]
            depth = self.depth_images.reshape(-1)[i] if self.depth_images is not None else None
            return self.ray_to_device(ray), pixel, depth

    def generate_spherical_poses(self, n_poses=120):
        """Generate a 360 degree spherical path for rendering."""
        p34_to_44 = lambda p: np.concatenate([
            p,
            np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])
        ], 1)
        rays_d = self.poses[:, :3, 2:3]
        rays_o = self.poses[:, :3, 3:4]

        def min_line_dist(rays_o, rays_d):
            a_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
            b_i = -a_i @ rays_o
            pt_mindist = np.squeeze(-np.linalg.inv(
                (np.transpose(a_i, [0, 2, 1]) @ a_i).mean(0)) @ (b_i).mean(0))
            return pt_mindist

        pt_mindist = min_line_dist(rays_o, rays_d)
        center = pt_mindist
        up = (self.poses[:, :3, 3] - center).mean(0)
        vec0 = normalize(up)
        vec1 = normalize(np.cross([.1, .2, .3], vec0))
        vec2 = normalize(np.cross(vec0, vec1))
        pos = center
        c2w = np.stack([vec1, vec2, vec0, pos], 1)
        poses_reset = (
                np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(self.poses[:, :3, :4]))
        rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))
        sc = 1. / rad
        poses_reset[:, :3, 3] *= sc
        self.bds *= sc
        rad *= sc
        centroid = np.mean(poses_reset[:, :3, 3], 0)
        zh = centroid[2]
        radcircle = np.sqrt(rad ** 2 - zh ** 2)
        new_poses = []

        for th in np.linspace(0., 2. * np.pi, n_poses):
            cam_origin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
            up = np.array([0, 0, -1.])
            vec2 = normalize(cam_origin)
            vec0 = normalize(np.cross(vec2, up))
            vec1 = normalize(np.cross(vec2, vec0))
            pos = cam_origin
            p = np.stack([vec0, vec1, vec2, pos], 1)
            new_poses.append(p)

        new_poses = np.stack(new_poses, 0)
        self.poses = np.concatenate([
            new_poses,
            np.broadcast_to(self.poses[0, :3, -1:], new_poses[:, :3, -1:].shape)
        ], -1)
        # self.poses = np.concatenate([
        #     poses_reset[:, :3, :4],
        #     np.broadcast_to(self.poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)
        # ], -1)

    def generate_spiral_poses(self, n_poses=120):
        """Generate a spiral path for rendering."""
        c2w = poses_avg(self.poses)
        # Get average pose.
        up = normalize(self.poses[:, :3, 1].sum(0))
        # Find a reasonable 'focus depth' for this dataset.
        close_depth, inf_depth = self.bds.min() * .9, self.bds.max() * 5.
        dt = .75
        mean_dz = 1. / (((1. - dt) / close_depth + dt / inf_depth))
        focal = mean_dz
        # Get radii for spiral path.
        tt = self.poses[:, :3, 3]
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        n_rots = 2
        # Generate poses for spiral path.
        render_poses = []
        rads = np.array(list(rads) + [1.])
        hwf = c2w_path[:, 4:5]
        zrate = .5
        for theta in np.linspace(0., 2. * np.pi * n_rots, n_poses + 1)[:-1]:
            c = np.dot(c2w[:3, :4], (np.array(
                [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads))
            z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
            render_poses.append(np.concatenate([look_at(z, up, c), hwf], 1))
        self.poses = np.array(render_poses).astype(np.float32)

    def generate_rays(self):
        """Generate normalized device coordinate rays for llff."""
        super().generate_rays()
        ndc_origins, ndc_directions = convert_to_ndc(self.rays.origins, self.rays.directions, self.focal, self.w, self.h)
        mat = ndc_origins
        # Distance from each unit-norm direction vector to its x-axis neighbor.
        dx = np.sqrt(np.sum((mat[:, :-1, :, :] - mat[:, 1:, :, :]) ** 2, -1))
        dx = np.concatenate([dx, dx[:, -2:-1, :]], 1)

        dy = np.sqrt(np.sum((mat[:, :, :-1, :] - mat[:, :, 1:, :]) ** 2, -1))
        dy = np.concatenate([dy, dy[:, :, -2:-1]], 2)
        # Cut the distance in half, and then round it out so that it's
        # halfway between inscribed by / circumscribed about the pixel.
        radii = (0.5 * (dx + dy))[..., None] * 2 / np.sqrt(12)

        ones = np.ones_like(ndc_origins[..., :1])
        self.rays = Rays(
            origins=ndc_origins,
            directions=ndc_directions,
            viewdirs=self.rays.directions,
            radii=radii,
            lossmult=ones,
            near=ones * self.near,
            far=ones * self.far)


dataset_dict = {
    'blender': Blender,
    'llff': LLFF,
    'multicam': Multicam,
}
