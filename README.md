# SparseNeRF-PyTorch / Dep-NeRF

[中文说明](README_CN.md)

An unofficial PyTorch research implementation inspired by
[SparseNeRF](https://github.com/Wanggcong/SparseNeRF) for sparse-view novel
view synthesis. This repository refers to the improved method as **Dep-NeRF**.
It keeps a Mip-NeRF backbone and adds local relative-depth ranking, edge-aware
RGB sampling, and configurable dataset sparsification.

> [!IMPORTANT]
> This is not an official or line-by-line port of SparseNeRF. Unlike the
> original method, this implementation uses **depth-ranking loss only** and
> does **not** include the spatial depth-continuity loss. The repository is
> intended for research, reproduction, and sparse-view novel view synthesis experiments.

## Highlights

- PyTorch implementation built around a Mip-NeRF-style multi-scale backbone.
- Local relative-depth ranking supervision from monocular depth priors.
- Four-ray ranking groups: `[far, near, far_neighbor, near_neighbor]`.
- Edge/corner auxiliary rays used for RGB reconstruction only.
- Configurable sparse-view selection for LLFF and Blender-style datasets.
- Evaluation with PSNR, SSIM, LPIPS, RGB renders, and rendered depth maps.
- Tested on a 6 GB NVIDIA GeForce RTX 3060.

## Method

Sparse-view NeRF training is underconstrained when RGB reconstruction is the
only supervision. Dep-NeRF preserves the Mip-NeRF rendering backbone and adds a
relative-depth ranking term:

```text
L_total = L_rgb + lambda_rank * L_rank
```

The ranking term uses local front/back relationships from monocular depth
priors instead of regressing absolute depth values. This makes the supervision
less sensitive to the scale ambiguity of monocular depth estimation.

The training sampler contains two parts:

1. **Local ranking samples** participate in both RGB reconstruction and
   depth-ranking supervision.
2. **Edge/corner auxiliary samples** improve image-boundary coverage and
   participate in RGB reconstruction only.

The network architecture and forward rendering process remain compatible with
the Mip-NeRF backbone; the main changes are in `loss.py`, `datasets.py`,
`train.py`, and `config.py`.

## Differences From SparseNeRF

| Item | This repository |
| --- | --- |
| Framework | PyTorch |
| Backbone | Mip-NeRF-style cone/cylinder sampling and integrated positional encoding |
| Geometric regularization | Local depth-ranking loss |
| Depth-continuity loss | Not implemented |
| Ranking sampler | Local four-ray groups with neighbor pairs |
| Boundary compensation | Edge/corner RGB-only auxiliary sampling |
| Sparse dataset support | Configurable LLFF ratio and Blender sparse JSON file |

## Experimental Results

The following results are from controlled comparison experiments. All models were
trained and evaluated under matched settings. Higher PSNR/SSIM and lower LPIPS
are better.

| Dataset | Train views | Model | PSNR | SSIM | LPIPS |
| --- | ---: | --- | ---: | ---: | ---: |
| LLFF horns | 5 | NeRF | 17.6062 | 0.5417 | 0.5128 |
| LLFF horns | 5 | Mip-NeRF | 19.9849 | 0.6132 | 0.4286 |
| LLFF horns | 5 | Dep-NeRF | **20.7247** | **0.6301** | **0.4164** |
| Self-captured locomotive wheelset | 6 | NeRF | 16.4132 | 0.5344 | 0.6816 |
| Self-captured locomotive wheelset | 6 | Mip-NeRF | 18.7422 | 0.6336 | 0.5612 |
| Self-captured locomotive wheelset | 6 | Dep-NeRF | **19.7217** | **0.6847** | **0.5501** |
| Self-captured indoor locomotive | 4 | NeRF | 12.3590 | 0.4210 | 0.7088 |
| Self-captured indoor locomotive | 4 | Mip-NeRF | 16.0914 | 0.5654 | 0.5536 |
| Self-captured indoor locomotive | 4 | Dep-NeRF | **16.4870** | **0.6042** | **0.5298** |

These results support the intended use case: **few-view training**. Additional
dense-view experiments did not show a consistent improvement over Mip-NeRF.
When multi-view supervision is already sufficient, reduce
`depth_rank_weight` or disable the ranking regularizer.

## Tested Environment

The tested experiments used:

| Component | Version |
| --- | --- |
| Python | 3.9.7 |
| PyTorch | 1.11.0 |
| CUDA | 11.3 |
| GPU | NVIDIA GeForce RTX 3060 6 GB |

Pinned Python dependencies are listed in `requirements.txt`. The versions
reflect the tested environment and may need adjustment for newer CUDA, GPU, or
operating-system combinations.

## Installation

```bash
git clone https://github.com/lansanyue031018-rgb/SparseNeRF-pytorch.git
cd SparseNeRF-pytorch

conda create -n depnerf python=3.9.7
conda activate depnerf

pip install -r requirements.txt
```

For the tested CUDA 11.3 environment, reinstall the CUDA-specific PyTorch
wheels after the requirements file:

```bash
pip install --force-reinstall \
  torch==1.11.0+cu113 \
  torchvision==0.12.0+cu113 \
  torchaudio==0.11.0 \
  --index-url https://download.pytorch.org/whl/cu113
```

Optional mesh-extraction dependencies such as `plyfile`, `PyMCubes`, and
`open3d` are not pinned in the current requirements file.

## Dataset Preparation

### LLFF layout

Place LLFF scenes under `data/nerf_llff_data/`:

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

Depth-map filenames must match the RGB image filenames with a `depth_` prefix.
For example, `images_4/000.png` corresponds to
`depth_maps/depth_000.png`.

Monocular relative-depth priors are part of dataset preparation rather than
the NeRF training loop. You can generate them with
[MiDaS](https://github.com/isl-org/MiDaS),
[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), or
another relative-depth model. Convert each output to a single-channel image
(16-bit PNG is recommended), add the `depth_` filename prefix, and place it in
the scene's `depth_maps/` directory.

The ranking sampler assumes an **inverse-depth convention: larger values are
nearer and smaller values are farther**. If a model exports metric depth or the
opposite ordering, invert or otherwise convert the values before training.

The included helper script `get_depth_map_for_llff_dtu.py` uses MiDaS/DPT. It
currently contains machine-specific absolute paths. Before running it, edit
both `repo_or_dir` entries to point to your local MiDaS checkout and update
`weight_path` to the downloaded checkpoint:

```bash
python get_depth_map_for_llff_dtu.py \
  --benchmark llff \
  --dataset_id horns \
  --root_path data/nerf_llff_data
```

### Sparse-view configuration

- LLFF uses `sparse_ratio` to uniformly retain a fraction of the training
  views.
- Blender-style datasets use `sparse_json_name`, for example
  `transforms_sparse_8.json`.
- The current legacy CLI defines `--use_sparse` with `action="store_false"`.
  Sparse mode is therefore enabled by default, and passing `--use_sparse`
  disables it.

## Training

Example configuration close to the sparse-view comparison setting:

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

For LLFF, default dataset settings set `factor=4`, `ray_shape=cylinder`,
`white_bkgd=False`, and `test_skip=1`. Pass `--override_defaults` if you need
the explicit command-line values to take precedence.

To resume training:

```bash
python train.py \
  --dataset_name llff \
  --scene horns \
  --log_dir log/horns_depnerf \
  --continue_training
```

### Batch experiments (experimental)

`run_experiments.py` is an experimental sequential scheduler for running
multiple dataset/configuration jobs. After one training job reaches its
configured milestone or exits, the scheduler starts the next run in
`exp_plan.json`; existing checkpoints can be resumed automatically.

The checked-in defaults are examples from the original Windows experiment
environment and are not portable as-is. Before use, update:

- `--plan`, `--python`, and `--logs-dir` defaults in `run_experiments.py`, or
  pass them explicitly on the command line.
- `global.log_dir` and every run's `cwd` in `exp_plan.json`.
- Scene names, sparse ratios, seeds, repeat counts, and `extra_args` for your
  own datasets and hardware.

```bash
python run_experiments.py \
  --plan /path/to/exp_plan.json \
  --python /path/to/python \
  --logs-dir /path/to/scheduler_logs
```

### Main configuration options

| Option | Code default | Description |
| --- | ---: | --- |
| `depth_rank_weight` | `0.5` | Weight of the depth-ranking regularizer |
| `corner_nearest_pair_ratio` | `0.0` | Probability of moving a local ranking window toward a nearest corner |
| `edge_rgb_only_ratio` | `0.05` | Batch fraction reserved for edge/corner RGB-only rays |
| `sparse_ratio` | `0.2` | Fraction of LLFF training views retained |
| `sparse_json_name` | `transforms_sparse_32.json` | Sparse Blender split file |
| `batch_size` | `1024` | Training rays per batch |
| `max_steps` | `200000` | Maximum training steps |
| `save_every` | `33000` | Checkpoint/evaluation interval |
| `chunks` | `4096` | Rendering chunk size |

In the current `loss.py`, `margin_pair` and `neighbor_pair_weight` are function
defaults rather than command-line arguments. The comparison setting used
`margin_pair=1e-4`; check the implementation before claiming exact
reproduction of a specific experiment.

## Evaluation

```bash
python evaluate.py \
  --dataset_name llff \
  --scene horns \
  --model_weight_path log/horns_depnerf/model.pt \
  --log_dir log/horns_depnerf
```

Evaluation outputs are written to:

```text
log/horns_depnerf/eval_results/
  pred/
  gt/
  eval_metrics.txt
```

## Rendering

```bash
python visualize.py \
  --dataset_name llff \
  --scene horns \
  --model_weight_path log/horns_depnerf/model.pt \
  --log_dir log/horns_depnerf \
  --visualize_depth
```

The renderer writes RGB frames and videos under the selected log directory.
Video export requires an FFmpeg installation supported by `imageio`.

## Repository Structure

```text
config.py                       # CLI configuration
datasets.py                     # Dataset loading, sparsification, and sampling
loss.py                         # RGB and local depth-ranking losses
model.py                        # Mip-NeRF model
train.py                        # Training and checkpointing
evaluate.py                     # PSNR, SSIM, and LPIPS evaluation
visualize.py                    # RGB/depth video rendering
get_depth_map_for_llff_dtu.py   # MiDaS/DPT depth-prior generation
run_experiments.py              # Sequential experiment scheduler
extract_mesh.py                 # Optional mesh extraction
```

## Known Limitations

- The method is designed for sparse-view settings and may not improve
  dense-view training.
- Results depend on monocular depth quality, camera-pose accuracy, view
  coverage, and scene appearance.
- Reflective, transparent, weak-texture, and repetitive-texture regions may
  produce incorrect depth rankings.
- The self-captured datasets and pretrained checkpoints are not currently
  included.
- The MiDaS helper and experimental batch scheduler contain example absolute
  Windows paths that must be replaced or overridden for each machine.
- Some legacy boolean CLI flags use `store_false` and may be unintuitive.

## Roadmap

- Add a cross-platform Conda or container environment definition.
- Remove machine-specific paths from depth generation and experiment scripts.
- Publish reproducible configs, checkpoints, and visual comparisons.
- Add ablation results for ranking and edge-sampling components.
- Normalize the sparse-mode CLI and expose all ranking hyperparameters.

## Acknowledgements

This project is inspired by and builds on ideas from:

- [SparseNeRF: Distilling Depth Ranking for Few-shot Novel View Synthesis](https://github.com/Wanggcong/SparseNeRF)
- [Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields](https://jonbarron.info/mipnerf/)
- [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://www.matthewtancik.com/nerf)
- [MiDaS](https://github.com/isl-org/MiDaS)
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)

## Citation

If this repository helps your research, please cite the original SparseNeRF
and Mip-NeRF papers:

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

## License

Original contributions in this repository are released under the
[MIT License](LICENSE).

Third-party code, upstream-derived code, models, checkpoints, datasets, and
assets remain subject to their respective licenses. In particular, the
original SparseNeRF repository uses the
[S-Lab License 1.0](https://github.com/Wanggcong/SparseNeRF/blob/main/LICENSE)
with non-commercial terms; this repository's MIT license does not override
those restrictions.
