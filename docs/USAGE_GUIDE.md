# Usage Guide (English)

This guide documents how we reproduced QUEEN (queen) on our multi-view video dataset on NYU Greene.

## 1) Environment

We strongly recommend **one environment per GPU type** on Greene (V100 / RTX8000 / A100 may vary).

Example (RTX8000):

- Env path: `/scratch/tc4146/conda_envs/queen_rtx8000`
- GPU: Quadro RTX 8000 (compute capability 7.5)

Key runtime environment variable (required on Greene for torch C++ extensions):

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib:$LD_LIBRARY_PATH
```

### GPU-specific CUDA extensions

QUEEN relies on multiple custom CUDA extensions (e.g., `simple-knn`, rasterizers). If you switch GPU types
on Greene, you may need to rebuild these extensions in a GPU-specific environment.

## 2) Data preparation

QUEEN expects:

- Per-camera image sequences under `source_path/camXX/images/0000.png ...`
- A COLMAP sparse model for a reference frame under `source_path/sparse/0`
- A static `images/` folder for COLMAP image names under `source_path/images`

We generate this layout by running:

```bash
python scripts/prepare_dataset4_for_queen.py \
  --raw-root /scratch/tc4146/dataset4/dataset4_rgb \
  --dgstream-frame0 /scratch/tc4146/3dgstream_experiment/dataset4_3dgstream/frame000000 \
  --output-root /scratch/tc4146/repro/data_adapters/dataset4_queen_scene
```

## 3) Required depth model checkpoint (MiDaS)

QUEEN uses MiDaS to initialize depth.

Download:

```bash
mkdir -p code/queen/MiDaS/weights
wget -O code/queen/MiDaS/weights/dpt_beit_large_512.pt \
  https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt
```

## 4) Training

Run from the repository root:

```bash
cd code/queen

PYTHONUNBUFFERED=1 python train.py \
  --config configs/dataset4_queen_full.yaml \
  --source_path /scratch/tc4146/repro/data_adapters/dataset4_queen_scene \
  --model_path /scratch/tc4146/repro/runs/dataset4_queen_full_rtx8000 \
  --depth_model_ckpt code/queen/MiDaS/weights/dpt_beit_large_512.pt
```

The run will write:

- `test/renders/cam11/*.png`
- `test/renders/cam13/*.png`
- `training_metrics.json` and `avg_metrics.json` (if enabled by the code)

## 5) Export MP4 videos (cam11/cam13)

Greene may not provide a system `ffmpeg`. We export MP4 via `imageio-ffmpeg` (bundled ffmpeg binary).

Install once (system-independent):

```bash
python -m pip install -U imageio-ffmpeg
```

```bash
python scripts/make_videos.py
```

Outputs:

- `/scratch/tc4146/repro/runs/dataset4_queen_full_rtx8000/videos/cam11.mp4`
- `/scratch/tc4146/repro/runs/dataset4_queen_full_rtx8000/videos/cam13.mp4`

