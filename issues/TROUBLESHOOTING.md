# Troubleshooting Notes (English)

This section summarizes the main issues we hit on NYU Greene and how we resolved them.

## 1) `no kernel image is available for execution on the device`

**Cause**: CUDA extensions (e.g., `simple-knn`, rasterizers) were compiled for a different GPU architecture.

**Fix**:

- Maintain **one conda env per GPU type** (e.g., `queen_rtx8000`).
- Install a matching CUDA toolkit (we used nvcc 11.8 in the env).
- Rebuild extensions with:

```bash
export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
pip install -v --no-build-isolation <submodule-path>
```

## 2) Missing MiDaS checkpoint

**Symptom**:

`FileNotFoundError: MiDaS/weights/dpt_beit_large_512.pt`

**Fix**: download MiDaS v3.1 weight file (see the usage guide).

## 3) Abnormal OOM in spiral rendering

**Symptom**: An OOM that tries to allocate an impossible amount of memory (tens/hundreds of TB) during a spiral visualization render.

**Fix**:

1) Rebuild `diff-gaussian-rasterization` inside the GPU-specific env.
2) Guard spiral render with an OOM-safe try/except so training does not abort on visualization.

Practical notes:

- This OOM can happen very late (near the end of a frame's optimization), because spiral rendering is triggered during logging.
- Spiral images are **visualization only**; skipping them does not affect training quality.

## 4) Greene runtime: `ImportError: libc10.so: cannot open shared object file`

**Fix**:

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib:$LD_LIBRARY_PATH
```

## 5) No system `ffmpeg` on Greene

If `ffmpeg` is not available, export MP4 using `imageio-ffmpeg`:

```bash
python -m pip install -U imageio-ffmpeg
python scripts/make_videos.py
```

