#!/usr/bin/env python
"""Prepare dataset4 (RGB) for QUEEN training.

This mirrors the dataset2 adapter script:
- Create per-camera sequences under: camXX/images/0000.png ... 0159.png
- Copy COLMAP sparse/ and frame0 images/ from 3DGStream frame000000

Output layout:
  output_root/
    cam00/images/0000.png ...
    ...
    cam24/images/...
    sparse/0/...
    images/cam00.png ... (frame000000 for COLMAP loader)
"""

import argparse
import os
import shutil
from pathlib import Path


def _symlink_or_copy(src: Path, dst: Path, copy: bool = False) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(src, dst)
    else:
        rel = os.path.relpath(src, dst.parent)
        os.symlink(rel, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare dataset4 for QUEEN")
    parser.add_argument(
        "--raw-root",
        type=str,
        default="/scratch/tc4146/dataset4/dataset4_rgb",
        help="Root of raw multi-view frames (contains cam00, cam01, ...)",
    )
    parser.add_argument(
        "--dgstream-frame0",
        type=str,
        default="/scratch/tc4146/3dgstream_experiment/dataset4_3dgstream/frame000000",
        help="Path to frame000000 directory with COLMAP sparse/ and images/",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="/scratch/tc4146/repro/data_adapters/dataset4_queen_scene",
        help="Output root to write QUEEN-formatted dataset",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy PNGs instead of creating symlinks (uses more disk).",
    )
    args = parser.parse_args()

    raw_root = Path(args.raw_root).resolve()
    frame0_root = Path(args.dgstream_frame0).resolve()
    out_root = Path(args.output_root).resolve()

    if not raw_root.exists():
        raise SystemExit(f"raw_root does not exist: {raw_root}")
    if not frame0_root.exists():
        raise SystemExit(f"dgstream-frame0 does not exist: {frame0_root}")

    out_root.mkdir(parents=True, exist_ok=True)

    # 1) Per-camera multi-frame images for MultiViewVideoDataset
    cam_dirs = sorted([p for p in raw_root.iterdir() if p.is_dir() and p.name.startswith("cam")])
    if not cam_dirs:
        raise SystemExit(f"No cam* directories found under {raw_root}")

    for cam_dir in cam_dirs:
        images = sorted(cam_dir.glob("frame*.png"))
        if not images:
            print(f"[WARN] No frame*.png under {cam_dir}, skipping")
            continue
        tgt_images_dir = out_root / cam_dir.name / "images"
        for idx, src in enumerate(images):
            dst = tgt_images_dir / f"{idx:04d}.png"
            _symlink_or_copy(src, dst, copy=args.copy)

    # 2) Static COLMAP data from frame000000 for Scene(readColmapSceneInfo)
    sparse_src = frame0_root / "sparse"
    images_src = frame0_root / "images"
    if not sparse_src.exists() or not images_src.exists():
        raise SystemExit(f"Expected sparse/ and images/ under {frame0_root}")

    for name, src_dir in ("sparse", sparse_src), ("images", images_src):
        dst_dir = out_root / name
        if dst_dir.exists():
            continue
        shutil.copytree(src_dir, dst_dir)

    print("[OK] Prepared dataset4 for QUEEN at:", out_root)


if __name__ == "__main__":
    main()



