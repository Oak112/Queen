#!/usr/bin/env python3
"""
Create MP4 videos from QUEEN render folders (no system ffmpeg required).

Uses imageio-ffmpeg (bundled ffmpeg binary) via imageio.
"""

from __future__ import annotations

from pathlib import Path
import re

import imageio.v2 as imageio
from PIL import Image
import numpy as np


def sorted_frames(folder: Path) -> list[Path]:
    pngs = list(folder.glob("*.png"))
    # Expect numeric filenames like 0001.png ... 0160.png
    def key(p: Path) -> int:
        m = re.match(r"^(\d+)\.png$", p.name)
        return int(m.group(1)) if m else 10**18
    return sorted(pngs, key=key)


def write_mp4(input_dir: Path, output_path: Path, fps: int = 30) -> None:
    frames = sorted_frames(input_dir)
    if not frames:
        raise FileNotFoundError(f"No PNGs found under: {input_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8) as writer:
        for p in frames:
            # imageio reads fine, but PIL guarantees RGB
            im = Image.open(p).convert("RGB")
            writer.append_data(np.asarray(im))

    print(f"[OK] Wrote {output_path} ({len(frames)} frames @ {fps} fps)")


def main() -> None:
    run_dir = Path("/scratch/tc4146/repro/runs/dataset4_queen_full_rtx8000")
    renders_root = run_dir / "test" / "renders"
    out_dir = run_dir / "videos"

    for cam in ["cam11", "cam13"]:
        write_mp4(renders_root / cam, out_dir / f"{cam}.mp4", fps=30)


if __name__ == "__main__":
    main()

