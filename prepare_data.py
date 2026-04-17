"""
Convert CIFAR-10 to a flat binary of 8x8 float32 blocks.

Layout of data/blocks.bin:
    N float32 values, row-major, = n_blocks * 8 * 8.

Layout of data/meta.json:
    { "n_blocks": ..., "block_size": 8, "dtype": "float32",
      "source": "CIFAR-10 Y-channel", "image_shape": [32,32],
      "blocks_per_image": 16 }

CIFAR-10: 60 000 images (50k train + 10k test) => 960 000 blocks.
We convert RGB -> Y (BT.601) then split each 32x32 image into 16 8x8 blocks.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torchvision


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
DATA_DIR.mkdir(exist_ok=True)


def rgb_to_y(img_uint8: np.ndarray) -> np.ndarray:
    """BT.601 luma. Input (H,W,3) uint8, output (H,W) float32 in [0,255]."""
    r = img_uint8[..., 0].astype(np.float32)
    g = img_uint8[..., 1].astype(np.float32)
    b = img_uint8[..., 2].astype(np.float32)
    return 0.299 * r + 0.587 * g + 0.114 * b


def image_to_blocks(y: np.ndarray, block: int = 8) -> np.ndarray:
    """(H,W) -> (H/block * W/block, block, block). Requires H,W divisible by block."""
    h, w = y.shape
    assert h % block == 0 and w % block == 0, (h, w, block)
    # reshape then swap: (nh, b, nw, b) -> (nh, nw, b, b) -> (nh*nw, b, b)
    nh, nw = h // block, w // block
    y = y.reshape(nh, block, nw, block).transpose(0, 2, 1, 3)
    return y.reshape(nh * nw, block, block).copy()


def main() -> None:
    print("Downloading CIFAR-10 (torchvision)...")
    train = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR / "_cifar10_raw"), train=True, download=True
    )
    test = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR / "_cifar10_raw"), train=False, download=True
    )

    # stack images -> (N, 32, 32, 3) uint8
    imgs = np.concatenate([train.data, test.data], axis=0)
    n_images, h, w, _ = imgs.shape
    print(f"  stacked: {imgs.shape} {imgs.dtype}")

    print("Computing Y channel...")
    y = rgb_to_y(imgs)  # (N, 32, 32) float32 in [0,255]

    # center to zero-mean-ish like JPEG (subtract 128) — doesn't matter for DCT
    # linearity but matches JPEG practice and keeps values symmetric around 0
    y -= 128.0

    print("Splitting into 8x8 blocks...")
    block = 8
    blocks = np.empty(
        (n_images * (h // block) * (w // block), block, block), dtype=np.float32
    )
    step = (h // block) * (w // block)
    for i in range(n_images):
        blocks[i * step : (i + 1) * step] = image_to_blocks(y[i], block)

    print(f"  blocks: {blocks.shape} {blocks.dtype}, total MB: "
          f"{blocks.nbytes / 1e6:.2f}")

    out_bin = DATA_DIR / "blocks.bin"
    out_meta = DATA_DIR / "meta.json"

    blocks.tofile(out_bin)  # raw row-major float32, little-endian on x86/arm
    meta = {
        "n_blocks": int(blocks.shape[0]),
        "block_size": int(block),
        "dtype": "float32",
        "byte_order": "little",
        "source": "CIFAR-10 Y-channel (BT.601), centered (-128)",
        "image_shape": [h, w],
        "blocks_per_image": step,
        "bin_bytes": int(blocks.nbytes),
    }
    out_meta.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {out_bin} ({blocks.nbytes} bytes) and {out_meta}")


if __name__ == "__main__":
    main()
