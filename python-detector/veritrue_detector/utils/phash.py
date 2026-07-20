from __future__ import annotations

import cv2
import numpy as np


def phash64(image_rgb: np.ndarray) -> int:
    # pHash: resize -> DCT -> keep top-left 8x8 (excluding DC) -> binarize against median
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)

    dct = cv2.dct(small)
    block = dct[:8, :8].copy()
    block[0, 0] = 0.0

    med = float(np.median(block))
    bits = (block > med).astype(np.uint8).flatten()

    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return int(h)
