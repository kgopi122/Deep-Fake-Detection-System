from __future__ import annotations

import cv2
import numpy as np


def load_image_rgb(path: str, max_side: int = 2048) -> np.ndarray:
    data = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if data is None:
        raise ValueError(f"Failed to read image: {path}")

    h, w = data.shape[:2]
    scale = min(1.0, float(max_side) / float(max(h, w)))
    if scale < 1.0:
        data = cv2.resize(data, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)

    rgb = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    return rgb
