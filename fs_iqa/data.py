"""Explicit image selection; no implicit train/test repartitioning."""

import csv
import random
from pathlib import Path

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def select_images(image_dir, *, image_list=None, start_index=0, shuffle=False,
                  seed=0, limit=None):
    root = Path(image_dir).resolve()
    if not root.is_dir():
        raise ValueError(f"Image directory does not exist: {root}")
    if start_index < 0 or (limit is not None and limit < 1):
        raise ValueError("start_index must be >= 0; limit must be positive")
    if image_list is not None:
        names = [line.strip() for line in Path(image_list).read_text().splitlines()
                 if line.strip()]
        images = []
        for name in names:
            path = (root / name).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f"Image-list entry is missing or outside image_dir: {name}")
            images.append(path)
        if len(set(images)) != len(images):
            raise ValueError("Image list contains duplicates")
    else:
        images = sorted(p for p in root.iterdir()
                        if p.is_file() and p.suffix.lower() in EXTENSIONS)
    images = images[start_index:]
    if shuffle:
        random.Random(seed).shuffle(images)
    if limit is not None:
        images = images[:limit]
    if not images:
        raise ValueError("No images selected")
    return images


def load_mos(path, column="MOS_zscore"):
    if path is None:
        return {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"image_name", column}.issubset(reader.fieldnames or []):
            raise ValueError(f"MOS CSV needs image_name and {column} columns")
        values = {}
        for row in reader:
            name = row["image_name"]
            if name in values:
                raise ValueError(f"Duplicate MOS key: {name}")
            values[name] = float(row[column])
    return values


def read_image(path, *, width=512, height=384):
    import cv2
    import numpy as np
    import torch
    from PIL import Image

    with Image.open(path) as im:
        pixels = np.asarray(im.convert("RGB"), dtype=np.float32)
    pixels = cv2.resize(pixels, (width, height), interpolation=cv2.INTER_LINEAR) / 255.0
    return torch.from_numpy(pixels.transpose(2, 0, 1).copy()).unsqueeze(0)
