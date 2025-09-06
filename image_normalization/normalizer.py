from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat

import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

from .config import ImageNormalizationConfig
from .metrics import export_additional_metrics


ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass
class ImageStats:
    count: int
    mean: float
    std: float
    min_val: float
    max_val: float
    width_mean: float
    height_mean: float
    aspect_ratio_mean: float


def _iter_image_paths(images_root: Path) -> Iterable[Path]:
    for sub in sorted(images_root.glob("*/")):
        # Expect third-level folders like 010, 011, ...
        for img_path in sub.glob("*.jpg"):
            yield img_path


def _compute_basic_stats_from_sample(paths: List[Path], sample_size: int) -> ImageStats:
    if not paths:
        return ImageStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    sample = paths if len(paths) <= sample_size else list(np.random.choice(paths, size=sample_size, replace=False))

    means = []
    stds = []
    mins = []
    maxs = []
    widths = []
    heights = []
    ratios = []

    for p in sample:
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                arr = np.asarray(im, dtype=np.float32) / 255.0
                means.append(float(arr.mean()))
                stds.append(float(arr.std()))
                mins.append(float(arr.min()))
                maxs.append(float(arr.max()))
                w, h = im.size
                widths.append(w)
                heights.append(h)
                ratios.append(w / h)
        except Exception:
            continue

    if len(means) == 0:
        return ImageStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    return ImageStats(
        count=len(means),
        mean=float(np.mean(means)),
        std=float(np.mean(stds)),
        min_val=float(np.mean(mins)),
        max_val=float(np.mean(maxs)),
        width_mean=float(np.mean(widths)),
        height_mean=float(np.mean(heights)),
        aspect_ratio_mean=float(np.mean(ratios)),
    )


def _resize_with_padding(image: Image.Image, target_size: int, fill_color=(255, 255, 255)) -> Image.Image:
    # Preserve aspect ratio, pad to square
    w, h = image.size
    scale = target_size / max(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    image = image.resize((new_w, new_h), Image.BILINEAR)
    new_img = Image.new("RGB", (target_size, target_size), fill_color)
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    new_img.paste(image, (paste_x, paste_y))
    return new_img


def normalize_single_image(
    src_path: Path,
    dst_root: Path,
    target_size: int,
    jpeg_quality: int,
) -> Optional[Path]:
    rel_subdir = src_path.parent.name
    dst_dir = dst_root / rel_subdir
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = dst_dir / src_path.name
    try:
        with Image.open(src_path) as im:
            im = im.convert("RGB")
            out = _resize_with_padding(im, target_size)
            out.save(dst_path, format="JPEG", quality=jpeg_quality, optimize=True)
        return dst_path
    except Exception:
        return None


def run_normalization(config: ImageNormalizationConfig) -> dict:
    config.ensure_dirs()

    all_paths = list(_iter_image_paths(config.images_dir))
    if config.max_images is not None:
        all_paths = all_paths[: config.max_images]

    before_stats = _compute_basic_stats_from_sample(all_paths, config.sample_for_metrics)

    results = {
        "total_images_found": len(all_paths),
        "normalized_success": 0,
        "normalized_failed": 0,
        "before_stats": before_stats.__dict__,
        "after_stats": None,
        "target_size": config.target_size,
        "jpeg_quality": config.jpeg_quality,
    }

    success_paths: List[Path] = []
    if config.num_workers and config.num_workers > 1:
        with ThreadPoolExecutor(max_workers=config.num_workers) as executor:
            mapped = executor.map(
                normalize_single_image,
                all_paths,
                repeat(config.normalized_images_dir),
                repeat(config.target_size),
                repeat(config.jpeg_quality),
            )
            for dst in tqdm(mapped, total=len(all_paths), desc="Normalizing images", unit="img"):
                if dst is not None:
                    success_paths.append(dst)
                    results["normalized_success"] += 1
                else:
                    results["normalized_failed"] += 1
    else:
        for p in tqdm(all_paths, desc="Normalizing images", unit="img"):
            dst = normalize_single_image(p, config.normalized_images_dir, config.target_size, config.jpeg_quality)
            if dst is not None:
                success_paths.append(dst)
                results["normalized_success"] += 1
            else:
                results["normalized_failed"] += 1

    after_stats = _compute_basic_stats_from_sample(success_paths, min(config.sample_for_metrics, len(success_paths)))
    results["after_stats"] = after_stats.__dict__

    # Additional aggregate metrics (disk usage, per-channel stats, simple histograms)
    try:
        extras = export_additional_metrics(
            before_dir=config.images_dir,
            after_dir=config.normalized_images_dir,
            plots_dir=config.plots_dir,
            sample_size=min(config.sample_for_metrics, len(success_paths)) or 0,
        )
        results.update(extras)
    except Exception:
        # Non-fatal if plotting or extra stats fail
        pass

    # Persist metrics
    metrics_path = config.metrics_dir / "normalization_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Normalize product images and compute metrics")
    parser.add_argument("--data_root", type=str, default="dataset", help="Path to dataset root containing images/")
    parser.add_argument("--output_dir", type=str, default="image_normalization/outputs", help="Output directory")
    parser.add_argument("--target_size", type=int, default=224, help="Target square size")
    parser.add_argument("--jpeg_quality", type=int, default=85, help="JPEG quality for output")
    parser.add_argument("--max_images", type=int, default=None, help="Limit number of images for processing")
    parser.add_argument("--sample_for_metrics", type=int, default=5000, help="Sample size for before/after stats")
    parser.add_argument("--num_workers", type=int, default=8, help="Parallel workers for normalization")

    args = parser.parse_args()
    cfg = ImageNormalizationConfig(
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
        target_size=args.target_size,
        jpeg_quality=args.jpeg_quality,
        max_images=args.max_images,
        sample_for_metrics=args.sample_for_metrics,
        num_workers=args.num_workers,
    )

    results = run_normalization(cfg)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()


