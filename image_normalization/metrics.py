from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_metrics(path: Union[Path, str]) -> Dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_dataset_size_on_disk(images_root: Path) -> int:
    total = 0
    for p in images_root.rglob("*.jpg"):
        try:
            total += p.stat().st_size
        except Exception:
            continue
    return total


def plot_histogram(values: List[float], title: str, save_path: Path, bins: int = 50) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=bins, color="#4C78A8", alpha=0.85)
    plt.title(title)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_disk_usage(bytes_before: int, bytes_after: int, save_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    labels = ["Before", "After"]
    sizes_mb = [bytes_before / (1024 * 1024 + 1e-9), bytes_after / (1024 * 1024 + 1e-9)]
    colors = ["#9ecae1", "#3182bd"]
    plt.bar(labels, sizes_mb, color=colors)
    plt.ylabel("Size (MB)")
    plt.title("Dataset size on disk")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_channel_stats(channel_before: dict, channel_after: dict, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    # Means
    means_b = channel_before.get("mean", [0, 0, 0])
    means_a = channel_after.get("mean", [0, 0, 0])
    x = np.arange(3)
    width = 0.35
    plt.figure(figsize=(7, 4))
    plt.bar(x - width / 2, means_b, width, label="Before", color="#a1d99b")
    plt.bar(x + width / 2, means_a, width, label="After", color="#31a354")
    plt.xticks(x, ["R", "G", "B"]) 
    plt.ylim(0, 1)
    plt.title("Per-channel mean (0-1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "channel_means_before_after.png")
    plt.close()

    # Stds
    stds_b = channel_before.get("std", [0, 0, 0])
    stds_a = channel_after.get("std", [0, 0, 0])
    plt.figure(figsize=(7, 4))
    plt.bar(x - width / 2, stds_b, width, label="Before", color="#fdd0a2")
    plt.bar(x + width / 2, stds_a, width, label="After", color="#e6550d")
    plt.xticks(x, ["R", "G", "B"]) 
    plt.ylim(0, 0.6)
    plt.title("Per-channel std (0-1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "channel_stds_before_after.png")
    plt.close()


def export_additional_metrics(before_dir: Path, after_dir: Path, plots_dir: Path, sample_size: int = 5000) -> dict:
    # Compute on-disk sizes
    before_size = compute_dataset_size_on_disk(before_dir)
    after_size = compute_dataset_size_on_disk(after_dir)

    # Sample a subset and compute per-channel means/stds
    def sample_images(paths: List[Path], n: int) -> List[Path]:
        if len(paths) <= n:
            return paths
        idx = np.random.choice(len(paths), size=n, replace=False)
        return [paths[i] for i in idx]

    before_paths = list(before_dir.rglob("*.jpg"))
    after_paths = list(after_dir.rglob("*.jpg"))

    before_paths_s = sample_images(before_paths, sample_size)
    after_paths_s = sample_images(after_paths, sample_size)

    def channel_stats(paths: List[Path]) -> dict:
        if not paths:
            return {"mean": [0, 0, 0], "std": [0, 0, 0]}
        means = []
        stds = []
        for p in paths:
            try:
                with Image.open(p) as im:
                    im = im.convert("RGB")
                    arr = np.asarray(im, dtype=np.float32) / 255.0
                    means.append(arr.reshape(-1, 3).mean(axis=0))
                    stds.append(arr.reshape(-1, 3).std(axis=0))
            except Exception:
                continue
        if not means:
            return {"mean": [0, 0, 0], "std": [0, 0, 0]}
        means = np.stack(means)
        stds = np.stack(stds)
        return {"mean": means.mean(axis=0).tolist(), "std": stds.mean(axis=0).tolist()}

    before_channel = channel_stats(before_paths_s)
    after_channel = channel_stats(after_paths_s)

    # Save simple histograms for documentation
    def collect_means(paths: List[Path]) -> List[float]:
        vals = []
        for p in paths:
            try:
                with Image.open(p) as im:
                    arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
                    vals.append(float(arr.mean()))
            except Exception:
                continue
        return vals

    plot_histogram(collect_means(before_paths_s), "Before normalization: mean intensity", plots_dir / "before_mean_hist.png")
    plot_histogram(collect_means(after_paths_s), "After normalization: mean intensity", plots_dir / "after_mean_hist.png")

    # Aspect ratio histogram (before)
    def collect_aspect_ratios(paths: List[Path]) -> List[float]:
        vals = []
        for p in paths:
            try:
                with Image.open(p) as im:
                    w, h = im.size
                    if h > 0:
                        vals.append(w / h)
            except Exception:
                continue
        return vals

    plot_histogram(
        collect_aspect_ratios(before_paths_s),
        "Before normalization: aspect ratio (w/h)",
        plots_dir / "before_aspect_ratio_hist.png",
        bins=60,
    )

    # Disk usage comparison and channel stats plots
    plot_disk_usage(before_size, after_size, plots_dir / "disk_usage_before_after.png")
    plot_channel_stats(before_channel, after_channel, plots_dir)

    return {
        "bytes_before": before_size,
        "bytes_after": after_size,
        "compression_ratio": float(after_size) / float(before_size) if before_size > 0 else 0.0,
        "channel_stats_before": before_channel,
        "channel_stats_after": after_channel,
    }


