from pathlib import Path
from typing import Optional, Union


class ImageNormalizationConfig:
    """
    Configuration for image normalization.
    """

    def __init__(
        self,
        data_root: Union[Path, str] = Path("dataset"),
        images_subdir: str = "images",
        output_dir: Union[Path, str] = Path("image_normalization") / "outputs",
        target_size: int = 224,
        jpeg_quality: int = 85,
        max_images: Optional[int] = None,
        num_workers: int = 8,
        sample_for_metrics: int = 5000,
    ) -> None:
        self.data_root = Path(data_root)
        self.images_dir = self.data_root / images_subdir
        self.output_dir = Path(output_dir)
        self.normalized_images_dir = self.output_dir / "normalized_images"
        self.metrics_dir = self.output_dir / "metrics"
        self.plots_dir = self.output_dir / "plots"
        self.target_size = target_size
        self.jpeg_quality = jpeg_quality
        self.max_images = max_images
        self.num_workers = max(1, int(num_workers))
        self.sample_for_metrics = sample_for_metrics

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.normalized_images_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)


