### Image Normalization Module

This module normalizes product images to a uniform square size while preserving aspect ratio via letterboxing (padding). It also computes before/after quality metrics to document the transformation.

#### Why normalization?
To use images as inputs for multimodal recommenders (e.g., CNN encoders), we need consistent input dimensions and pixel scaling. Normalization reduces variance due to heterogeneous resolutions and aspect ratios, improves batch efficiency, and enables reproducible training. Here, we: (1) convert all images to RGB, (2) resize with preserved aspect ratio to a fixed `target_size` using bilinear interpolation, (3) pad to square with a neutral background, and (4) save as optimized JPEG with configurable quality.

#### Usage

```bash
python -m image_normalization.normalizer \
  --data_root dataset \
  --output_dir image_normalization/outputs \
  --target_size 224 \
  --jpeg_quality 85 \
  --num_workers 8
```

Outputs:
- Normalized images in `image_normalization/outputs/normalized_images/` preserving the original subfolder structure.
- Metrics JSON: `image_normalization/outputs/metrics/normalization_metrics.json`
- Plots: `image_normalization/outputs/plots/*.png`

Para procesar TODAS las imágenes, no establezcas `--max_images` (por defecto procesa todo). Para subconjuntos de prueba usa `--max_images N`.


