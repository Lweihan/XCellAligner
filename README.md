# XCellAligner

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Segmentation: Cellpose](https://img.shields.io/badge/Segmentation-Cellpose-2ea44f)](https://www.cellpose.org/)
[![Segmentation: InstanSeg](https://img.shields.io/badge/Segmentation-InstanSeg-1f6feb)](https://github.com/instanseg/instanseg)
[![Registration: SimpleITK](https://img.shields.io/badge/Registration-SimpleITK-ff7f0e)](https://simpleitk.org/)
[![I/O: tiffslide](https://img.shields.io/badge/WSI%20I%2FO-tiffslide-6f42c1)](https://pypi.org/project/tiffslide/)

**A cross-modal cell alignment framework for H&E and mIF pathology analysis**

[Features](#features) · [Architecture](#architecture) · [Quick Start](#quick-start) · [Project Structure](#project-structure) · [Configuration](#configuration) · [Contributing](#contributing) · [License](#license)

## Dependency Jump Links

Use these icon-style links for quick reference to core components:

[![Cellpose](https://img.shields.io/badge/Cellpose-Docs-2ea44f?logo=python&logoColor=white)](https://www.cellpose.org/)
[![InstanSeg](https://img.shields.io/badge/InstanSeg-GitHub-1f6feb?logo=github&logoColor=white)](https://github.com/instanseg/instanseg)
[![SimpleITK](https://img.shields.io/badge/SimpleITK-Website-ff7f0e)](https://simpleitk.org/)
[![tiffslide](https://img.shields.io/badge/tiffslide-PyPI-6f42c1?logo=pypi&logoColor=white)](https://pypi.org/project/tiffslide/)
[![slideio](https://img.shields.io/badge/slideio-PyPI-0052cc?logo=pypi&logoColor=white)](https://pypi.org/project/slideio/)
[![bioio](https://img.shields.io/badge/bioio-PyPI-0a7ea4?logo=pypi&logoColor=white)](https://pypi.org/project/bioio/)

XCellAligner is an unsupervised framework for cell-level semantic alignment between H&E and mIF images from adjacent tissue sections.

## Features

1. Unsupervised cross-modal alignment between H&E and mIF.
2. Cell-level semantic matching with strong transferability.
3. Dual segmentation modes in CellEngine:
  quality mode with Cellpose and efficiency mode with InstanSeg.
4. Detail-controlled logging with detail false/true.
5. End-to-end pipeline support from registration and patching to training and slide inference.

## Overview

Cross-modal cell alignment between hematoxylin-and-eosin (H&E) and multiplex immunofluorescence (mIF) is important for transferring marker-driven biological knowledge to H&E-based pathology tasks.

XCellAligner follows two key stages:

1. Modality-level alignment with contrastive learning on aggregated cell tokens.
2. Fine-grained cell matching with Hungarian-style supervision guided by mIF discriminative features.

This enables high-quality transferable cell representations for downstream tasks such as classification and segmentation.

![Pipeline Preview](https://github.com/Lweihan/XCellAligner/blob/main/experiments/generate.png)

## Architecture

XCellAligner follows a modular architecture:

1. Coarse registration and patch generation.
2. Cell segmentation and morphology-aware feature extraction.
3. Contrastive alignment and fine-grained cell matching.
4. Downstream inference on patches and whole-slide images.

For visual guidance, see:

1. [experiments/generate.png](experiments/generate.png)
2. [coarse_registration/registration_instruction.ipynb](coarse_registration/registration_instruction.ipynb)

## Component Guide Links (Diagram Style)

Use the following links as visual guidance entry points when onboarding the project:

1. Coarse registration walkthrough notebook: [coarse_registration/registration_instruction.ipynb](coarse_registration/registration_instruction.ipynb)
2. Whole-slide inference entry script: [slide_inference.py](slide_inference.py)
3. Alignment training entry script: [multidata_aligner_trainer.py](multidata_aligner_trainer.py)

If you maintain external architecture flowcharts (for example, Draw.io/Figma/Mermaid), append them here for team-wide standardized access.

## Project Structure

Main directories and scripts:

1. `coarse_registration/`: coarse alignment and patch extraction utilities.
2. `dataset/`: dataset wrappers and data loading logic.
3. `module/`: encoders and backbones (including CTransPath utilities).
4. `slide_inference/`: whole-slide processing helpers.
5. `pre_extract_features.py`: feature cache generation.
6. `multidata_aligner_trainer.py`: multi-dataset alignment training.
7. `CellEngine.py`: inference engine with `mode` and `detail` controls.

## Installation

```bash
conda env create -f environment.yml
conda activate aligner
```

Note for `.kfb` inference: the `libp11-kit.so` issue may appear with Python 3.10 in some environments. See the workaround in this blog:
[https://blog.csdn.net/qq_38606680/article/details/129118491](https://blog.csdn.net/qq_38606680/article/details/129118491)

## Configuration

Common runtime options are managed in the inference scripts and CellEngine:

1. mode:
  quality for Cellpose segmentation, efficiency for InstanSeg segmentation.
2. detail:
  false for silent mode, true for verbose logs.
3. environment:
  use [environment.yml](environment.yml) for reproducible setup.

## Quick Start

### Patch Inference

```bash
python he_transformer_inference.py --image_path <IMAGE_PATH> --model_path <WEIGHT_PATH> --save_path <SAVE_PATH> --k <CLUSTER_NUM>
```

### Whole-Slide Inference

```bash
python slide_inference.py --slide_path <SLIDE_PATH> --model_path <WEIGHT_PATH> --temp_path <TEMP_DIR> --output_path <OUTPUT_DIR> --type <ORGAN_CLASS> --k <CLUSTER_NUM>
```

### CellInferenceEngine Modes

`CellEngine.py` supports two segmentation modes:

1. `mode="quality"`: keep the original Cellpose workflow.
2. `mode="efficiency"`: use InstanSeg-based segmentation and batched feature extraction.

It also supports verbosity control:

1. `detail=False` (default): silent mode, no logs.
2. `detail=True`: print detailed processing logs.

## Training

### 1) Raw Slide Format

Recommended raw data layout:

```text
origin_data
|-- mIF
|   |-- mIF_slide_channel0.tif
|   |-- ...
|   `-- mIF_slide_channel{Channel_Num-1}.tif
`-- he_slide(.svs/.tiff/.tif/.kfb)
```

### 2) Coarse Registration and Patch Format

Follow: [coarse_registration/registration_instruction.ipynb](coarse_registration/registration_instruction.ipynb)

Expected patch-level structure:

```text
data
|-- he
|   |-- he_x0_y0.png
|   |-- ...
|   `-- he_x{X}_y{Y}.png
`-- mIF
    |-- 0-Hoechst
    |   |-- mF0_x0_y0.png
    |   |-- ...
    |   `-- mF0_x{X}_y{Y}.png
    `-- {Channel_ID}-{Channel_Name}
        |-- mF{Channel_ID}_x0_y0.png
        |-- ...
        `-- mF{Channel_ID}_x{X}_y{Y}.png
```

### 3) Feature Cache Extraction

Create `cache-mGPU/he` and `cache-mGPU/mif` first, then run:

```bash
python pre_extract_features.py \
  --he_dir /data/he \
  --mif_dir /data/mIF \
  --cache_dir /data/cache-mGPU \
  --log_dir /data/log
```

### 4) Alignment Training

Single dataset:

```bash
python multidata_aligner_trainer.py \
  --cache_dir /data/cache-mGPU \
  --start_index 0 \
  --mif_channel <CHANNEL_NUM> \
  --output_dir /output \
  --batch_size 64 \
  --epochs 350 \
  --lambda_contrast 0.01
```

Multiple datasets:

```bash
python multidata_aligner_trainer.py \
  --cache_dir /data/dataset1/cache-mGPU /data/dataset2/cache-mGPU /data/dataset3/cache-mGPU \
  --start_index 0 <CHANNEL_NUM1> <CHANNEL_NUM1+CHANNEL_NUM2> \
  --mif_channel <CHANNEL_NUM1> <CHANNEL_NUM2> <CHANNEL_NUM3> \
  --output_dir /output \
  --batch_size 64 \
  --epochs 350 \
  --lambda_contrast 0.01
```

## License

This project is licensed under the MIT License.

See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome.

1. Fork the repository and create a feature branch.
2. Keep code style and README docs consistent.
3. Submit a pull request with clear change description and usage notes.
