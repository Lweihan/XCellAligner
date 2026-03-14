## XCellaligner

### Introduction

Cross-modal cell alignment between hematoxylin-and-eosin (H&E) and multiplex immunofluorescence (mIF) images is crucial for leveraging protein marker–driven annotations to enhance H&E-based pathology analysis. However, H&E and mIF are typically acquired from adjacent tissue sections, which—due to physical sectioning—are inherently unregistrable and share no pixel-wise cell correspondence. To address this gap, we propose XCellAligner, a novel framework designed for cell-level semantic alignment between H&E and mIF images. Our approach avoids the need for manual annotation by first aligning modality-level representations using contrastive learning on aggregated cell tokens. It then refines the cell-to-cell correspondence through fine-grained matching, utilizing Hungarian loss guided by the discriminative features of mIF. XCellAligner is fully unsupervised, and its extracted cellular features exhibit high discriminative power, achieving state-of-the-art performance in downstream tasks such as classification, segmentation, and visual question answering. Furthermore, this method demonstrates strong generalization ability to previously unseen tumor types and tissue sites, providing a practical foundation for tumor microenvironment analysis.

![demo](experuments/generate.png)

### Installation
```shell
conda env create -f environment.yml
```
If you plan to infer .kfb images, please note that the libp11-kit.so file used by KFBReader is incompatible with Python 3.10. A solution can be found in this [blog](https://blog.csdn.net/qq_38606680/article/details/129118491). 

### Usage

#### Patch Inference
```shell
python he_transformer_inference.py --image_path <Image Path> --model_path <Weight Path> --save_path <Save Path> --k <Cluster Number>
```

#### Whole Slide Inference
```shell
python slide_inference.py --slide_path <Slide Path> --model_path <Weight Path> --temp_path <Temporary File Storage Path> --output_path <Cluster Output Path> --type <Organ Class> --k <Cluster Number>
```

### Training

#### Training Data Preparation

### Slide Data Format

The recommended raw format is as follows, but it is not a limitation; it can be processed into a specified patch data format.

```shell
origin_data
|-- mIF
|   `-- mIF_slide_channel0(.tif)
|   `-- ...
|   `-- mIF_slide_channel{Channel_Num - 1}(.tif)
`-- he_slide(.svs/.tiff/tif/.kfb)

```

##### Coarse Registration & Patch Data Format

For instructions on how to perform coarse registration, please refer to [registration instruction](coarse_registration/registration_instruction.ipynb).

After registration is completed, a dataset consisting of patches is obtained, in the following format:
```shell
data
|-- he
|   `-- he_x0_y0.png
|   `-- ...
|   `-- he_x{X_Coords}_y{Y_Coords}.png
`-- mIF
    |-- 0-Hoechst
    |   `-- mF0_x0_y0.png
    |   `-- ...
    |   `-- mF0_x{X_Coords}_y{Y_Coords}.png
    `-- {Channel_ID}-{Channel_Name}
        `-- mF{Channel_ID}_x0_y0.png
        `-- ...
        `-- mF{Channle_ID}_x{X_Coords}_y{Y_Coords}.png
```

#### Feature Cache

Please execute the `pre_extract_features.py` file with the following command. Note that you should create a `cache-mGPU` folder containing two subfolders named `he` and `mif`.
```shell
python pre_extract_features.py \
--he_dir /data/he \
--mif_dir /data/mIF \
--cache_dir /data/cache-mGPU \
--log_dir /data/log
```

#### Alignment Training

Please execute the `multidata_aligner_trainer.py` file.

If there is only one dataset, the reference command is as follows:
```shell
python multidata_aligner_trainer.py \
--cache_dir \
  /data/cache-mGPU \
--start_index 0 \
--mif_channel channel_num \
--output_dir /output \
--batch_size 64 \
--epochs 350 \
--lambda_contrast 0.01
```

If multiple datasets exist, the reference command is as follows:
```shell
python multidata_aligner_trainer.py \
--cache_dir \
  /data/dataset1/cache-mGPU \
  /data/dataset2/cache-mGPU \
  /data/dataset3/cache-mGPU \
--start_index 0 {Channel_Num1} {Channel_Num1+Channel_Num2} \
--mif_channel {Channel_Num1} {Channel_Num2} {Channel_Num3} \
--output_dir /output \
--batch_size 64 \
--epochs 350 \
--lambda_contrast 0.01
```
