## XCellaligner

### Installation
```shell
conda env create -f environment.yml
cd module/nnUNet
pip install -e .
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

#### Training Data Format


#### H&E Transformer Training
```shell
python he_transformer_train.py --data_root <Data Path> --save_path <Weight Save Path> --epochs <Epochs> --batch_size <Batch Size> --lr <Learning Rate>
```

#### Modal Transformer Training
```shell
python modal_transformer_train.py --data_root <Data Path> --save_path <Weight Save Path> --epochs <Epochs> --batch_size <Batch Size> --lr <Learning Rate>
```

#### Transformer Alignment
```shell
python he_modal_alignment.py --he_model_path <H&E Model Path> --modal_model_path <Modal Model Path> --he_data_root <H&E Data Path> --mif_data_root <Modal Data Path> --save_path <Weight Save Path> --epochs <Epochs> --lr <Learning Rate>
```
