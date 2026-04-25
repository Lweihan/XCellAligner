import os
import argparse
import pickle
import torch
import torch.nn.functional as F
import numpy as np
import tifffile
import zarr
from PIL import Image
from tqdm import tqdm
import logging
import scipy.ndimage as ndimage
from utils import load_cellpose_model
from module.TransPath.ctran import ctranspath
from module.ModalEncoder.cell_density_extractor import CellDensityExtractor

# =========================
# 配置与日志
# =========================
def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("WSI_Stream")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(log_dir, "stream_extract.log"))
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def is_patch_valid(he_patch, black_threshold=15, white_threshold=240, std_threshold=5.0, black_area_threshold=0.6):
    if he_patch.size == 0: return False
    mean_val = he_patch.mean()
    if mean_val < black_threshold or mean_val > white_threshold: return False
    if he_patch.std() < std_threshold: return False
    # 统计背景比例
    gray = he_patch.mean(axis=-1)
    black_ratio = np.sum(gray < black_threshold) / gray.size
    if black_ratio > black_area_threshold: return False
    return True

def adaptive_load_ctranspath(model, weight_path, device):
    """
    CTransPath 权重适配器 (timm 0.5.4 -> 0.9.x)
    """
    logger = logging.getLogger("WSI_Stream")
    checkpoint = torch.load(weight_path, map_location=device)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        if "layers.0.downsample" in k:
            new_k = k.replace("layers.0.downsample", "layers.1.downsample")
        elif "layers.1.downsample" in k:
            new_k = k.replace("layers.1.downsample", "layers.2.downsample")
        elif "layers.2.downsample" in k:
            new_k = k.replace("layers.2.downsample", "layers.3.downsample")
        new_state_dict[new_k] = v
    
    from module.TransPath.ctran import ConvStem
    model.patch_embed = ConvStem(img_size=224, patch_size=4, in_chans=3, embed_dim=model.embed_dim)
    
    model.load_state_dict(new_state_dict, strict=False)
    
    class GlobalPoolHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
        def forward(self, x):
            if x.dim() == 4:
                x = x.permute(0, 3, 1, 2)
                x = self.pool(x)
            return torch.flatten(x, 1)

    model.head = GlobalPoolHead()
    model.to(device)
    logger.info(f"CTransPath weights adapted.")
    return model

# =========================
# 核心提取逻辑 (优化版)
# =========================
def main():
    parser = argparse.ArgumentParser(description="WSI Stream Feature Extraction (Optimized)")
    parser.add_argument("--he", type=str, required=True, help="Path to HE ome.tif")
    parser.add_argument("--mif", type=str, required=True, help="Path to mIF ome.tiff")
    parser.add_argument("--cache_dir", type=str, required=True, help="Output cache directory")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Log directory")
    parser.add_argument("--size", type=int, default=512, help="Patch size")
    parser.add_argument("--weights", type=str, default="./module/checkpoint/ctranspath.pth", help="CTransPath weights")
    parser.add_argument("--batch_size", type=int, default=64, help="Cell inference batch size")
    args = parser.parse_args()

    logger = setup_logger(args.log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载模型
    logger.info("Loading models...")
    cellpose_model = load_cellpose_model(model_type='/home/zyh/.cellpose/models/cpsam', device=device)
    
    if not os.path.exists(args.weights):
        logger.error(f"Weights not found at {args.weights}!")
        return
        
    ctp_model = ctranspath().to(device)
    ctp_model = adaptive_load_ctranspath(ctp_model, args.weights, device)
    ctp_model.eval()
    
    mif_extractor = CellDensityExtractor()
    
    # 2. 打开 WSI
    logger.info("Opening WSIs...")
    he_z = zarr.open(tifffile.imread(args.he, aszarr=True), mode='r')['0']
    mif_z = zarr.open(tifffile.imread(args.mif, aszarr=True), mode='r')['0']
    H, W = he_z.shape[0], he_z.shape[1]
    num_mif_channels = mif_z.shape[0]
    
    # 3. 准备输出目录
    he_cache_dir = os.path.join(args.cache_dir, "he")
    mif_cache_dir = os.path.join(args.cache_dir, "mif")
    he_img_dir = os.path.join(args.cache_dir, "he_images")
    os.makedirs(he_cache_dir, exist_ok=True)
    os.makedirs(mif_cache_dir, exist_ok=True)
    os.makedirs(he_img_dir, exist_ok=True)

    coords = []
    for y in range(0, H - args.size, args.size):
        for x in range(0, W - args.size, args.size):
            coords.append((x, y))
    
    logger.info(f"Total potential patches: {len(coords)}. Starting extraction...")

    # 标准化常量 (移至 GPU)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    for x, y in tqdm(coords, desc="Streaming Extraction"):
        he_pkl = os.path.join(he_cache_dir, f"he_x{x}_y{y}.pkl")
        mif_pkl = os.path.join(mif_cache_dir, f"mif_x{x}_y{y}.pkl")
        
        if os.path.exists(he_pkl) and os.path.exists(mif_pkl):
            continue
            
        he_patch = he_z[y:y+args.size, x:x+args.size, :]
        if not is_patch_valid(he_patch):
            continue
            
        # Cellpose 分割
        try:
            cp_results = cellpose_model.eval(he_patch, diameter=18, channels=[0, 0])
            masks = cp_results[0]
            unique_labels = np.unique(masks)
            unique_labels = unique_labels[unique_labels != 0]
            if len(unique_labels) == 0: continue
        except Exception as e:
            logger.error(f"Cellpose error at x={x}, y={y}: {e}")
            continue
        
        # 特征提取
        try:
            # 1. 边缘过滤 (使用 scipy 优化)
            margin = 20
            centers = ndimage.center_of_mass(np.ones_like(masks), labels=masks, index=unique_labels)
            
            valid_labels = []
            for label, (cy, cx) in zip(unique_labels, centers):
                if margin <= cx < args.size - margin and margin <= cy < args.size - margin:
                    valid_labels.append(label)
            
            if not valid_labels: continue
            valid_labels = valid_labels[:255] # 限制数量
            
            # 2. 准备 GPU 上的 HE Patch 和 Mask
            he_tensor = torch.from_numpy(he_patch).permute(2, 0, 1).float().to(device) / 255.0
            masks_tensor = torch.from_numpy(masks).to(device)
            
            # 3. 批量生成输入 (保持原有的“黑底背景”逻辑)
            # 我们通过一次性在 GPU 上生成所有 mask 来加速
            # 这里如果 N 很大，为了显存安全，可以分小批
            all_input_tensors = []
            for label in valid_labels:
                cell_mask = (masks_tensor == label).float().unsqueeze(0) # [1, H, W]
                masked_patch = he_tensor * cell_mask # [3, H, W]
                # 调整到模型输入尺寸
                input_t = F.interpolate(masked_patch.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
                all_input_tensors.append(input_t)
            
            input_batch = torch.cat(all_input_tensors, dim=0)
            input_batch = (input_batch - mean) / std # 标准化
            
            # 4. 批量推理
            cell_features_list = []
            with torch.no_grad():
                for i in range(0, input_batch.size(0), args.batch_size):
                    batch_out = ctp_model(input_batch[i : i + args.batch_size])
                    cell_features_list.append(batch_out.cpu().numpy())
            
            cell_features = np.concatenate(cell_features_list, axis=0)

            # 保存 HE PKL
            feat_arr = np.zeros((255, 768), dtype=np.float32)
            mask_arr = np.zeros(255, dtype=np.float32)
            n_cells = cell_features.shape[0]
            feat_arr[:n_cells] = cell_features
            mask_arr[:n_cells] = 1.0
            
            # 构造过滤后的 mask 数组 (仅包含 valid_labels)
            filtered_masks = np.zeros_like(masks)
            for label in valid_labels:
                filtered_masks[masks == label] = label

            with open(he_pkl, "wb") as f:
                pickle.dump({"features": torch.from_numpy(feat_arr).unsqueeze(0), 
                            "mask": torch.from_numpy(mask_arr).unsqueeze(0), 
                            "cell_masks": filtered_masks}, f)
            
            Image.fromarray(he_patch).save(os.path.join(he_img_dir, f"he_x{x}_y{y}.png"))

            # D. mIF 密度提取
            mif_patch = mif_z[:, y:y+args.size, x:x+args.size]
            mif_list = [mif_patch[c] for c in range(num_mif_channels)]
            # 密度提取也可以受益于 filtered_masks
            density = mif_extractor.process_image_pair(mif_list, [0] + [1]*(num_mif_channels-1), filtered_masks)
            
            mif_feat_arr = np.zeros((255, num_mif_channels), dtype=np.float32)
            mif_mask_arr = np.zeros(255, dtype=np.float32)
            n_mif = min(255, density.shape[0])
            mif_feat_arr[:n_mif] = density[:n_mif]
            mif_mask_arr[:n_mif] = 1.0
            
            with open(mif_pkl, "wb") as f:
                pickle.dump({"features": torch.from_numpy(mif_feat_arr).unsqueeze(0), 
                            "mask": torch.from_numpy(mif_mask_arr).unsqueeze(0)}, f)

        except Exception as e:
            import traceback
            logger.error(f"Error at x={x}, y={y}: {e}")
            logger.error(traceback.format_exc())
            continue

    logger.info(f"All done! Features saved to {args.cache_dir}")

if __name__ == "__main__":
    main()
