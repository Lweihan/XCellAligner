import os
import argparse
import numpy as np
import tifffile
import zarr
from PIL import Image
from tqdm import tqdm
import concurrent.futures
import sys

# 设置PIL允许加载非常大的图像
Image.MAX_IMAGE_PIXELS = None

def is_patch_valid(he_patch: np.array, black_threshold=15, white_threshold=240, std_threshold=5.0, black_area_threshold=0.6):
    """判断 HE patch 是否有效（非纯黑/纯白/无纹理）"""
    if he_patch.size == 0:
        return False
    
    mean_val = he_patch.mean()
    std_val = he_patch.std()
    
    # 判断是否是近乎纯黑或纯白
    if mean_val < black_threshold or mean_val > white_threshold:
        return False
    
    # 判断标准差，排除完全无纹理的背景
    if std_val < std_threshold:
        return False
    
    # 判断黑色/背景区域比例
    # 对于多通道大图，有时背景是全0
    black_area = np.sum(he_patch.mean(axis=-1) < black_threshold) / (he_patch.shape[0] * he_patch.shape[1])
    if black_area > black_area_threshold:
        return False
    
    return True

def process_single_patch(x, y, patch_size, he_path, mif_path, output_dir):
    """单个进程处理一个位置的裁剪任务"""
    try:
        # 在子进程中重新打开 zarr 以确保线程安全/进程安全
        he_z = zarr.open(tifffile.imread(he_path, aszarr=True), mode='r')['0']
        mif_z = zarr.open(tifffile.imread(mif_path, aszarr=True), mode='r')['0']
        
        # 1. 裁剪并验证 HE
        he_patch = he_z[y:y+patch_size, x:x+patch_size, :]
        if not is_patch_valid(he_patch):
            return False
            
        # 2. 保存 HE
        he_out_dir = os.path.join(output_dir, "he")
        os.makedirs(he_out_dir, exist_ok=True)
        he_name = f"he_x{x}_y{y}.png"
        Image.fromarray(he_patch).save(os.path.join(he_out_dir, he_name))
        
        # 3. 裁剪并保存 mIF 19个通道
        # mif_z shape: (C, H, W)
        num_channels = mif_z.shape[0]
        for c in range(num_channels):
            mif_ch_dir = os.path.join(output_dir, "mif", f"{c}-Ch{c}")
            os.makedirs(mif_ch_dir, exist_ok=True)
            
            mif_patch = mif_z[c, y:y+patch_size, x:x+patch_size]
            # mIF 通常是 uint16，存 PNG 前需要缩放到 uint8 以便后续 CellDensityExtractor 处理
            # 或者保持 uint16 存为 PNG (需要某些库支持)，这里建议缩放到 uint8 提高兼容性
            # 但为了保留原始精度供统计，我们先检查一下 extractor 是否支持 uint16
            
            # 简单线性拉伸到 0-255
            m_min, m_max = mif_patch.min(), mif_patch.max()
            if m_max > m_min:
                mif_patch_u8 = ((mif_patch - m_min) / (m_max - m_min) * 255).astype(np.uint8)
            else:
                mif_patch_u8 = np.zeros_like(mif_patch, dtype=np.uint8)
                
            mif_name = f"mF{c}_x{x}_y{y}.png"
            Image.fromarray(mif_patch_u8).save(os.path.join(mif_ch_dir, mif_name))
            
        return True
    except Exception as e:
        print(f"Error at x={x}, y={y}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Crop OME-TIFF to PNG patches")
    parser.add_argument("--he", type=str, required=True, help="Path to HE ome.tif")
    parser.add_argument("--mif", type=str, required=True, help="Path to mIF ome.tiff")
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    parser.add_argument("--size", type=int, default=512, help="Patch size")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers")
    args = parser.parse_args()
    
    print("Opening WSIs...")
    he_z = zarr.open(tifffile.imread(args.he, aszarr=True), mode='r')['0']
    mif_z = zarr.open(tifffile.imread(args.mif, aszarr=True), mode='r')['0']
    
    H, W = he_z.shape[0], he_z.shape[1]
    print(f"Full image size: {W} x {H}")
    
    # 构造任务坐标
    coords = []
    for y in range(0, H - args.size, args.size):
        for x in range(0, W - args.size, args.size):
            coords.append((x, y))
            
    print(f"Total potential patches: {len(coords)}")
    
    # 并发处理
    success_count = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_single_patch, x, y, args.size, args.he, args.mif, args.out) for x, y in coords]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Cropping"):
            if future.result():
                success_count += 1
                
    print(f"Finished! Saved {success_count} valid patches to {args.out}")

if __name__ == "__main__":
    main()
