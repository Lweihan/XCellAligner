import os
import numpy as np
import openslide
import math
import tifffile
import pyvips
from PIL import Image
from math import ceil
from tqdm import tqdm
from module.KFBreader.kfbreader import KFBSlide

# 假设已导入 pyvips 和其他需要的库

def open_slide(slide_path):
    """
    根据文件扩展名选择合适的库来打开图像
    """
    _, ext = os.path.splitext(slide_path)
    if ext.lower() == '.kfb':
        # 使用 KFBReader 打开 .kfb 文件
        slide = KFBSlide(slide_path)
    else:
        # 使用 OpenSlide 打开其他格式的文件（如 .svs 或 .tiff）
        slide = openslide.OpenSlide(slide_path)
    return slide

def extract_coords_from_filename(filename):
    """
    从文件名中提取坐标信息
    假设文件名格式为 patch_x_y.png
    """
    coords = filename.replace(".png", "").split('_')[1:3]  # 假设文件名为 patch_x_y.png
    return int(coords[0]), int(coords[1])

def calculate_pyramid_levels(width, height, min_level_size=256):
    """
    计算需要的金字塔层级数量
    """
    levels = 1  # 至少有一层（原始层）
    current_width, current_height = width, height
    
    while current_width > min_level_size and current_height > min_level_size:
        current_width //= 2
        current_height //= 2
        levels += 1
        
    return levels

def stitch_patches_to_multilevel_tiff_alternative(slide_path, patch_dir, out_path, patch_size=1024):
    """
    替代方法：使用传统的pyvips金字塔保存
    """
    # 读取整体尺寸
    _, ext = os.path.splitext(slide_path)
    if ext.lower() in ['.tiff', '.tif']:
        with tifffile.TiffFile(slide_path) as tif:
            page = tif.pages[0]
            full_height, full_width = page.shape[0], page.shape[1]
    else:
        slide = openslide.OpenSlide(slide_path)
        full_width, full_height = slide.dimensions
        slide.close()

    print(f"Adjusted dimensions: {full_width} x {full_height}")
    
    # 获取所有 patch 并基于其坐标信息进行排序
    files = [f for f in os.listdir(patch_dir) if f.endswith(".png")]
    print(f"Found {len(files)} patches")
    sorted_files = sorted(files, key=lambda x: extract_coords_from_filename(x))

    cols = ceil(full_width / patch_size)

    rows_imgs = []
    for r in tqdm(range(ceil(full_height / patch_size)), desc="拼接行"):
        row_imgs = []
        for c in range(cols):
            idx = r * cols + c
            if idx >= len(sorted_files):
                # 用空白 patch 填充（防止最后一行/列不足）
                blank = pyvips.Image.black(patch_size, patch_size).cast("uchar")
                row_imgs.append(blank)
            else:
                fname = sorted_files[idx]
                img_path = os.path.join(patch_dir, fname)
                
                img = pyvips.Image.new_from_file(img_path)

                img = img.crop(
                    0, 0,
                    min(patch_size, full_width - c * patch_size),
                    min(patch_size, full_height - r * patch_size)
                )

                row_imgs.append(img)
        
        # 横向拼接当前行的所有patch
        if row_imgs:
            rows_imgs.append(pyvips.Image.arrayjoin(row_imgs, across=len(row_imgs)))

    # 纵向拼接所有行
    if rows_imgs:
        big_img = pyvips.Image.arrayjoin(rows_imgs, across=1)
        
        # 裁剪到原始尺寸
        big_img = big_img.crop(0, 0, full_width, full_height)
        
        print(f"Final image size: {big_img.width} x {big_img.height}")
        
        if big_img.format != pyvips.enums.BandFormat.UCHAR:
            big_img = big_img.cast("uchar")
        
        big_img.tiffsave(
            out_path,
            bigtiff=True,
            pyramid=True,
            tile=True,
            compression="jpeg",
            Q=80,
            tile_width=256,
            tile_height=256,
            # 关键：使用最近邻插值保持像素值不变
            region_shrink="nearest"
        )
        
        print(f"Successfully saved multi-level TIFF to {out_path}")