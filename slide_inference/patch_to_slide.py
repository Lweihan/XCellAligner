import os
import numpy as np
import openslide
import math
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

def stitch_patches_to_svs(slide_path, patch_dir, out_path, patch_size=1024):
    """
    从 patch 生成完整的 SVS 图像并保存
    """
    # 打开原始病理切片图像
    slide = open_slide(slide_path)
    full_width, full_height = slide.dimensions
    
    # 获取所有 patch 文件并按坐标排序
    files = [f for f in os.listdir(patch_dir) if f.endswith(".png")]
    sorted_files = sorted(files, key=lambda x: extract_coords_from_filename(x))

    cols = ceil(full_width / patch_size)
    rows_imgs = []

    for r in range(ceil(full_height / patch_size)):
        row_imgs = []
        for c in range(cols):
            idx = r * cols + c
            if idx >= len(sorted_files):
                # 如果没有足够的 patch，使用白色空白图像填充
                blank = np.ones((patch_size, patch_size, 3), dtype=np.uint8) * 255
                row_imgs.append(blank)
            else:
                # 打开并裁剪每个 patch 图像
                fname = sorted_files[idx]
                img = Image.open(os.path.join(patch_dir, fname))
                img = img.crop((0, 0, min(patch_size, full_width - c * patch_size),
                                min(patch_size, full_height - r * patch_size)))
                row_imgs.append(np.array(img))  # 转换为 NumPy 数组以便拼接
        rows_imgs.append(np.concatenate(row_imgs, axis=1))  # 横向拼接

    # 纵向拼接
    big_img = np.concatenate(rows_imgs, axis=0)

    # 使用 PIL 重新创建图像对象
    big_img_pil = Image.fromarray(big_img)

    # 创建金字塔结构
    pyramid_images = []
    current_img = big_img_pil
    while current_img.width > 256 and current_img.height > 256:
        pyramid_images.append(current_img)
        current_img = current_img.resize((current_img.width // 2, current_img.height // 2), Image.ANTIALIAS)

    # 最小层（最小分辨率）
    pyramid_images.append(current_img)

    # 保存为多层金字塔 TIFF 文件
    big_img_pil.save(
        out_path,
        format="TIFF",
        save_all=True,
        append_images=pyramid_images[1:],  # 附加其他分辨率层
        compression="jpeg",
        tile=(128, 128),
        resolution=96,  # 每英寸的像素数
        dpi=(96, 96)  # 设置 DPI
    )