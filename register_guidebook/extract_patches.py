#!/usr/bin/env python3

"""
Patch 提取工具 - 应用配准变换参数从 HE 和 mIF 原图提取 patch（带二次精细配准）
先判断 HE patch 是否有效，再保存 HE 和 mIF patches

# 处理单个病例
python extract_patches.py \
  --he_path "/nfs-medical3/sxz/XCellAligner/data/he_rotated/7F 25-015162 5_rotated.tiff" \
  --transform_dir "/nfs-medical3/sxz/XCellAligner/data/mif_registered/7F 25-015162 5 520-HER2+670-P53+570-KI67" \
  --output_dir "/nfs-medical3/sxz/XCellAligner/data/patches_test"\

# 文件夹模式
python extract_patches.py \
  --he_dir "/nfs-medical3/sxz/XCellAligner/data/he_rotated" \
  --transform_root "/nfs-medical3/sxz/XCellAligner/data/mif_registered" \
  --output_dir "/nfs-medical3/sxz/XCellAligner/data/patches" \

  
#根据不同数据需要修改数据读取方式：
1、CHANNEL_MAP
2、def parse_mif_folder_name
3、def find_mif_files

"""

import os
import pickle
import json
import argparse
import sys
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import SimpleITK as sitk
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from skimage.color import rgb2hed
from skimage.filters import threshold_otsu
from scipy.signal import correlate2d
import logging
import torch
import torch.nn.functional as F

# 允许处理更大的图像
Image.MAX_IMAGE_PIXELS = None

# 允许加载被截断的图像
Image.OPEN_IGNORE_TRUNCATED_IMAGES = True

# 检查 CUDA 是否可用
CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# =========================
# Logging 配置
# =========================
def setup_logger(log_dir):
    """设置日志记录器"""
    os.makedirs(log_dir, exist_ok=True)

    tag = "main"
    log_file = os.path.join(log_dir, f"extract_patches_{tag}.log")

    logger = logging.getLogger(tag)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "[%(asctime)s][%(levelname)s][PID:%(process)d] %(message)s"
    )

    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger


# 通道名称映射（从 channel.txt 读取）
# 键是通道编号（字符串），值是标准通道名
CHANNEL_MAP = {
    '0': 'DAPI',
    '1': 'CD34',
    '2': 'PANCK',
    '3': 'VIMENTIN',
    '4': 'CEA',
    '5': 'COX2',
    '6': 'MUC5AC',
    '7': 'HER2',
    '8': 'P53',
    '9': 'KI67',
    '10': 'MPO',
    '11': 'CD138',
    '12': 'D240',
    '13': 'MUC6',
    '14': '18.2',
    '15': 'SYN'
}

# 通道名别名映射（处理拼写变体）
# 格式: {别名: 标准名}
# 检测到这些名称时会自动转换为标准名


# =========================
# 工具函数
# =========================

def blank_ratio(pil_img):
    """计算空白区域比例"""
    arr = np.array(pil_img)
    total = arr.size // 3
    mask1 = np.all(arr >= 220, axis=-1)
    avg = np.mean(arr, axis=-1)
    rng = np.ptp(arr, axis=-1)
    mask2 = (avg >= 215) & (rng <= 15)
    return np.sum(mask1 | mask2) / total


def black_ratio(pil_img, thr=15):
    """计算黑暗区域比例"""
    gray = np.array(pil_img.convert("L"))
    return (gray <= thr).sum() / gray.size


def parse_mif_folder_name(mif_folder_name: str) -> List[Tuple[str, str]]:
    """从 mIF 文件夹名称解析通道信息

    文件夹格式: "7C 25-064821 2 520-MPO+670-CD138+570-D240"
             或 "520-MUC6+670-18.2+570-SYN"

    返回: [(波长, 通道名), ...]
    例如: [('520', 'MPO'), ('670', 'CD138'), ('570', 'D240')]

    注意：这里返回的是波长和通道名，而不是 channel_id
    channel_id 需要通过通道名从 CHANNEL_MAP 反向查找
    """
    # 找到第一个连续的 "数字-字母" 模式（这是第一个通道的开始）
    # 例如：从 "7C 25-064821 2 520-MPO+..." 中找到 "520-MPO" 的开始位置
    match = re.search(r'\d{3}-[A-Z0-9.]+', mif_folder_name)
    if match:
        # 从第一个通道开始截取字符串
        channel_part = mif_folder_name[match.start():]
    else:
        # 如果没找到，使用整个字符串
        channel_part = mif_folder_name

    parts = channel_part.split('+')
    channels = []

    for part in parts:
        # 修复正则表达式，支持小数点（如 18.2）
        match = re.search(r'(\d+)-([A-Z0-9.]+)', part.upper())
        if match:
            wavelength = match.group(1)  # 波长，如 "520", "670", "570"
            name = match.group(2)  # 通道名，如 "MUC6", "18.2", "SYN"
            channels.append((wavelength, name))

    return channels


def find_mif_files(mif_folder: str) -> Dict[str, str]:
    """获取 mIF 文件夹中的所有通道文件

    返回: {波长: 文件路径} 的映射
    例如: {'DAPI': '/path/to/DAPI.tif', '520': '/path/to/TSA-520-CEA.tif', ...}
    """
    mif_files = {}

    mif_path = Path(mif_folder)
    if not mif_path.exists():
        return mif_files

    for file in mif_path.iterdir():
        if file.is_file() and (file.suffix.upper() in ['.TIF', '.TIFF']):
            name_upper = file.name.upper()

            # DAPI
            if 'DAPI' in name_upper:
                mif_files['DAPI'] = str(file)
            # TSA 通道 - 动态提取波长
            elif 'TSA' in name_upper:
                match = re.search(r'TSA[\s\-_]*(\d+)', name_upper)
                if match:
                    wavelength = match.group(1)  # 例如 "520", "570", "670"
                    mif_files[wavelength] = str(file)

    return mif_files


def load_transform_data(transform_dir: str, he_name: str) -> Optional[dict]:
    """加载变换参数"""
    transform_path = os.path.join(transform_dir, f"{he_name}_transform.pkl")
    if not os.path.exists(transform_path):
        return None
    with open(transform_path, 'rb') as f:
        return pickle.load(f)


# =========================
# GPU 加速工具函数
# =========================

def rgb_to_hed_gpu(rgb_tensor):
    """
    将 RGB 图像转换为 H&E 分解图像 (1:1 严格复刻 skimage.color.rgb2hed)
    支持单图 [H, W, 3], [3, H, W] 或 批量 [N, 3, H, W]
    """
    need_permute_4d = False
    need_permute_3d = False

    # 统一转换到 channel last [..., 3] 格式以便进行矩阵乘法
    if rgb_tensor.dim() == 4 and rgb_tensor.shape[1] == 3:
        # [N, 3, H, W] -> [N, H, W, 3]
        rgb_tensor = rgb_tensor.permute(0, 2, 3, 1)
        need_permute_4d = True
    elif rgb_tensor.dim() == 3 and rgb_tensor.shape[0] == 3:
        # [3, H, W] -> [H, W, 3]
        rgb_tensor = rgb_tensor.permute(1, 2, 0)
        need_permute_3d = True

    rgb = rgb_tensor[..., :3]

    # --- 完全对齐 skimage 的底层对数处理 ---
    rgb_safe = torch.clamp(rgb, min=1e-6)
    log_adjust = torch.log(torch.tensor(1e-6, device=rgb.device, dtype=rgb.dtype))
    stains_pre = torch.log(rgb_safe) / log_adjust

    # skimage 内置的 RGB -> HED 转换矩阵
    rgb_from_hed = torch.tensor([
        [0.650, 0.704, 0.286],
        [0.072, 0.990, 0.105],
        [0.268, 0.570, 0.106]
    ], dtype=rgb.dtype, device=rgb.device)

    # 乘以逆矩阵
    hed_from_rgb = torch.linalg.inv(rgb_from_hed)
    hed = torch.matmul(stains_pre, hed_from_rgb)

    # 如果输入时转换了维度，现在转回常规的 Pytorch [C, H, W] 格式
    if need_permute_4d:
        hed = hed.permute(0, 3, 1, 2)
    elif need_permute_3d:
        hed = hed.permute(2, 0, 1)

    return hed


def get_hematoxylin_channel_gpu(rgb_tensor):
    """
    提取苏木精通道，归一化逻辑与 CPU 版完全一致。
    如果是 Batch 处理，会自动逐张图独立计算 Max/Min 进行归一化。
    """
    hed = rgb_to_hed_gpu(rgb_tensor)

    # 根据维度提取 H 通道 (索引为 0)
    if hed.dim() == 4: # [N, 3, H, W]
        h_channel = hed[:, 0, :, :]
        # 逐图计算 min, max
        h_min = h_channel.amin(dim=(1, 2), keepdim=True)
        h_max = h_channel.amax(dim=(1, 2), keepdim=True)
    elif hed.dim() == 3 and hed.shape[-1] == 3: # [H, W, 3]
        h_channel = hed[..., 0]
        h_min = h_channel.min()
        h_max = h_channel.max()
    else: # [3, H, W]
        h_channel = hed[0, :, :]
        h_min = h_channel.min()
        h_max = h_channel.max()

    # --- 严格对齐原来的 if h_max > h_min 逻辑 ---
    diff = h_max - h_min
    safe_diff = torch.where(diff > 0, diff, torch.ones_like(diff))
    h_norm = (h_channel - h_min) / safe_diff
    h_norm = torch.where(diff > 0, h_norm, torch.zeros_like(h_norm))

    return h_norm


def batch_extract_and_resize_gpu(h_mask_large, dapi_mask_binary, positions, target_h, target_w, dapi_size, batch_size=128):
    """
    GPU 批量：从已经全局二值化的 HE 掩码中提取多个位置，resize 并计算 Dice
    逻辑与 CPU 完全一致：裁剪 Mask -> Resize -> 算 Dice
    """
    if len(positions) == 0:
        return torch.tensor([], device=h_mask_large.device)

    num_positions = len(positions)
    dice_scores = torch.zeros(num_positions, device=h_mask_large.device)
    sum_dapi = torch.sum(dapi_mask_binary)

    # Mini-batch 处理防止 OOM
    for batch_start in range(0, num_positions, batch_size):
        batch_end = min(batch_start + batch_size, num_positions)
        batch_positions = positions[batch_start:batch_end]
        current_batch_size = len(batch_positions)

        # 1. 批量裁剪二值化 Mask
        crops = torch.zeros((current_batch_size, 1, target_h, target_w),
                           device=h_mask_large.device, dtype=torch.float32)

        for i, (y, x) in enumerate(batch_positions):
            crops[i, 0] = h_mask_large[y:y+target_h, x:x+target_w]

        # 2. 批量 Resize 到 dapi_size x dapi_size
        crops_resized = F.interpolate(crops, size=(dapi_size, dapi_size),
                                     mode='bilinear', align_corners=False).squeeze(1)

        # 模拟 CPU 中 PIL (uint8) 的截断特性，严格对齐结果
        crops_resized = torch.round(crops_resized)

        # 3. 批量计算 Dice
        intersections = torch.sum(crops_resized * dapi_mask_binary, dim=(1, 2))
        sum_h = torch.sum(crops_resized, dim=(1, 2))

        dice = 2.0 * intersections / (sum_h + sum_dapi + 1e-8)
        dice_scores[batch_start:batch_end] = dice

        # 清理显存
        del crops, crops_resized
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return dice_scores


# =========================
# 二次精细配准函数
# =========================

def get_hematoxylin_channel(rgb_image: np.ndarray) -> np.ndarray:
    """
    从 RGB 图像提取苏木精通道

    Args:
        rgb_image: RGB 图像，范围 [0, 255] 或 [0, 1]

    Returns:
        苏木精通道的灰度图像，范围 [0, 1]
    """
    # 确保输入在 [0, 1] 范围内
    if rgb_image.max() > 1.0:
        rgb_image = rgb_image.astype(float) / 255.0

    # 使用 skimage 的 rgb2hed 进行颜色分解
    hed = rgb2hed(rgb_image)

    # H 通道是第一个通道
    h_channel = hed[:, :, 0]

    # 归一化到 [0, 1]
    h_min, h_max = h_channel.min(), h_channel.max()
    if h_max > h_min:
        h_channel = (h_channel - h_min) / (h_max - h_min)
    else:
        h_channel = np.zeros_like(h_channel)

    return h_channel


def refine_patch_alignment(
    he_patch_large: np.ndarray,
    dapi_patch: np.ndarray,
    target_h: int,
    target_w: int,
    search_range: int = 100,
    prev_offset: Tuple[int, int] = (100, 100),
    dapi_size: int = 512,
    skip_dapi_check: bool = False
) -> Tuple[Tuple[int, int], np.ndarray, float]:
    """
    对 HE patch 进行二次精细配准（GPU 加速版本）

    1. CPU 上提取苏木精通道并全局二值化
    2. CPU 上处理 DAPI 并二值化
    3. GPU 上批量裁剪、resize、计算 Dice

    Args:
        he_patch_large: 扩大后的HE patch（比target_h×target_w大）
        dapi_patch: DAPI patch（512×512）
        target_h: 原始HE crop的高度
        target_w: 原始HE crop的宽度
        search_range: 搜索范围（像素）
        prev_offset: 预期最佳偏移量的中心位置（expand_size, expand_size）
        dapi_size: DAPI patch的尺寸（默认512）
        skip_dapi_check: 是否跳过DAPI mask占比检查

    Returns:
        (best_offset, he_patch_refined, best_dice): 最佳偏移量、resize后的512×512 HE patch、最佳Dice值
    """
    try:
        # ==========================================
        # 步骤1：HE 苏木精提取与二值化（CPU）
        # ==========================================
        hed = rgb2hed(he_patch_large.astype(float) / 255.0)
        h_channel = hed[:, :, 0]

        h_min, h_max = h_channel.min(), h_channel.max()
        if h_max > h_min:
            h_channel = (h_channel - h_min) / (h_max - h_min)
        else:
            h_channel = np.zeros_like(h_channel)

        # HE 苏木精二值化
        h_threshold = threshold_otsu((h_channel * 255).astype(np.uint8))
        h_mask_binary = (h_channel > h_threshold / 255.0).astype(np.uint8)

        # ==========================================
        # 步骤2：DAPI 处理与二值化（CPU）
        # ==========================================
        if len(dapi_patch.shape) == 3:
            dapi_gray = np.dot(dapi_patch[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            dapi_gray = dapi_patch.copy()

        if dapi_gray.max() > 1.0:
            dapi_gray = dapi_gray / 255.0

        # DAPI 二值化
        dapi_threshold = threshold_otsu((dapi_gray * 255).astype(np.uint8))
        dapi_mask_binary = (dapi_gray > dapi_threshold / 255.0).astype(np.uint8)
        dapi_mask_ratio = dapi_mask_binary.sum() / dapi_mask_binary.size

        # 如果不跳过DAPI检查，进行占比验证
        if not skip_dapi_check:
            if dapi_mask_ratio < 0.05:
                raise ValueError(f"DAPI掩码占比不足 ({dapi_mask_ratio*100:.1f}% < 5%)")

        # ==========================================
        # 步骤3：将二值化掩码移到 GPU
        # ==========================================
        h_mask_binary_gpu = torch.from_numpy(h_mask_binary).float().to(device)
        dapi_mask_binary_gpu = torch.from_numpy(dapi_mask_binary).float().to(device)

        # ==========================================
        # 步骤4：计算搜索范围
        # ==========================================
        large_h, large_w = h_mask_binary.shape
        max_offset_x = large_w - target_w
        max_offset_y = large_h - target_h

        # 以 prev_offset 为中心定义搜索范围
        start_x = max(0, prev_offset[0] - search_range)
        end_x = min(max_offset_x, prev_offset[0] + search_range + 1)
        start_y = max(0, prev_offset[1] - search_range)
        end_y = min(max_offset_y, prev_offset[1] + search_range + 1)

        # 如果没有搜索空间，返回中心 resize
        if start_x >= end_x or start_y >= end_y:
            center_x = max(0, (large_w - target_w) // 2)
            center_y = max(0, (large_h - target_h) // 2)
            he_crop = he_patch_large[center_y:center_y+target_h, center_x:center_x+target_w]
            he_resized = np.array(Image.fromarray(he_crop).resize((dapi_size, dapi_size), Image.Resampling.LANCZOS))
            # 计算中心位置的 dice
            h_crop_binary = h_mask_binary[center_y:center_y+target_h, center_x:center_x+target_w]
            h_crop_resized = np.array(Image.fromarray(h_crop_binary).resize((dapi_size, dapi_size), Image.Resampling.LANCZOS))
            intersection = np.sum(h_crop_resized * dapi_mask_binary)
            center_dice = 2.0 * intersection / (np.sum(h_crop_resized) + np.sum(dapi_mask_binary) + 1e-8)
            return (center_x, center_y), he_resized, center_dice

        # ==========================================
        # 步骤5：生成所有搜索位置
        # ==========================================
        step = 5
        positions = []
        for offset_y in range(start_y, end_y, step):
            for offset_x in range(start_x, end_x, step):
                if offset_y + target_h <= large_h and offset_x + target_w <= large_w:
                    positions.append((offset_y, offset_x))

        if len(positions) == 0:
            center_x = max(0, (large_w - target_w) // 2)
            center_y = max(0, (large_h - target_h) // 2)
            he_crop = he_patch_large[center_y:center_y+target_h, center_x:center_x+target_w]
            he_resized = np.array(Image.fromarray(he_crop).resize((dapi_size, dapi_size), Image.Resampling.LANCZOS))
            return (center_x, center_y), he_resized, 0.0

        # ==========================================
        # 步骤6：GPU 批量计算所有位置的 Dice
        # ==========================================
        dice_scores = batch_extract_and_resize_gpu(
            h_mask_binary_gpu, dapi_mask_binary_gpu, positions, target_h, target_w, dapi_size
        )

        # 找到最佳位置
        best_idx = torch.argmax(dice_scores).item()
        best_dice = dice_scores[best_idx].item()

        # 使用最佳偏移裁剪并 resize HE 原图
        best_y, best_x = positions[best_idx]

        he_crop_original = he_patch_large[best_y:best_y+target_h, best_x:best_x+target_w]
        he_final = np.array(Image.fromarray(he_crop_original).resize((dapi_size, dapi_size), Image.Resampling.LANCZOS))

        return (best_x, best_y), he_final, best_dice

    except ValueError:
        # 苏木精掩码不足，向上抛出
        raise
    except Exception:
        # 其他异常：打印错误并返回中心 resize
        large_h, large_w = he_patch_large.shape[:2]
        center_x = max(0, (large_w - target_w) // 2)
        center_y = max(0, (large_h - target_h) // 2)
        he_crop = he_patch_large[center_y:center_y+target_h, center_x:center_x+target_w]
        he_final = np.array(Image.fromarray(he_crop).resize((dapi_size, dapi_size), Image.Resampling.LANCZOS))
        return (center_x, center_y), he_final, 0.0


# =========================
# Patch 映射函数（带二次配准）
# =========================

def map_patch_mif_to_he(
    mif_patch_x: int, mif_patch_y: int, patch_width: int, patch_height: int,
    he_slide: Image.Image,
    transform: sitk.Transform,
    mif_thumb_size: Tuple[int, int],
    mif_orig_size: Tuple[int, int],
    he_thumb_size: Tuple[int, int],
    he_orig_size: Tuple[int, int],
    mif_thumb_original_size: Optional[Tuple[int, int]] = None,
    he_thumb_original_size: Optional[Tuple[int, int]] = None,
    refine_alignment: bool = False,
    dapi_patch: Optional[np.ndarray] = None,
    search_range: int = 100
) -> Tuple[Optional[Image.Image], Tuple[int, int, int, int], Tuple[int, int], float]:
    """
    将 mIF 原图 patch 坐标映射到 HE 原图坐标（支持二次精细配准）

    支持填充模式（register_mif_test.py）：
    - 如果 mif_thumb_original_size/he_thumb_original_size 不为 None，说明使用了填充
    - 需要先转换到原始缩略图，然后考虑填充偏移，再应用变换

    二次配准：
    - refine_alignment: 是否启用二次精细配准
    - dapi_patch: DAPI patch（用于二次配准）
    - search_range: 二次配准的搜索范围

    返回：
    - he_patch_img: HE patch PIL Image（如果失败返回 None）
    - he_coords: (x, y, w, h) HE 原图上的坐标
    - best_offset: 二次配准的最佳偏移 (offset_x, offset_y)
    - dice_score: 二次配准的Dice分数（如果不使用二次配准则返回-1.0）
    """
    try:
        # 判断是否使用填充模式
        use_padding = (mif_thumb_original_size is not None and he_thumb_original_size is not None)

        if use_padding:
            # ========== 填充模式 ==========
            # mIF 原图 → mIF 原始缩略图（填充前）
            mif_downsample_x = mif_orig_size[0] / mif_thumb_original_size[0]
            mif_downsample_y = mif_orig_size[1] / mif_thumb_original_size[1]

            # HE 原图 → HE 原始缩略图（填充前）
            he_downsample_x = he_orig_size[0] / he_thumb_original_size[0]
            he_downsample_y = he_orig_size[1] / he_thumb_original_size[1]

            # mIF 原图坐标 → mIF 原始缩略图坐标
            x_thumb_orig = mif_patch_x / mif_downsample_x
            y_thumb_orig = mif_patch_y / mif_downsample_y
            w_thumb_orig = patch_width / mif_downsample_x
            h_thumb_orig = patch_height / mif_downsample_y

            # 计算填充偏移（居中填充）
            mif_pad_x = (mif_thumb_size[0] - mif_thumb_original_size[0]) // 2
            mif_pad_y = (mif_thumb_size[1] - mif_thumb_original_size[1]) // 2

            # mIF 原始缩略图坐标 → mIF 填充后缩略图坐标（加上偏移）
            x_thumb = x_thumb_orig + mif_pad_x
            y_thumb = y_thumb_orig + mif_pad_y

            # 计算四个角点（在填充后缩略图空间）
            corners = np.array([
                [x_thumb, y_thumb],
                [x_thumb + w_thumb_orig, y_thumb],
                [x_thumb, y_thumb + h_thumb_orig],
                [x_thumb + w_thumb_orig, y_thumb + h_thumb_orig]
            ])
        else:
            # ========== 原始模式（无填充） ==========
            mif_downsample_x = mif_orig_size[0] / mif_thumb_size[0]
            mif_downsample_y = mif_orig_size[1] / mif_thumb_size[1]
            he_downsample_x = he_orig_size[0] / he_thumb_size[0]
            he_downsample_y = he_orig_size[1] / he_thumb_size[1]

            # mIF 原图坐标 -> mIF 缩略图坐标
            x_thumb = mif_patch_x / mif_downsample_x
            y_thumb = mif_patch_y / mif_downsample_y
            w_thumb = patch_width / mif_downsample_x
            h_thumb = patch_height / mif_downsample_y

            # 计算 patch 四个角点
            corners = np.array([
                [x_thumb, y_thumb],
                [x_thumb + w_thumb, y_thumb],
                [x_thumb, y_thumb + h_thumb],
                [x_thumb + w_thumb, y_thumb + h_thumb]
            ])

        # 应用配准变换（使用逆变换）
        # 变换是 moving(DAPI) → fixed(HE)，所以要用逆变换映射 mIF 坐标到 HE
        inverted_transform = transform.GetInverse()
        transformed_corners = []
        for pt in corners:
            transformed = inverted_transform.TransformPoint((float(pt[0]), float(pt[1])))
            transformed_corners.append(transformed)
        transformed_corners = np.array(transformed_corners)

        if use_padding:
            # ========== 填充模式 ==========
            # HE 填充后缩略图坐标 → HE 原始缩略图坐标（减去填充偏移）
            # 居中填充
            he_pad_x = (he_thumb_size[0] - he_thumb_original_size[0]) // 2
            he_pad_y = (he_thumb_size[1] - he_thumb_original_size[1]) // 2

            # 检查变换后的 patch 是否完全在 HE 原始缩略图范围内（不在填充区域）
            he_valid_x_min = he_pad_x
            he_valid_x_max = he_pad_x + he_thumb_original_size[0]
            he_valid_y_min = he_pad_y
            he_valid_y_max = he_pad_y + he_thumb_original_size[1]

            # 如果任何一个角点在有效区域外，说明 patch 进入了填充区域（黑边），跳过
            out_of_bounds = False
            for pt in transformed_corners:
                if not (he_valid_x_min <= pt[0] <= he_valid_x_max and he_valid_y_min <= pt[1] <= he_valid_y_max):
                    out_of_bounds = True
                    break
            if out_of_bounds:
                return None, (0, 0, patch_width, patch_height), (0, 0), -1.0

            # 减去填充偏移
            transformed_corners[:, 0] -= he_pad_x
            transformed_corners[:, 1] -= he_pad_y

            # HE 原始缩略图坐标 -> HE 原图坐标
            he_corners_level0 = np.empty_like(transformed_corners)
            he_corners_level0[:, 0] = transformed_corners[:, 0] * he_downsample_x
            he_corners_level0[:, 1] = transformed_corners[:, 1] * he_downsample_y
        else:
            # ========== 原始模式（无填充） ==========
            # HE 缩略图坐标 -> HE 原图坐标
            he_corners_level0 = np.empty_like(transformed_corners)
            he_corners_level0[:, 0] = transformed_corners[:, 0] * he_downsample_x
            he_corners_level0[:, 1] = transformed_corners[:, 1] * he_downsample_y

        # 计算包围盒
        min_x = int(np.floor(np.min(he_corners_level0[:, 0])))
        min_y = int(np.floor(np.min(he_corners_level0[:, 1])))
        max_x = int(np.ceil(np.max(he_corners_level0[:, 0])))
        max_y = int(np.ceil(np.max(he_corners_level0[:, 1])))

        # 边界检查
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(he_orig_size[0], max_x)
        max_y = min(he_orig_size[1], max_y)

        if min_x >= max_x or min_y >= max_y:
            return None, (0, 0, patch_width, patch_height), (0, 0), -1.0

        # ========== 二次精细配准 ==========
        if refine_alignment and dapi_patch is not None:
            # 固定扩大区域大小为 100 像素（与 refine_gpu.py 一致）
            expand_size = 100

            # 扩大提取区域
            large_min_x = max(0, min_x - expand_size)
            large_min_y = max(0, min_y - expand_size)
            large_max_x = min(he_orig_size[0], max_x + expand_size)
            large_max_y = min(he_orig_size[1], max_y + expand_size)

            # 提取扩大后的 HE patch
            he_patch_large = he_slide.crop((large_min_x, large_min_y, large_max_x, large_max_y))
            he_patch_large_array = np.array(he_patch_large)

            # 计算原始crop的实际尺寸（映射到HE后可能不是正方形）
            original_crop_h = max_y - min_y
            original_crop_w = max_x - min_x

            # 确保扩大区域足够大（对比实际crop尺寸，而非固定patch_size）
            # 因为 mIF 和 HE 宽高比不同，512×512 mIF patch 映射到 HE 可能是非正方形
            if (he_patch_large_array.shape[0] >= original_crop_h and
                he_patch_large_array.shape[1] >= original_crop_w):

                try:
                    # 调用二次配准函数（返回已经是512×512）
                    best_offset, he_refined_array, dice_score = refine_patch_alignment(
                        he_patch_large_array,
                        dapi_patch,
                        target_h=original_crop_h,
                        target_w=original_crop_w,
                        search_range=search_range,
                        prev_offset=(expand_size, expand_size),
                        dapi_size=patch_width,  # 512
                        skip_dapi_check=True  # 阶段3跳过DAPI检查（阶段1已验证）
                    )

                    # 将 numpy 数组转回 PIL Image（已经是512×512）
                    he_patch_img = Image.fromarray(he_refined_array)

                    # 更新坐标（加上扩大区域的偏移）
                    actual_x = large_min_x + best_offset[0]
                    actual_y = large_min_y + best_offset[1]

                    return he_patch_img, (actual_x, actual_y, patch_width, patch_height), best_offset, dice_score

                except ValueError:
                    # DAPI掩码不足，跳过这对 patch
                    return None, (0, 0, patch_width, patch_height), (0, 0), -1.0

        # ========== 不使用二次配准，直接裁剪 ==========
        # 从 HE 原图裁剪
        he_patch_img = he_slide.crop((min_x, min_y, max_x, max_y))
        he_patch_img = he_patch_img.resize((patch_width, patch_height), Image.Resampling.BILINEAR)

        return he_patch_img, (min_x, min_y, max_x - min_x, max_y - min_y), (0, 0), -1.0

    except Exception:
        return None, (0, 0, patch_width, patch_height), (0, 0), -1.0


# =========================
# Patch 保存函数（多线程）
# =========================

def save_single_patch_task(task: Dict) -> bool:
    """
    保存单个 patch 的任务函数（用于多线程）

    task 格式：
    {
        'he_patch': np.ndarray,
        'mif_patches': {channel_id: np.ndarray, ...},
        'mif_x': int,
        'mif_y': int,
        'he_output_dir': str,
        'mif_output_dir': str,
        'all_channels': [(channel_id, channel_name), ...]
    }
    """
    try:
        he_patch = task['he_patch']
        mif_patches = task['mif_patches']
        mif_x, mif_y = task['mif_x'], task['mif_y']
        he_output_dir = task['he_output_dir']
        mif_output_dir = task['mif_output_dir']
        all_channels = task['all_channels']

        # 保存 HE patch（使用 mIF 坐标命名）
        he_patch_path = os.path.join(he_output_dir, f"he_x{mif_x}_y{mif_y}.png")

        # 如果文件已存在，跳过保存
        if os.path.exists(he_patch_path):
            pass
        else:
            # HE 图像转换为 uint8 保存
            if he_patch.dtype != np.uint8:
                he_patch_to_save = (he_patch / 256).astype(np.uint8) if he_patch.max() > 255 else he_patch.astype(np.uint8)
            else:
                he_patch_to_save = he_patch
            Image.fromarray(he_patch_to_save).save(he_patch_path)

        # 保存 mIF patches（不归一化，保存原始值）
        for channel_id, channel_name in all_channels:
            if channel_id not in mif_patches:
                continue

            channel_dir = os.path.join(mif_output_dir, f"{channel_id}-{channel_name}")
            mif_patch_path = os.path.join(channel_dir, f"mF{channel_id}_x{mif_x}_y{mif_y}.png")

            # 如果文件已存在，跳过保存
            if os.path.exists(mif_patch_path):
                continue

            mif_patch = mif_patches[channel_id]

            # 保存原始数据
            if len(mif_patch.shape) == 2:
                Image.fromarray(mif_patch, mode='L').save(mif_patch_path)
            else:
                Image.fromarray(mif_patch).save(mif_patch_path)

        return True

    except Exception as e:
        print(f"  保存 patch ({mif_x}, {mif_y}) 失败: {e}")
        return False


# =========================
# 主处理函数
# =========================

def extract_patches_for_case(
    he_path: str,
    transform_dir: str,
    output_dir: str,
    patch_size: int = 512,
    stride: int = 512,  # 修改默认值：不重叠
    black_threshold: float = 10,
    white_threshold: float = 245,
    std_threshold: float = 5.0,
    black_area_threshold: float = 0.5,
    max_workers: int = 4
) -> bool:
    """为单个病例提取 patch"""

    # 获取病例名称
    he_name = Path(he_path).stem.replace('_rotated', '').replace('.tiff', '').replace('.tif', '').replace('.svs', '')

    print(f"\n{'='*60}")
    print(f"提取 patches: {he_name}")
    print(f"{'='*60}")

    # 加载变换参数
    transform_data = load_transform_data(transform_dir, he_name)
    if transform_data is None:
        print(f"  ✗ 未找到变换参数: {os.path.join(transform_dir, f'{he_name}_transform.pkl')}")
        return False

    # 解析通道信息
    mif_folder = transform_data['mif_folder']
    mif_folder_name = os.path.basename(mif_folder)
    wavelength_channels = parse_mif_folder_name(mif_folder_name)  # [(波长, 通道名), ...]

    # 获取 mIF 文件
    mif_files_by_wavelength = find_mif_files(mif_folder)  # {波长: 文件路径}

    print(f"  从文件夹名解析的通道:")
    for wavelength, channel_name in wavelength_channels:
        print(f"    {wavelength}nm: {channel_name}")

    print(f"  找到 mIF 文件:")
    for wavelength, path in mif_files_by_wavelength.items():
        print(f"    {wavelength}: {os.path.basename(path)}")

    # 将波长+通道名映射到 channel_id
    all_channels = [(0, 'DAPI')]  # DAPI 通道
    channel_id_to_wavelength = {}  # {channel_id: 波长}

    for wavelength, channel_name in wavelength_channels:
        # 从 CHANNEL_MAP 反向查找 channel_id
        channel_id = None
        for cid, cname in CHANNEL_MAP.items():
            if cname == channel_name:
                channel_id = int(cid)
                break

        if channel_id is None:
            print(f"  警告: 未找到通道 '{channel_name}' 的编号，跳过")
            continue

        # 检查 mIF 文件是否存在
        if wavelength not in mif_files_by_wavelength:
            print(f"  警告: 未找到波长 {wavelength} 的 mIF 文件，跳过通道 '{channel_name}'")
            continue

        all_channels.append((channel_id, channel_name))
        channel_id_to_wavelength[channel_id] = wavelength

    print(f"\n  检测到通道 (channel_id: channel_name):")
    for channel_id, channel_name in all_channels:
        wavelength = channel_id_to_wavelength.get(channel_id, 'DAPI')
        print(f"    {channel_id}: {channel_name} ({wavelength}nm)")

    # 加载 HE 原图
    print(f"\n  加载 HE 原图...")
    try:
        he_img = Image.open(he_path)
        he_orig_size = he_img.size
        print(f"    HE 原始大小: {he_orig_size[0]} × {he_orig_size[1]}")
    except Exception as e:
        print(f"    ✗ 加载 HE 失败: {e}")
        return False

    # 读取变换参数
    rigid_transform = transform_data['Transformation']
    # displacement_tx = transform_data['DisplacementField']  # 不再使用：displacement field是在缩略图计算的，不适合patch级别的图像变换
    thumb_size = transform_data['thumbnail_size']
    dapi_orig_size = transform_data['dapi_original_size']

    # 检查是否使用填充（兼容 register_mif_test.py 新旧两种 pkl 格式）
    padding_used = transform_data.get('padding_used', False)
    if padding_used:
        # ===== 新格式：pkl 中已标记 padding_used=True =====
        print("  检测到填充模式（pkl 中已标记），使用特殊坐标转换")
        mif_thumb_original_size = transform_data.get('dapi_thumb_original_size')
        he_thumb_original_size = transform_data.get('he_thumb_original_size')
        if mif_thumb_original_size is None or he_thumb_original_size is None:
            print("  ✗ 错误：填充模式下缺少原始缩略图尺寸信息")
            return False
        print(f"    mIF 原始缩略图: {mif_thumb_original_size}, 填充后: {thumb_size}")
        print(f"    HE 原始缩略图: {he_thumb_original_size}, 填充后: {thumb_size}")
    else:
        # ===== 旧格式：pkl 中无 padding_used 字段，按无填充方式处理 =====
        mif_thumb_original_size = None
        he_thumb_original_size = None

    # 创建输出目录
    case_output_dir = os.path.join(output_dir, he_name)
    he_output_dir = os.path.join(case_output_dir, 'he')
    mif_output_dir = os.path.join(case_output_dir, 'mIF')
    os.makedirs(he_output_dir, exist_ok=True)
    os.makedirs(mif_output_dir, exist_ok=True)

    # 创建各通道输出目录
    for channel_id, channel_name in all_channels:
        channel_dir = os.path.join(mif_output_dir, f"{channel_id}-{channel_name}")
        os.makedirs(channel_dir, exist_ok=True)

    # 检查是否存在已有的 patch_coords.json
    json_path = os.path.join(case_output_dir, "patch_coords.json")
    use_existing_json = False

    if os.path.exists(json_path):
        print(f"\n 发现已存在的坐标文件: {json_path}")
        try:
            with open(json_path, "r") as f:
                valid_coords = json.load(f)
            print(f"    ✓ 加载了 {len(valid_coords)} 个有效坐标")
            use_existing_json = True
        except Exception as e:
            print(f"    ✗ 加载 JSON 失败: {e}，将重新扫描")
            use_existing_json = False
            valid_coords = []

    if not use_existing_json:
        # 生成网格坐标（基于 DAPI 原图大小）
        dapi_width, dapi_height = dapi_orig_size
        print(f"\n  生成 patch 网格（基于 DAPI 原图大小: {dapi_width} × {dapi_height}）...")

        coords = []
        for y in range(0, dapi_height - patch_size + 1, stride):
            for x in range(0, dapi_width - patch_size + 1, stride):
                coords.append((x, y))

        print(f"    共 {len(coords)} 个 patch 候选")

    # 预加载 mIF 图像数据（用于后续裁剪）
    print(f"\n  预加载 mIF 图像...")
    mif_images = {}
    for wavelength, path in mif_files_by_wavelength.items():
        try:
            img = Image.open(path)
            if getattr(img, 'n_frames', 1) > 1:
                img.seek(0)
            mif_images[wavelength] = img
            print(f"    ✓ 加载 {wavelength}nm")
        except Exception as e:
            print(f"    ✗ 加载 {wavelength}nm 失败: {e}")

    save_tasks = []

    # ==================== 阶段1: 扫描并验证（如果 JSON 不存在）====================
    if not use_existing_json:
        print(f"\n  阶段1: 扫描并验证 HE 和 DAPI...")

        valid_coords = []
        skipped_count = 0
        invalid_he_count = 0
        dapi_mask_low_count = 0  # DAPI掩码占比不足计数

        for mif_x, mif_y in tqdm(coords, desc=f"  {he_name}"):
            try:
                # ===== 1. 快速验证 HE =====
                he_patch, _, _, _ = map_patch_mif_to_he(
                    mif_patch_x=mif_x,
                    mif_patch_y=mif_y,
                    patch_width=patch_size,
                    patch_height=patch_size,
                    he_slide=he_img,
                    transform=rigid_transform,
                    mif_thumb_size=thumb_size,
                    mif_orig_size=dapi_orig_size,
                    he_thumb_size=thumb_size,
                    he_orig_size=he_orig_size,
                    mif_thumb_original_size=mif_thumb_original_size,
                    he_thumb_original_size=he_thumb_original_size,
                    refine_alignment=False,  # 阶段1不使用二次配准，快速验证
                    dapi_patch=None
                )

                if he_patch is None:
                    skipped_count += 1
                    continue

                he_patch_array = np.array(he_patch)

                # ===== 2. 快速验证 HE（使用空白/黑暗检测）=====
                blank_r = blank_ratio(he_patch)
                black_r = black_ratio(he_patch)

                # 阈值：空白 > 50% 或 黑暗 > 50% 则无效
                if blank_r > 0.5 or black_r > 0.5:
                    invalid_he_count += 1
                    continue

                # 判断标准差，排除无纹理的区域
                he_gray = np.array(he_patch.convert('L')) if hasattr(he_patch, 'convert') else he_patch_array
                if he_gray.ndim == 3:
                    he_gray = np.dot(he_gray[...,:3], [0.2989, 0.5870, 0.1140])
                std_val = he_gray.std()
                std_threshold = 5.0
                if std_val < std_threshold:
                    invalid_he_count += 1
                    continue

                # ===== 3. 验证 DAPI 掩码占比（过滤<1%的patch）=====
                if 'DAPI' not in mif_images:
                    # 没有 DAPI 图像，跳过这个坐标
                    dapi_mask_low_count += 1
                    continue

                dapi_img = mif_images['DAPI']
                dapi_patch_temp = dapi_img.crop((mif_x, mif_y, mif_x + patch_size, mif_y + patch_size))
                dapi_patch_array = np.array(dapi_patch_temp)

                # DAPI转灰度
                if len(dapi_patch_array.shape) == 3:
                    dapi_gray = np.dot(dapi_patch_array[..., :3], [0.2989, 0.5870, 0.1140])
                else:
                    dapi_gray = dapi_patch_array.copy()

                if dapi_gray.max() > 1.0:
                    dapi_gray = dapi_gray / 255.0

                # 预检查：DAPI 最大值 < 0.005（约 1.275/255），视为全黑，直接跳过
                if dapi_gray.max() < 0.005:
                    dapi_mask_low_count += 1
                    continue

                # Otsu阈值二值化
                try:
                    dapi_threshold = threshold_otsu((dapi_gray * 255).astype(np.uint8))
                except:
                    # Otsu 失败（图像全为单一值，无有效阈值），视为全黑
                    dapi_mask_low_count += 1
                    continue

                dapi_mask_binary = (dapi_gray > dapi_threshold / 255.0).astype(np.uint8)

                # 计算掩码占比
                dapi_mask_ratio = dapi_mask_binary.sum() / dapi_mask_binary.size

                # 过滤 DAPI 掩码占比 < 1% 的 patch（包含 mask 占比为 0 的情况）
                if dapi_mask_ratio < 0.01:
                    dapi_mask_low_count += 1
                    continue

                # ===== 4. HE 和 DAPI 都有效，保存坐标 =====
                valid_coords.append({"x": mif_x, "y": mif_y})

            except Exception as e:
                # 打印异常信息，帮助调试
                print(f"    ❌ 处理坐标 ({mif_x}, {mif_y}) 异常: {type(e).__name__}: {e}")
                continue

        # 扫描结束，统计
        print(f"\n  扫描完成:")
        print(f"    总坐标数: {len(coords)}")
        print(f"    跳过（映射失败）: {skipped_count}")
        print(f"    无效（HE）: {invalid_he_count}")
        print(f"    跳过（无DAPI或掩码<1%）: {dapi_mask_low_count}")
        print(f"    有效: {len(valid_coords)}")

        # 保存有效坐标
        json_path = os.path.join(case_output_dir, "patch_coords.json")
        with open(json_path, "w") as f:
            json.dump(valid_coords, f, indent=2)
        print(f"  ✓ 有效坐标已保存到: {json_path}")

    # ==================== 阶段3: 根据 JSON 批量提取 patches ====================
    if use_existing_json:
        print(f"\n  阶段3: 从 JSON 加载有效坐标并提取 patches...")
    else:
        print(f"\n  阶段3: 根据有效坐标提取 patches...")

    # 重新加载 JSON（确保使用最新的有效坐标）
    json_path = os.path.join(case_output_dir, "patch_coords.json")
    if not os.path.exists(json_path):
        print(f"  ✗ 未找到坐标文件: {json_path}")
        return False

    with open(json_path, "r") as f:
        valid_coords = json.load(f)

    dapi_mask_insufficient_count = 0
    low_dice_skip_count = 0  # dice<0.1跳过保存图片的计数

    # 相似度记录（用于生成 case_similarity.txt）
    similarity_records = []

    # 用于跟踪需要保留的坐标（用于更新 patch_coords.json）
    coords_to_keep = []

    dice_score_list = []  # 记录所有dice分数用于统计

    for coord in tqdm(valid_coords, desc=f"  {he_name}"):
        mif_x, mif_y = coord['x'], coord['y']

        try:
            # ===== 1. 先提取 DAPI patch（用于二次配准）=====
            dapi_patch_for_refine = None
            if 'DAPI' in mif_images:
                dapi_img = mif_images['DAPI']
                dapi_patch_temp = dapi_img.crop((mif_x, mif_y, mif_x + patch_size, mif_y + patch_size))
                dapi_patch_for_refine = np.array(dapi_patch_temp)

            # ===== 2. 提取 HE patch（带二次配准）=====
            he_patch, he_coords, best_offset, dice_score = map_patch_mif_to_he(
                mif_patch_x=mif_x,
                mif_patch_y=mif_y,
                patch_width=patch_size,
                patch_height=patch_size,
                he_slide=he_img,
                transform=rigid_transform,
                mif_thumb_size=thumb_size,
                mif_orig_size=dapi_orig_size,
                he_thumb_size=thumb_size,
                he_orig_size=he_orig_size,
                mif_thumb_original_size=mif_thumb_original_size,
                he_thumb_original_size=he_thumb_original_size,
                refine_alignment=True,  # 启用二次配准
                dapi_patch=dapi_patch_for_refine,  # 传入 DAPI patch
                search_range=100  # 固定使用全局搜索范围
            )

            # 记录dice分数
            if dice_score >= 0:
                dice_score_list.append(dice_score)

            # 检查是否需要跳过保存图片
            skip_save_image = False

            if he_patch is None:
                dapi_mask_insufficient_count += 1
                skip_save_image = True
                # he_patch 为 None 时完全不记录
            elif dice_score < 0:
                # dice_score == -1.0：计算失败（边界错误、DAPI掩码不足等），完全不记录
                skip_save_image = True
                dapi_mask_insufficient_count += 1
            elif dice_score < 0.1:
                low_dice_skip_count += 1
                skip_save_image = True
                # dice < 0.1：记录信息但不保存图片
                similarity_records.append((mif_x, mif_y, dice_score))
            else:
                # dice >= 0.1：正常情况
                similarity_records.append((mif_x, mif_y, dice_score))
                coords_to_keep.append({'x': mif_x, 'y': mif_y})
                skip_save_image = False

            if skip_save_image:
                continue

            he_patch_array = np.array(he_patch)

            # ===== 6. 提取所有 mIF patches（DAPI + TSA 通道）=====
            mif_patches = {}
            for channel_id, channel_name in all_channels:
                # 获取对应的波长
                if channel_id == 0:
                    wavelength = 'DAPI'
                else:
                    wavelength = channel_id_to_wavelength.get(channel_id)
                    if wavelength is None:
                        continue

                if wavelength not in mif_images:
                    continue

                try:
                    mif_img = mif_images[wavelength]
                    mif_patch = mif_img.crop((mif_x, mif_y, mif_x + patch_size, mif_y + patch_size))
                    mif_patch_array = np.array(mif_patch)

                    mif_patches[channel_id] = mif_patch_array

                except Exception:
                    continue

            # ===== 4. 添加到保存任务 =====
            save_tasks.append({
                'he_patch': he_patch_array,
                'mif_patches': mif_patches,
                'mif_x': mif_x,
                'mif_y': mif_y,
                'he_output_dir': he_output_dir,
                'mif_output_dir': mif_output_dir,
                'all_channels': all_channels
            })

        except Exception:
            continue

    # 统计完成
    print(f"\n  处理完成:")
    print(f"    跳过保存图片（dice<0.1）: {low_dice_skip_count}")

    # 输出Dice分数统计（始终输出，帮助判断二次配准质量）
    if dice_score_list:
        dice_arr = np.array(dice_score_list)
        print(f"    Dice分数统计 (共{len(dice_arr)}个):")
        print(f"      min={dice_arr.min():.4f}, max={dice_arr.max():.4f}, "
              f"mean={dice_arr.mean():.4f}, median={np.median(dice_arr):.4f}")
        # 分布直方图
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        hist, _ = np.histogram(dice_arr, bins=bins)
        print(f"      分布: ", end="")
        for i in range(len(hist)):
            print(f"[{bins[i]:.1f}-{bins[i+1]:.1f}):{hist[i]}  ", end="")
        print()
    else:
        print(f"    ⚠️ 没有有效的Dice分数（所有patch二次配准均失败）")

    print(f"    he_patch=None 或 dice<0: {dapi_mask_insufficient_count}")
    print(f"    有效保存: {len(coords_to_keep)}/{len(valid_coords)}")

    # ==================== 更新 patch_coords.json ====================
    if len(coords_to_keep) < len(valid_coords):
        json_path = os.path.join(case_output_dir, "patch_coords.json")
        with open(json_path, "w") as f:
            json.dump(coords_to_keep, f, indent=2)
        print(f"    ✓ 更新坐标文件: {json_path} (保留 {len(coords_to_keep)}/{len(valid_coords)} 个坐标)")

    # ==================== 保存相似度记录到 case_similarity.txt ====================
    if similarity_records:
        similarity_txt_path = os.path.join(case_output_dir, "case_similarity.txt")
        with open(similarity_txt_path, 'w') as f:
            f.write("# Case Similarity Record\n")
            f.write("# Format: X Y Dice_Score\n")
            f.write(f"# Total patches: {len(similarity_records)}\n")
            for x, y, dice in similarity_records:
                f.write(f"{x}\t{y}\t{dice:.6f}\n")
        print(f"    ✓ 保存相似度记录: {similarity_txt_path}")

    # ==================== 阶段4: 多线程保存 patches ====================

    if len(save_tasks) > 0:
        print(f"\n  保存 patches（使用 {max_workers} 线程）...")

        success_count = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(save_single_patch_task, task): task for task in save_tasks}

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"  {he_name}"):
                try:
                    if future.result():
                        success_count += 1
                except Exception as e:
                    print(f"    ❌ 保存失败: {e}")

        print(f"    成功保存: {success_count}/{len(save_tasks)}")

    print(f"\n  ✓ 完成处理: {he_name}")
    return True


def process_folder(
    he_dir: str,
    transform_root: str,
    output_dir: str,
    patch_size: int = 512,
    stride: int = 512,  # 修改默认值：不重叠
    black_threshold: float = 10,
    white_threshold: float = 245,
    std_threshold: float = 5.0,
    black_area_threshold: float = 0.5,
    max_workers: int = 4
):
    """处理文件夹中的所有病例"""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"批量提取 patches")
    print(f"{'='*60}")

    # 获取所有 HE 文件
    he_files = []
    for file in os.listdir(he_dir):
        if file.endswith('.tiff') or file.endswith('.tif') or file.endswith('.svs'):
            he_files.append(os.path.join(he_dir, file))

    if not he_files:
        print("⚠️  文件夹中没有找到 HE 文件")
        return

    print(f"找到 {len(he_files)} 张 HE 图像")

    # 处理每个病例
    success_count = 0
    for i, he_path in enumerate(he_files, start=1):
        he_name = Path(he_path).stem.replace('_rotated', '')

        # 查找对应的变换目录
        transform_dir = None
        if os.path.exists(transform_root):
            for subdir in os.listdir(transform_root):
                if os.path.isdir(os.path.join(transform_root, subdir)) and subdir.startswith(he_name):
                    if subdir.replace(' ', '').startswith(he_name.replace(' ', '')):
                        transform_dir = os.path.join(transform_root, subdir)
                        break

        if not transform_dir:
            print(f"\n[全局 {i}] 跳过（未找到变换目录）: {os.path.basename(he_path)}")
            continue

        # 处理
        print(f"\n[全局 {i}] 处理: {os.path.basename(he_path)}")
        try:
            if extract_patches_for_case(
                he_path, transform_dir, output_dir,
                patch_size, stride,
                black_threshold, white_threshold, std_threshold, black_area_threshold,
                max_workers
            ):
                success_count += 1
        except Exception as e:
            print(f"理失败: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"处理完成: {success_count} 个病例成功")
    print(f"{'='*60}")


# =========================
# 主函数
# =========================

def main():
    parser = argparse.ArgumentParser(description='Patch 提取工具 - 应用配准变换参数，判断 HE patch 有效性')
    parser.add_argument("--he_path", type=str, default=None,
                        help="HE 文件路径（单图模式）")
    parser.add_argument("--he_dir", type=str, default=None,
                        help="HE 文件夹路径（文件夹模式）")
    parser.add_argument("--transform_dir", type=str, default=None,
                        help="变换参数目录（单图模式）")
    parser.add_argument("--transform_root", type=str, default=None,
                        help="变换参数根目录（文件夹模式）")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--patch_size", type=int, default=512,
                        help="Patch 大小（默认: 512）")
    parser.add_argument("--stride", type=int, default=512,
                        help="Patch 步长（默认: 512，即不重叠；设置为256可获得50%重叠）")
    parser.add_argument("--black_threshold", type=float, default=10,
                        help="HE patch 黑色阈值（默认: 10）")
    parser.add_argument("--white_threshold", type=float, default=245,
                        help="HE patch 白色阈值（默认: 245）")
    parser.add_argument("--std_threshold", type=float, default=5.0,
                        help="HE patch 标准差阈值（默认: 5.0）")
    parser.add_argument("--black_area_threshold", type=float, default=0.5,
                        help="HE patch 黑色区域比例阈值（默认: 0.5）")
    parser.add_argument("--max_workers", type=int, default=4,
                        help="多线程保存时的线程数（默认: 4）")
    parser.add_argument("--log_dir", type=str, default="./logs",
                        help="日志目录（默认: ./logs）")

    args = parser.parse_args()

    # 单图模式
    if args.he_path:
        if not args.transform_dir:
            print("❌ 单图模式需要指定 --transform_dir")
            return
        print("📄 单张图像模式")
        extract_patches_for_case(
            args.he_path, args.transform_dir, args.output_dir,
            args.patch_size, args.stride,
            args.black_threshold, args.white_threshold, args.std_threshold, args.black_area_threshold,
            args.max_workers
        )

    # 文件夹模式
    elif args.he_dir:
        if not args.transform_root:
            print("❌ 文件夹模式需要指定 --transform_root")
            return
        print("📁 文件夹模式")

        # ==================== 自动检测 GPU 并多进程处理 ====================
        # 创建日志目录
        os.makedirs(args.log_dir, exist_ok=True)
        logger = setup_logger(args.log_dir)
        logger.info("==== Patch extraction started ====")
        logger.info(f"HE dir: {args.he_dir}")
        logger.info(f"Transform root: {args.transform_root}")
        logger.info(f"Output dir: {args.output_dir}")

        # 收集所有需要处理的 case
        he_files = []
        for f in os.listdir(args.he_dir):
            if f.endswith('.tiff') or f.endswith('.tif'):
                he_files.append(os.path.join(args.he_dir, f))
        he_files = sorted(he_files)

        # 构建 case 信息列表
        case_list = []
        for he_path in he_files:
            he_name = Path(he_path).stem.replace('_rotated', '')
            transform_dir = None
            for subdir in os.listdir(args.transform_root):
                subdir_path = os.path.join(args.transform_root, subdir)
                if os.path.isdir(subdir_path) and subdir.startswith(he_name):
                    if subdir.replace(' ', '').startswith(he_name.replace(' ', '')):
                        transform_dir = subdir_path
                        break

            if transform_dir:
                case_list.append({
                    'he_path': he_path,
                    'transform_dir': transform_dir,
                    'he_name': he_name
                })

        logger.info(f"Total cases to process: {len(case_list)}")

        # 单进程模式
        success_count = 0
        for case_info in tqdm(case_list, desc="处理进度"):
            he_path = case_info['he_path']
            transform_dir = case_info['transform_dir']
            he_name = case_info['he_name']

            print(f"\n处理: {he_name}")
            try:
                if extract_patches_for_case(
                    he_path, transform_dir, args.output_dir,
                    args.patch_size, args.stride,
                    args.black_threshold, args.white_threshold, args.std_threshold, args.black_area_threshold,
                    args.max_workers
                ):
                    success_count += 1
            except Exception as e:
                logger.error(f"Failed: {he_name} - {e}")

        logger.info(f"==== Finished ====")
        logger.info(f"Total success: {success_count}/{len(case_list)}")
        print(f"\n✅ 处理完成: {success_count}/{len(case_list)} 个 case")


    else:
        print("请指定 --he_path 或 --he_dir")


if __name__ == "__main__":
    main()
