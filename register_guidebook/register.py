#!/usr/bin/env python3

"""
MIF 图像配准工具

数据标准路径：
he图:(DEFAULT_HE_DIR)
/path/to/he_rotated/
├── 7A 25-008831 2_rotated.tiff
├── 7A 25-000232 2_rotated.tiff
└── ...

DAPI图:
/path/to/DAPI  (DEFAULT_DAPI_ROOT)
├── 7A 25-008831 2/
│   └── XXX_DAPI.tiff
├── 7A 25-000232 2/
│   └── YYY_DAPI.tiff
└── ...



# 单张图像
python register_mif_test.py \
  --input_path "/nfs-medical3/sxz/XCellAligner/data/he_rotated/7A 25-000232 2_rotated.tiff"

# 文件夹模式
python register_mif_test.py \
  --input_path "/nfs-medical3/sxz/XCellAligner/data/he_rotated"

若DAPI文件的路径特殊，可修改def get_dapi_file(he_file_path):
"""

import os
import pickle
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, Tuple
import SimpleITK as sitk
from skimage.transform import resize
import tifffile

# 允许处理更大的图像
Image.MAX_IMAGE_PIXELS = None
Image.OPEN_IGNORE_TRUNCATED_IMAGES = True

# 默认路径
DEFAULT_HE_DIR = "/nfs-medical3/sxz/XCellAligner/data/he_rotated"
DEFAULT_DAPI_ROOT = "/nfs-medical4/data/浙一多重荧光/胃癌/mIF-Channel"
DEFAULT_OUTPUT_DIR = "/nfs-medical3/sxz/XCellAligner/data/mif_registered_test"

# 设置 ITK 线程数
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "32"


# =========================
# 图像加载函数
# =========================

def load_he_thumbnail(he_path, max_long_side=1024):
    """加载 HE 图像的缩略图，长边缩放到 max_long_side"""
    try:
        img = Image.open(he_path)

        if hasattr(img, 'n_frames') and img.n_frames > 1:
            best_level = 0
            best_size = (0, 0)
            for i in range(min(img.n_frames, 10)):
                try:
                    img.seek(i)
                    level_size = img.size
                    if level_size[0] >= max_long_side or level_size[1] >= max_long_side:
                        best_level = i
                        best_size = level_size
                        break
                    if level_size[0] > best_size[0]:
                        best_level = i
                        best_size = level_size
                except:
                    break
            img.seek(best_level)

        # 保持宽高比缩放，使长边为 max_long_side
        w, h = img.size
        if max(w, h) > max_long_side:
            if w > h:
                new_w = max_long_side
                new_h = int(h * max_long_side / w)
            else:
                new_h = max_long_side
                new_w = int(w * max_long_side / h)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        return np.array(img)
    except Exception as e:
        print(f"  ✗ 加载HE失败: {e}")
        return None


def load_dapi_thumbnail(dapi_path, max_long_side=1024):
    """
    加载 DAPI 图像的缩略图，长边缩放到 max_long_side
    """
    try:
        img_pil = Image.open(dapi_path)
        if getattr(img_pil, 'n_frames', 1) > 1:
            img_pil.seek(0)

        # 保持宽高比缩放，使长边为 max_long_side
        w, h = img_pil.size
        if max(w, h) > max_long_side:
            if w > h:
                new_w = max_long_side
                new_h = int(h * max_long_side / w)
            else:
                new_h = max_long_side
                new_w = int(w * max_long_side / h)
            img_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

        img_array = np.array(img_pil)

        # 转为灰度
        if img_array.ndim == 3 and img_array.shape[2] == 3:
            gray = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
            img_array = gray.astype(np.uint8)

        return img_array
    except Exception as e:
        print(f"  ✗ 加载DAPI失败: {e}")
        return None


def get_original_image_sizes(he_path, dapi_path):
    """获取 HE 和 DAPI 原始图像尺寸"""
    he_size = None
    dapi_size = None

    try:
        with Image.open(he_path) as img:
            he_size = img.size
    except:
        pass

    try:
        with Image.open(dapi_path) as img:
            dapi_size = img.size
    except:
        pass

    return he_size, dapi_size


# =========================
# 新的尺寸匹配函数（填充黑边）
# =========================

def pad_to_match_width(img1, img2) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    通过填充黑边使两个图像宽度一致，避免变形

    返回:
        (img1_padded, img2_padded, target_w, target_h)
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 找到较大的宽度
    target_w = max(w1, w2)
    # 找到较大的高度
    target_h = max(h1, h2)

    # 准备填充后的图像
    if img1.ndim == 2:
        img1_padded = np.zeros((target_h, target_w), dtype=img1.dtype)
    else:
        img1_padded = np.zeros((target_h, target_w, img1.shape[2]), dtype=img1.dtype)

    if img2.ndim == 2:
        img2_padded = np.zeros((target_h, target_w), dtype=img2.dtype)
    else:
        img2_padded = np.zeros((target_h, target_w, img2.shape[2]), dtype=img2.dtype)

    # 计算居中偏移
    y1_offset = (target_h - h1) // 2
    x1_offset = (target_w - w1) // 2
    y2_offset = (target_h - h2) // 2
    x2_offset = (target_w - w2) // 2

    # 填充图像
    if img1.ndim == 2:
        img1_padded[y1_offset:y1_offset+h1, x1_offset:x1_offset+w1] = img1
    else:
        img1_padded[y1_offset:y1_offset+h1, x1_offset:x1_offset+w1, :] = img1

    if img2.ndim == 2:
        img2_padded[y2_offset:y2_offset+h2, x2_offset:x2_offset+w2] = img2
    else:
        img2_padded[y2_offset:y2_offset+h2, x2_offset:x2_offset+w2, :] = img2

    return img1_padded, img2_padded, target_w, target_h

def get_dapi_file(he_file_path):
    """根据 HE 文件路径查找对应的 DAPI 文件"""
    he_path = Path(he_file_path)
    he_filename = he_path.name
    base_name = he_filename
    for ext in ['.tiff', '.tif', '.svs', '.kfb']:
        base_name = base_name.replace(ext, '')
    base_name = base_name.replace('_rotated', '')

    dapi_path = Path(DEFAULT_DAPI_ROOT)
    if not dapi_path.exists():
        return None

    for dapi_subdir in dapi_path.iterdir():
        if dapi_subdir.is_dir() and dapi_subdir.name.startswith(base_name):
            for file in dapi_subdir.iterdir():
                if file.is_file():
                    name_upper = file.name.upper()
                    if 'DAPI' in name_upper:
                        return str(file)
    return None


# =========================
# 配准函数
# =========================

def resample_image(image_np, reference_sitk, transform):
    """图像重采样（支持灰度和 RGB）"""
    if len(image_np.shape) == 2:
        channel = sitk.GetImageFromArray(image_np)
        channel.CopyInformation(reference_sitk)
        registered = sitk.Resample(channel, reference_sitk, transform, sitk.sitkLinear, 0.0)
        return sitk.GetArrayFromImage(registered).astype(np.uint8)
    else:
        registered_channels = []
        for i in range(min(3, image_np.shape[2])):
            channel = sitk.GetImageFromArray(image_np[..., i])
            channel.CopyInformation(reference_sitk)
            registered = sitk.Resample(channel, reference_sitk, transform, sitk.sitkLinear, 0.0)
            registered_channels.append(sitk.GetArrayFromImage(registered))
        return np.stack(registered_channels, axis=-1).astype(np.uint8)


def warp_image(image_np, reference_sitk, transform):
    """warp 图像用 displacement field（支持灰度和 RGB）"""
    if len(image_np.shape) == 2:
        channel = sitk.GetImageFromArray(image_np)
        channel.CopyInformation(reference_sitk)
        warped = sitk.Resample(channel, reference_sitk, transform, sitk.sitkLinear, 0.0)
        return sitk.GetArrayFromImage(warped).astype(np.uint8)
    else:
        warped_channels = []
        for i in range(min(3, image_np.shape[2])):
            channel = sitk.GetImageFromArray(image_np[..., i])
            channel.CopyInformation(reference_sitk)
            warped = sitk.Resample(channel, reference_sitk, transform, sitk.sitkLinear, 0.0)
            warped_channels.append(sitk.GetArrayFromImage(warped))
        return np.stack(warped_channels, axis=-1).astype(np.uint8)


def register_images(moving_rgb_np, fixed_rgb_np):
    """
    配准图像（匹配 patch_registration.py 的参数）

    参数:
    - moving_rgb_np: 移动图像（DAPI）
    - fixed_rgb_np: 固定图像（HE）
    """
    # 固定参数
    sigma = 1.5
    iterations = 5000
    learning_rate = 1.5
    hist_bins = 50
    demons_iter = 80

    # 转灰度
    if len(moving_rgb_np.shape) == 3 and moving_rgb_np.shape[2] == 3:
        moving_gray = np.dot(moving_rgb_np[...,:3], [0.2989, 0.5870, 0.1140])
    else:
        moving_gray = moving_rgb_np

    if len(fixed_rgb_np.shape) == 3 and fixed_rgb_np.shape[2] == 3:
        fixed_gray = np.dot(fixed_rgb_np[...,:3], [0.2989, 0.5870, 0.1140])
    else:
        fixed_gray = fixed_rgb_np

    # 归一化
    if moving_gray.max() > 1.0:
        moving_gray = moving_gray / 255.0
    if fixed_gray.max() > 1.0:
        fixed_gray = fixed_gray / 255.0
    moving_gray = np.clip(moving_gray, 0, 1)
    fixed_gray = np.clip(fixed_gray, 0, 1)

    moving = sitk.GetImageFromArray((moving_gray * 255).astype(np.uint8))
    fixed = sitk.GetImageFromArray((fixed_gray * 255).astype(np.uint8))

    fixed.SetOrigin((0.0, 0.0))
    fixed.SetSpacing((1.0, 1.0))
    moving.SetOrigin((0.0, 0.0))
    moving.SetSpacing((1.0, 1.0))

    # 高斯平滑
    fixed = sitk.SmoothingRecursiveGaussian(fixed, sigma=sigma)
    moving = sitk.SmoothingRecursiveGaussian(moving, sigma=sigma)

    # 初始变换
    initial_transform = sitk.AffineTransform(2)
    matrix = [1.05, 0.0, 0.0, 1.00]
    initial_transform.SetMatrix(matrix)

    def get_center(image):
        size = image.GetSize()
        return np.array([size[0] / 2.0, size[1] / 2.0])

    center_moving = get_center(moving)
    center_fixed = get_center(fixed)
    translation = center_fixed - center_moving
    initial_transform.SetTranslation([translation[0], translation[1]])

    # 配准设置
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=hist_bins)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(1.0)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=learning_rate, minStep=1e-6, numberOfIterations=iterations, gradientMagnitudeTolerance=1e-6
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # 执行配准
    final_transform = registration_method.Execute(fixed, moving)

    # 应用变换
    registered_rigid = resample_image(moving_rgb_np, fixed, final_transform)

    # 非刚性微调
    moving_gray_registered = sitk.Resample(moving, fixed, final_transform, sitk.sitkLinear, 0.0)
    demons = sitk.DemonsRegistrationFilter()
    demons.SetNumberOfIterations(demons_iter)
    demons.SetStandardDeviations(1.0)
    displacement_field = demons.Execute(fixed, moving_gray_registered)
    displacement_tx = sitk.DisplacementFieldTransform(displacement_field)
    displacement_tx.SetSmoothingGaussianOnUpdate(0.0, 1.0)

    registered_final = warp_image(registered_rigid, fixed, displacement_tx)

    return {
        'RegisteredImage': registered_final,
        'Transformation': final_transform,
        'DisplacementField': displacement_tx
    }


# =========================
# 预览图生成
# =========================

def create_overview(he_array, dapi_before, dapi_after, overview_path, he_name, alpha=0.5):
    """创建 overview 图（4图布局）"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'{he_name} - DAPI Registration Overview (Test: Pad Instead of Resize)', fontsize=14)

        # 左上：HE 原图 - 完整显示
        axes[0, 0].imshow(he_array)
        axes[0, 0].set_title(f'HE (Reference) {he_array.shape}', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        # 右上：配准前 DAPI - 完整显示
        axes[0, 1].imshow(dapi_before, cmap='gray')
        axes[0, 1].set_title(f'DAPI Before Registration {dapi_before.shape}', fontsize=12, color='red')
        axes[0, 1].axis('off')

        # 左下：配准前叠加 - 创建足够大的画布
        max_h = max(he_array.shape[0], dapi_before.shape[0])
        max_w = max(he_array.shape[1], dapi_before.shape[1])

        # 创建空白画布
        if he_array.ndim == 3:
            canvas_before = np.zeros((max_h, max_w, 3), dtype=he_array.dtype)
        else:
            canvas_before = np.zeros((max_h, max_w), dtype=he_array.dtype)

        # 计算居中偏移
        he_y_offset = (max_h - he_array.shape[0]) // 2
        he_x_offset = (max_w - he_array.shape[1]) // 2
        dapi_y_offset = (max_h - dapi_before.shape[0]) // 2
        dapi_x_offset = (max_w - dapi_before.shape[1]) // 2

        # 填充图像
        if he_array.ndim == 3:
            canvas_before[he_y_offset:he_y_offset+he_array.shape[0],
                         he_x_offset:he_x_offset+he_array.shape[1]] = he_array
        else:
            canvas_before[he_y_offset:he_y_offset+he_array.shape[0],
                         he_x_offset:he_x_offset+he_array.shape[1]] = he_array

        axes[1, 0].imshow(canvas_before)

        # 叠加 DAPI（处理灰度图）
        if dapi_before.ndim == 2:
            dapi_overlay = np.zeros((max_h, max_w), dtype=dapi_before.dtype)
            dapi_overlay[dapi_y_offset:dapi_y_offset+dapi_before.shape[0],
                        dapi_x_offset:dapi_x_offset+dapi_before.shape[1]] = dapi_before
            axes[1, 0].imshow(dapi_overlay, alpha=alpha, cmap='gray')
        else:
            dapi_overlay = np.zeros((max_h, max_w, 3), dtype=dapi_before.dtype)
            dapi_overlay[dapi_y_offset:dapi_y_offset+dapi_before.shape[0],
                        dapi_x_offset:dapi_x_offset+dapi_before.shape[1]] = dapi_before
            axes[1, 0].imshow(dapi_overlay, alpha=alpha)

        axes[1, 0].set_title('Overlay: Before Registration', fontsize=12, color='red')
        axes[1, 0].axis('off')

        # 右下：配准后叠加 - 使用 HE 作为基准
        max_h_after = max(he_array.shape[0], dapi_after.shape[0])
        max_w_after = max(he_array.shape[1], dapi_after.shape[1])

        if he_array.ndim == 3:
            canvas_after = np.zeros((max_h_after, max_w_after, 3), dtype=he_array.dtype)
        else:
            canvas_after = np.zeros((max_h_after, max_w_after), dtype=he_array.dtype)

        he_y_offset_after = (max_h_after - he_array.shape[0]) // 2
        he_x_offset_after = (max_w_after - he_array.shape[1]) // 2
        dapi_y_offset_after = (max_h_after - dapi_after.shape[0]) // 2
        dapi_x_offset_after = (max_w_after - dapi_after.shape[1]) // 2

        if he_array.ndim == 3:
            canvas_after[he_y_offset_after:he_y_offset_after+he_array.shape[0],
                         he_x_offset_after:he_x_offset_after+he_array.shape[1]] = he_array
        else:
            canvas_after[he_y_offset_after:he_y_offset_after+he_array.shape[0],
                         he_x_offset_after:he_x_offset_after+he_array.shape[1]] = he_array

        axes[1, 1].imshow(canvas_after)

        if dapi_after.ndim == 2:
            dapi_overlay_after = np.zeros((max_h_after, max_w_after), dtype=dapi_after.dtype)
            dapi_overlay_after[dapi_y_offset_after:dapi_y_offset_after+dapi_after.shape[0],
                              dapi_x_offset_after:dapi_x_offset_after+dapi_after.shape[1]] = dapi_after
            axes[1, 1].imshow(dapi_overlay_after, alpha=alpha, cmap='gray')
        else:
            dapi_overlay_after = np.zeros((max_h_after, max_w_after, 3), dtype=dapi_after.dtype)
            dapi_overlay_after[dapi_y_offset_after:dapi_y_offset_after+dapi_after.shape[0],
                              dapi_x_offset_after:dapi_x_offset_after+dapi_after.shape[1]] = dapi_after
            axes[1, 1].imshow(dapi_overlay_after, alpha=alpha)

        axes[1, 1].set_title('Overlay: After Registration ✓', fontsize=12, color='green', fontweight='bold')
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(overview_path, dpi=120, bbox_inches='tight')
        plt.close()
        return True
    except Exception as e:
        print(f"生成 overview 图失败：{e}")
        return False


# =========================
# 处理函数
# =========================

def process_single_case(he_path, dapi_file, output_dir, case_name, show_preview=True):
    """处理单个病例"""
    # 创建输出文件夹
    he_name = os.path.basename(he_path).replace('.tiff', '').replace('.tif', '').replace('.svs', '').replace('_rotated', '')
    case_output_dir = os.path.join(output_dir, he_name)
    os.makedirs(case_output_dir, exist_ok=True)

    # 检查变换参数是否已存在
    he_name = os.path.basename(he_path).replace('.tiff', '').replace('.tif', '').replace('.svs', '').replace('_rotated', '')
    transform_path = os.path.join(case_output_dir, f"{he_name}_transform.pkl")
    overview_path = os.path.join(case_output_dir, f"{he_name}_overview.png")

    transform_exists = os.path.exists(transform_path)
    overview_exists = os.path.exists(overview_path)

    if show_preview:
        if transform_exists and overview_exists:
            print(f"\n⏭跳过（已存在）: {case_name}")
            return True
        elif transform_exists and not overview_exists:
            print(f"\n找到 transform 但缺少 overview，只生成 overview: {case_name}")
            with open(transform_path, 'rb') as f:
                transform_data = pickle.load(f)
            print("  生成 overview 图...")
            create_overview(
                transform_data['he_thumbnail'],
                transform_data['dapi_thumbnail'],
                transform_data['dapi_registered_thumbnail'],
                overview_path,
                he_name
            )
            print(f"  ✓ 保存 overview: {overview_path}")
            print(f"\n处理完成: {case_name}")
            return True
    else:
        if transform_exists:
            print(f"\n跳过（已存在）: {case_name}")
            return True

    # 正常处理流程
    print(f"\n{'='*60}")
    print(f"处理病例: {case_name}")
    print(f"{'='*60}")
    print(f"HE: {he_path}")

    # 检查 DAPI 文件是否有效
    if not dapi_file:
        print("  ✗ 未找到 DAPI 文件")
        return False

    print(f"  DAPI: {os.path.basename(dapi_file)}")

    # 加载 HE 图像
    print("\n[1/4] 加载 HE 图像...")
    he_image = load_he_thumbnail(he_path, max_long_side=1024)
    if he_image is None:
        print("  ✗ HE 图像加载失败")
        return False
    print(f"  ✓ HE 图像加载成功: {he_image.shape}")

    # 加载 DAPI 图像
    print("\n[2/4] 加载 DAPI 图像...")
    dapi_image = load_dapi_thumbnail(dapi_file, max_long_side=1024)
    if dapi_image is None:
        print("  ✗ DAPI 图像加载失败")
        return False
    print(f"  ✓ DAPI 图像加载成功: {dapi_image.shape}")

    # 获取原始尺寸
    he_orig_size, dapi_orig_size = get_original_image_sizes(he_path, dapi_file)

    # 填充匹配宽度
    print("\n[3/4] 填充匹配宽度并配准...")
    print(f"  原始尺寸 - HE: {he_image.shape}, DAPI: {dapi_image.shape}")

    dapi_padded, he_padded, thumb_w, thumb_h = pad_to_match_width(dapi_image, he_image)
    print(f"  填充后尺寸 - HE: {he_padded.shape}, DAPI: {dapi_padded.shape}")

    result = register_images(dapi_padded, he_padded)
    print("  ✓ 配准完成")

    # 保存变换参数
    print("\n[4/4] 保存结果...")
    transform_data = {
        'Transformation': result['Transformation'],
        'DisplacementField': result['DisplacementField'],
        'he_original_size': he_orig_size,
        'dapi_original_size': dapi_orig_size,
        'thumbnail_size': (thumb_w, thumb_h),  # 保持兼容：填充后的尺寸（配准使用）
        'he_thumbnail': he_padded,
        'dapi_thumbnail': dapi_padded,
        'dapi_registered_thumbnail': result['RegisteredImage'],
        'he_path': he_path,
        'dapi_path': dapi_file,
        'padding_used': True,
        # 原始缩略图尺寸（填充前）
        'he_thumb_original_size': (he_image.shape[1], he_image.shape[0]),  # (width, height)
        'dapi_thumb_original_size': (dapi_image.shape[1], dapi_image.shape[0])  # (width, height)
    }

    with open(transform_path, 'wb') as f:
        pickle.dump(transform_data, f)
    print(f"  ✓ 保存变换参数: {transform_path}")

    # 生成 overview 图
    if show_preview:
        print("  生成 overview 图...")
        create_overview(he_padded, dapi_padded, result['RegisteredImage'], overview_path, he_name)
        print(f"  ✓ 保存 overview: {overview_path}")

    print(f"\n处理完成: {case_name}")
    return True


def process_folder(he_dir, output_dir, show_preview=True):
    """处理文件夹中的所有病例"""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"处理文件夹: {he_dir}")
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

    # 匹配 HE 和 DAPI
    success_count = 0
    for i, he_path in enumerate(he_files, start=1):
        case_name = Path(he_path).stem.replace('_rotated', '')

        # 查找对应的 DAPI 文件
        dapi_file = get_dapi_file(he_path)
        if not dapi_file:
            print(f"\n[全局 {i}] ⏭跳过（未找到DAPI）: {os.path.basename(he_path)}")
            continue

        # 处理
        print(f"\n[全局 {i}] 处理: {os.path.basename(he_path)}")
        try:
            if process_single_case(he_path, dapi_file, output_dir, case_name, show_preview):
                success_count += 1
        except Exception as e:
            print(f"处理失败: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"处理完成: {success_count} 个病例成功")
    print(f"{'='*60}")


# =========================
# 主函数
# =========================

def main():
    parser = argparse.ArgumentParser(description='MIF 图像配准工具 - 测试版（填充黑边而非变形）')
    parser.add_argument("--input_path", type=str, required=True,
                        help="HE 文件路径或 HE 文件夹路径")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"输出目录（默认: {DEFAULT_OUTPUT_DIR}）")
    parser.add_argument("--no_preview", action="store_true",
                        help="不生成 overview 图")

    args = parser.parse_args()

    show_preview = not args.no_preview

    print("测试模式：使用填充黑边而非变形来匹配宽度")

    if os.path.isfile(args.input_path):
        print("单张图像模式")
        dapi_file = get_dapi_file(args.input_path)
        if not dapi_file:
            print(f"错误：未找到对应的 DAPI 文件")
            return
        case_name = Path(args.input_path).stem.replace('_rotated', '')
        process_single_case(args.input_path, dapi_file, args.output_dir, case_name, show_preview)

    elif os.path.isdir(args.input_path):
        print("文件夹模式")
        process_folder(args.input_path, args.output_dir, show_preview)

    else:
        print(f"错误：路径不存在或无效：{args.input_path}")


if __name__ == "__main__":
    main()
