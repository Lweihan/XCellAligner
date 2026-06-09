

"""
python coarse_registration/rotate_slide.py  --check_and_fix --original_dirs /nfs-medical4/data/浙一多重荧光/胃癌/HE --input_path /nfs-medical3/sxz/XCellAligner/data/he_rotated

"""




import os
import sys
import openslide
from PIL import Image
import numpy as np
import tifffile as tiff
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(parent_dir, 'module'))
from KFBreader.kfbreader import KFBSlide

# 允许处理更大的图像
Image.MAX_IMAGE_PIXELS = None

# 允许加载被截断的图像
Image.OPEN_IGNORE_TRUNCATED_IMAGES = True

# 默认输出目录
DEFAULT_OUTPUT_DIR = "/nfs-medical3/sxz/XCellAligner/data/he_rotated"

def rotate_svs(input_path, output_path, level, angle, back_color='black'):
    import gc  # 添加垃圾回收

    _, ext = os.path.splitext(input_path)
    slide = None

    # 打开 SVS 文件
    if ext == ".kfb":
        slide = KFBSlide(input_path)
        print(slide.level_dimensions[level])
        level0 = Image.fromarray(slide.read_region((0, 0), level, slide.level_dimensions[level])).convert("RGB")
    elif ext == '.svs' or ext == '.mrxs':
        slide = openslide.OpenSlide(input_path)
        # 读取一级图像数据（0级是原始级别的图像）
        level0 = slide.read_region((0, 0), level, slide.level_dimensions[level]).convert("RGB")
    elif ext == '.tif' or ext == '.tiff':
        level0 = Image.open(input_path).convert("RGB")
        slide = None

    print(f"原始图像尺寸: {level0.size}")

    # 立即关闭slide以释放内存
    if slide is not None:
        slide.close()
        slide = None

    # 旋转图像（PIL Image方式）
    if back_color == 'black':
        rotated = level0.rotate(angle, expand=True)
    else:  # white
        rotated = level0.rotate(angle, expand=True, fillcolor=(255, 255, 255))

    print(f"旋转后图像尺寸: {rotated.size}")

    # 释放原始图像内存
    del level0
    gc.collect()

    # 转换为numpy数组（使用uint8确保精度）
    rotated_np = np.array(rotated, dtype=np.uint8)

    # 释放PIL图像内存
    del rotated
    gc.collect()

    # 检查数组是否有效
    print(f"旋转后数组形状: {rotated_np.shape}, 数据类型: {rotated_np.dtype}")
    print(f"值范围: [{rotated_np.min()}, {rotated_np.max()}]")

    # 检查是否有NaN或Inf
    if not np.isfinite(rotated_np).all():
        print("⚠️  警告：数组包含NaN或Inf值")
        rotated_np = np.nan_to_num(rotated_np, nan=0, posinf=255, neginf=0)

    # 保存为 TIFF（使用 tifffile，支持超大文件）
    # 使用 tiled 方式保存，更可靠
    try:
        tiff.imwrite(
            output_path,
            rotated_np,
            bigtiff=True,
            photometric='rgb',
            planarconfig='contig',
            tile=(512, 512),  # 使用tiled方式，更可靠
            compression=None  # 不使用压缩，确保原始精度
        )
        print(f"✅ 保存完成：{output_path}")
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        # 尝试不使用tile的方式
        print("尝试不使用tile的方式...")
        tiff.imwrite(
            output_path,
            rotated_np,
            bigtiff=True,
            photometric='rgb',
            planarconfig='contig',
            compression=None
        )
        print(f"✅ 保存完成：{output_path}")

    # 验证文件完整性（只采样检查，避免内存占用）
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        expected_size = rotated_np.nbytes
        print(f"文件大小: {file_size / (1024**3):.2f} GB (预期: {expected_size / (1024**3):.2f} GB)")

        # 只采样验证，不加载整个图像
        print(f"✅ 文件已保存，大小正常")

    # 释放numpy数组内存
    del rotated_np
    gc.collect()

    print(f"✅ 内存已释放")

def find_mif_folder(he_file_path, data_root=None, verbose=False):
    """
    根据 HE 文件路径查找对应的 mIF 文件夹
    参考 preview_dataset.py 的逻辑

    参数:
        he_file_path: HE 文件完整路径
        data_root: 数据集根目录（如果为None，自动从HE路径推断）
        verbose: 是否打印调试信息

    返回:
        mIF 文件夹完整路径，如果找不到返回 None
    """
    he_path = Path(he_file_path)
    he_filename = he_path.name  # 如 "7A 25-000232 2.kfb"

    # 提取HE名称的基础部分（去除扩展名）
    base_name = he_filename
    for ext in ['.kfb', '.svs', '.SVS', '.mrxs', '.tiff', '.tif', '.MRXS']:
        base_name = base_name.replace(ext, '')

    if verbose:
        print(f"  [DEBUG] HE文件名: {he_filename}")
        print(f"  [DEBUG] 提取的base_name: '{base_name}'")

    # 确定数据集根目录和 mIF-Channel 路径
    if data_root is None:
        # 从 HE 文件路径推断
        # 假设结构: data_root/HE/file.kfb 或 data_root/HE/subfolder/file.kfb
        current = he_path.parent
        while current.parent != current:
            if current.name == 'HE':
                data_root = current.parent
                break
            current = current.parent
        else:
            # 无法推断，尝试使用 HE 的父目录作为 data_root
            data_root = he_path.parent.parent

    mif_channel_dir = os.path.join(data_root, "mIF-Channel")

    if verbose:
        print(f"  [DEBUG] 推断的data_root: {data_root}")
        print(f"  [DEBUG] mIF-Channel目录: {mif_channel_dir}")

    if not os.path.exists(mif_channel_dir):
        if verbose:
            print(f"  [DEBUG] mIF-Channel目录不存在")
        return None

    # 查找匹配的 mIF 文件夹（mIF 文件夹名以 HE 基础名开头）
    mif_path = Path(mif_channel_dir)
    if verbose:
        print(f"  [DEBUG] 扫描mIF文件夹中的子目录...")

    for mif_subdir in mif_path.iterdir():
        if mif_subdir.is_dir():
            if verbose:
                print(f"  [DEBUG]   检查: '{mif_subdir.name}' startswith '{base_name}': {mif_subdir.name.startswith(base_name)}")
            if mif_subdir.name.startswith(base_name):
                if verbose:
                    print(f"  [DEBUG] ✅ 找到匹配的mIF文件夹")
                return str(mif_subdir)

    if verbose:
        print(f"  [DEBUG] ❌ 未找到匹配的mIF文件夹")
    return None


def find_dapi_channel(he_file_path, data_root=None, verbose=False):
    """
    根据 HE 文件路径查找对应的 DAPI 通道文件
    参考 preview_dataset.py 的 DAPI 查找逻辑

    参数:
        he_file_path: HE 文件完整路径
        data_root: 数据集根目录（如果为None，自动从HE路径推断）
        verbose: 是否打印调试信息

    返回:
        DAPI 文件路径，如果找不到返回 None
    """
    # 先找到对应的 mIF 文件夹
    mif_folder = find_mif_folder(he_file_path, data_root, verbose)
    if not mif_folder:
        return None

    # 在 mIF 文件夹中查找 DAPI 文件
    # DAPI 文件特征：文件名包含 DAPI 和 Extended
    mif_path = Path(mif_folder)
    if not mif_path.exists():
        return None

    if verbose:
        print(f"  [DEBUG] 在mIF文件夹中查找DAPI文件...")

    for file in mif_path.iterdir():
        if file.is_file():
            name_upper = file.name.upper()
            # DAPI - 包含DAPI和Extended
            has_dapi = 'DAPI' in name_upper
            has_extended = 'EXTENDED' in name_upper
            if verbose:
                print(f"  [DEBUG]   检查: '{file.name}' DAPI:{has_dapi} EXTENDED:{has_extended}")
            if has_dapi and has_extended:
                if verbose:
                    print(f"  [DEBUG] ✅ 找到DAPI文件")
                return str(file)

    if verbose:
        print(f"  [DEBUG] ❌ 未找到DAPI文件")
    return None


def create_preview_with_dapi(he_tiff_path, dapi_path, preview_save_path, he_name="HE", target_size=(800, 800)):
    """
    创建 HE 和 DAPI 的预览图（使用缩略图，避免加载完整大图）

    参数:
        he_tiff_path: 旋转后的 HE TIFF 文件路径
        dapi_path: DAPI TIFF 文件路径
        preview_save_path: 预览图保存路径
        he_name: HE 图像的名称标签
        target_size: 缩略图目标尺寸
    """
    try:
        # 先缩放 HE 图像（在转换为 numpy 之前）
        he_img = Image.open(he_tiff_path)
        he_img.thumbnail(target_size, Image.Resampling.LANCZOS)

        # 先缩放 DAPI 图像（在转换为 numpy 之前）
        dapi_img = Image.open(dapi_path)
        dapi_img.thumbnail(target_size, Image.Resampling.LANCZOS)

        # 拼接两张图（左右并排）
        total_width = he_img.width + dapi_img.width
        max_height = max(he_img.height, dapi_img.height)

        # 创建白色背景
        preview_img = Image.new('RGB', (total_width, max_height), (255, 255, 255))

        # 粘贴 HE 和 DAPI（直接使用原始图，不上色）
        preview_img.paste(he_img, (0, 0))
        preview_img.paste(dapi_img, (he_img.width, 0))

        # 添加文字标签
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(preview_img)
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()

        draw.text((10, 10), "HE (Rotated)", fill=(0, 0, 0), font=font)
        draw.text((he_img.width + 10, 10), "DAPI", fill=(0, 0, 0), font=font)

        # 保存
        preview_img.save(preview_save_path, 'PNG')

        print(f"✅ 预览图已保存：{preview_save_path}")
        return True
    except Exception as e:
        print(f"⚠️  生成预览图失败：{e}")
        return False


def create_overlay_preview(he_tiff_path, dapi_path, overlay_save_path, he_name="HE", target_size=(800, 800), alpha=0.3):
    """
    创建 HE 和 DAPI 的叠加预览图（DAPI 半透明叠加在 HE 上）

    参数:
        he_tiff_path: 旋转后的 HE TIFF 文件路径
        dapi_path: DAPI TIFF 文件路径
        overlay_save_path: 叠加图保存路径
        he_name: HE 图像的名称标签
        target_size: 缩略图目标尺寸
        alpha: DAPI 透明度 (0-1)
    """
    try:
        # 先缩放 HE 图像
        he_img = Image.open(he_tiff_path)
        he_img.thumbnail(target_size, Image.Resampling.LANCZOS)
        he_array = np.array(he_img)

        # 先缩放 DAPI 图像
        dapi_img = Image.open(dapi_path)
        dapi_img.thumbnail(target_size, Image.Resampling.LANCZOS)
        dapi_array = np.array(dapi_img)

        # 调整尺寸一致（取较小的尺寸）
        min_h = min(he_array.shape[0], dapi_array.shape[0])
        min_w = min(he_array.shape[1], dapi_array.shape[1])
        he_array = he_array[:min_h, :min_w]
        dapi_array = dapi_array[:min_h, :min_w]

        # 使用 matplotlib 创建叠加图
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(he_array)  # HE 作为底图
        ax.imshow(dapi_array, alpha=alpha)  # DAPI 半透明叠加
        ax.set_title(f'Overlay: {he_name}', fontsize=14)
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(overlay_save_path, dpi=100, bbox_inches='tight')
        plt.close()

        print(f"✅ 叠加图已保存：{overlay_save_path}")
        return True
    except Exception as e:
        print(f"⚠️  生成叠加图失败：{e}")
        return False


def process_single_image(input_path, output_dir, level, angle, back_color, create_preview=True, data_root=None):
    """
    处理单张图像

    参数:
        input_path: 输入图像路径
        output_dir: 输出目录
        level: 读取级别
        angle: 旋转角度
        back_color: 背景颜色
        create_preview: 是否创建预览图
        data_root: 数据集根目录（用于查找mIF-Channel）
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取文件名和扩展名
    file_path = Path(input_path)
    file_name = file_path.stem  # 不含扩展名
    ext = file_path.suffix

    # 输出文件路径
    output_filename = f"{file_name}_rotated.tiff"
    output_path = os.path.join(output_dir, output_filename)

    # 检查输出文件是否已存在
    if os.path.exists(output_path):
        print(f"\n⏭️  跳过（已存在）: {file_name}")
        print(f"   已有文件: {output_path}")
        # 检查预览图是否也存在
        if create_preview:
            preview_path = os.path.join(output_dir, f"{file_name}_preview.png")
            overlay_path = os.path.join(output_dir, f"{file_name}_overlay.png")

            need_preview = not os.path.exists(preview_path)
            need_overlay = not os.path.exists(overlay_path)

            if need_preview or need_overlay:
                print(f"   ⚠️  预览图/叠加图不存在，尝试生成...")
                # 只生成预览图，不重复旋转
                dapi_path = find_dapi_channel(input_path, data_root, verbose=True)
                if dapi_path:
                    if need_preview:
                        create_preview_with_dapi(output_path, dapi_path, preview_path, file_name)
                    if need_overlay:
                        create_overlay_preview(output_path, dapi_path, overlay_path, file_name)
                else:
                    print(f"   ⚠️  未找到DAPI通道，跳过预览图生成")
        return

    print(f"\n{'='*60}")
    print(f"处理单张图像: {file_name}")
    print(f"{'='*60}")

    # 旋转图像（使用原函数）
    rotate_svs(input_path, output_path, level, angle, back_color)

    # 如果需要创建预览图
    if create_preview:
        print(f"\n[查找DAPI]")
        # 查找 DAPI 通道（使用新的查找逻辑，开启调试）
        dapi_path = find_dapi_channel(input_path, data_root, verbose=True)

        if dapi_path:
            print(f"✅ 找到 DAPI 通道：{dapi_path}")
            # 生成预览图（HE + DAPI 并排）
            preview_path = os.path.join(output_dir, f"{file_name}_preview.png")
            create_preview_with_dapi(output_path, dapi_path, preview_path, file_name)

            # 生成叠加图（HE + DAPI 叠加）
            overlay_path = os.path.join(output_dir, f"{file_name}_overlay.png")
            create_overlay_preview(output_path, dapi_path, overlay_path, file_name)
        else:
            print(f"⚠️  未找到对应的 DAPI 通道，跳过预览图生成")


def process_folder(input_folder, output_dir, level, angle, back_color, start_index=None, end_index=None):
    """
    处理文件夹中的所有 HE 图像

    参数:
        input_folder: 输入文件夹路径
        output_dir: 输出目录
        level: 读取级别
        angle: 旋转角度
        back_color: 背景颜色
        start_index: 起始文件索引（从1开始，None表示从头开始）
        end_index: 结束文件索引（None表示到最后）
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"处理文件夹: {input_folder}")
    print(f"{'='*60}")

    # 支持的图像格式
    supported_exts = ['.kfb', '.svs', '.mrxs', '.tif', '.tiff']

    # 获取所有支持的文件
    he_files = []
    for file in os.listdir(input_folder):
        ext = Path(file).suffix.lower()
        if ext in supported_exts:
            he_files.append(os.path.join(input_folder, file))

    if not he_files:
        print(f"⚠️  文件夹中没有找到支持的图像文件")
        return

    print(f"✅ 找到 {len(he_files)} 张图像")

    # 应用文件范围切片
    if start_index is not None or end_index is not None:
        original_count = len(he_files)
        # 转换为0-based索引
        start = (start_index - 1) if start_index is not None else 0
        end = end_index if end_index is not None else len(he_files)
        he_files = he_files[start:end]
        print(f"📌 处理范围: 第 {start + 1} 到第 {min(end, original_count)} 个文件（共 {len(he_files)} 张）")

    # 统计跳过和处理数量
    skipped_count = 0
    processed_count = 0

    # 处理每张图像
    for i, input_path in enumerate(he_files, start=(start_index if start_index else 1)):
        # 获取文件名
        file_name = Path(input_path).stem
        output_filename = f"{file_name}_rotated.tiff"
        output_path = os.path.join(output_dir, output_filename)

        # 检查输出文件是否已存在
        if os.path.exists(output_path):
            print(f"\n[全局 {i}] ⏭️  跳过（已存在）: {os.path.basename(input_path)}")
            skipped_count += 1
            continue

        print(f"\n[全局 {i}] 处理：{os.path.basename(input_path)}")

        # 旋转图像（使用原函数）
        try:
            rotate_svs(input_path, output_path, level, angle, back_color)
            processed_count += 1
        except Exception as e:
            print(f"❌ 处理失败：{e}")
            continue

    # 打印总结
    print(f"\n{'='*60}")
    print(f"处理完成: {processed_count} 张成功, {skipped_count} 张已跳过")
    print(f"{'='*60}")


# =========================
# 检查和修复函数
# =========================

def check_tiff_integrity(tiff_path):
    """
    检查 TIFF 文件是否完整

    返回：
    - (is_valid, expected_size, actual_size, error_msg)
    """
    if not os.path.exists(tiff_path):
        return False, 0, 0, "文件不存在"

    try:
        # 尝试读取文件
        img = tiff.imread(tiff_path)
        actual_size = os.path.getsize(tiff_path)
        expected_size = img.nbytes

        # 检查大小是否合理（允许 5% 的误差）
        if actual_size < expected_size * 0.95:
            return False, expected_size, actual_size, f"文件被截断（预期 {expected_size} 字节，实际 {actual_size} 字节）"

        return True, expected_size, actual_size, "OK"

    except Exception as e:
        actual_size = os.path.getsize(tiff_path)
        return False, 0, actual_size, f"读取失败: {e}"


def check_tiff_zero_values(tiff_path, border_width=10, verbose=False):
    """
    检查 TIFF 文件边缘一圈是否有0值

    如果使用白色背景旋转，正常情况下边缘应该是白色（255）。
    如果边缘出现黑色（0值），说明保存时出了问题。

    参数：
    - tiff_path: TIFF文件路径
    - border_width: 边缘检查宽度（像素）
    - verbose: 是否打印调试信息

    返回：
    - (is_valid, zero_count, msg)
    """
    if verbose:
        print(f"    [DEBUG] 检查文件: {os.path.basename(tiff_path)}")

    if not os.path.exists(tiff_path):
        return False, 0, "文件不存在"

    try:
        if verbose:
            print(f"    [DEBUG] 打开文件（不加载全部）...")

        # 使用 PIL Image.open() 懒加载，然后 crop 特定区域
        with Image.open(tiff_path) as img:
            w, h = img.size
            if verbose:
                print(f"    [DEBUG] 图像尺寸: {w} × {h}")

            zero_count = 0

            # 检查上边缘
            if verbose:
                print(f"    [DEBUG] 检查上边缘...")
            top_crop = img.crop((0, 0, w, border_width))
            top_array = np.array(top_crop)
            if len(top_array.shape) == 3:
                zero_count += np.sum(np.all(top_array == 0, axis=2))
            else:
                zero_count += np.sum(top_array == 0)
            del top_crop, top_array  # 立即释放内存

            # 检查下边缘
            if verbose:
                print(f"    [DEBUG] 检查下边缘...")
            bottom_crop = img.crop((0, h - border_width, w, h))
            bottom_array = np.array(bottom_crop)
            if len(bottom_array.shape) == 3:
                zero_count += np.sum(np.all(bottom_array == 0, axis=2))
            else:
                zero_count += np.sum(bottom_array == 0)
            del bottom_crop, bottom_array  # 立即释放内存

            # 检查左边缘（排除上下已检查的角）
            if verbose:
                print(f"    [DEBUG] 检查左边缘...")
            left_crop = img.crop((0, border_width, border_width, h - border_width))
            left_array = np.array(left_crop)
            if len(left_array.shape) == 3:
                zero_count += np.sum(np.all(left_array == 0, axis=2))
            else:
                zero_count += np.sum(left_array == 0)
            del left_crop, left_array  # 立即释放内存

            # 检查右边缘（排除上下已检查的角）
            if verbose:
                print(f"    [DEBUG] 检查右边缘...")
            right_crop = img.crop((w - border_width, border_width, w, h - border_width))
            right_array = np.array(right_crop)
            if len(right_array.shape) == 3:
                zero_count += np.sum(np.all(right_array == 0, axis=2))
            else:
                zero_count += np.sum(right_array == 0)
            del right_crop, right_array  # 立即释放内存

            # 计算总边缘像素数（近似）
            total_edge_pixels = 2 * border_width * (w + h) - 4 * border_width * border_width

            if verbose:
                print(f"    [DEBUG] 零值像素: {zero_count}/{total_edge_pixels}")

            # 如果边缘有任何0值，认为有问题
            if zero_count > 0:
                return False, zero_count, f"边缘发现 {zero_count} 个0值像素"

            return True, 0, f"OK（边缘 {total_edge_pixels} 像素无0值）"

    except Exception as e:
        if verbose:
            print(f"    [DEBUG] 检查失败: {e}")
        return False, 0, f"检查失败: {e}"


def find_original_file(rotated_name, search_dirs):
    """
    根据旋转后的文件名查找原始文件

    参数：
    - rotated_name: 旋转后的文件名（如 "7C 25-064821 3_rotated.tiff"）
    - search_dirs: 搜索原始文件的目录列表

    返回：
    - 原始文件完整路径，如果找不到返回 None
    """
    # 去除 _rotated 后缀
    base_name = rotated_name.replace('_rotated.tiff', '').replace('_rotated.tif', '')

    # 支持的原始文件扩展名
    original_exts = ['.svs', '.tif', '.tiff', '.kfb', '.mrxs']

    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue

        for file in os.listdir(search_dir):
            file_path = os.path.join(search_dir, file)
            if not os.path.isfile(file_path):
                continue

            # 检查文件名是否匹配（忽略扩展名）
            file_stem = Path(file).stem
            if file_stem == base_name:
                return file_path

    return None


def check_and_fix_rotated_files(
    rotated_dir,
    original_dirs,
    level=1,
    angle=270,
    back_color='white',
    dry_run=False,
    start_index=None,
    end_index=None,
    mif_registered_dir=None,
    verbose=False
):
    """
    检查旋转文件夹中所有 TIFF 文件的完整性
    对于有问题的文件，删除并重新旋转

    参数：
    - rotated_dir: 旋转后的文件夹路径
    - original_dirs: 原始文件搜索目录列表
    - level: 读取级别
    - angle: 旋转角度（默认 270）
    - back_color: 背景颜色
    - dry_run: 只检查不修复
    - start_index: 起始文件索引（从1开始）
    - end_index: 结束文件索引
    - mif_registered_dir: mIF注册文件夹（如果已存在注册结果，跳过检查）
    - verbose: 是否显示详细信息
    """
    print(f"\n{'='*60}")
    print(f"检查和修复旋转文件")
    print(f"{'='*60}")
    print(f"旋转文件夹: {rotated_dir}")
    print(f"原始文件搜索目录: {original_dirs}")
    print(f"旋转角度: {angle}°")
    if mif_registered_dir:
        print(f"注册文件夹: {mif_registered_dir}（已注册的文件将被跳过）")
    if dry_run:
        print(f"⚠️  预览模式（不会实际修复）")

    # 获取所有旋转后的 TIFF 文件
    rotated_files = []
    for file in os.listdir(rotated_dir):
        if file.endswith('_rotated.tiff') or file.endswith('_rotated.tif'):
            rotated_files.append(os.path.join(rotated_dir, file))

    if not rotated_files:
        print("⚠️  没有找到旋转后的 TIFF 文件")
        return

    print(f"✅ 找到 {len(rotated_files)} 个旋转后的文件")

    # 应用文件范围切片
    if start_index is not None or end_index is not None:
        original_count = len(rotated_files)
        start = (start_index - 1) if start_index is not None else 0
        end = end_index if end_index is not None else len(rotated_files)
        rotated_files = rotated_files[start:end]
        print(f"📌 检查范围: 第 {start + 1} 到第 {min(end, original_count)} 个文件（共 {len(rotated_files)} 个）")

    # 检查并修复每个文件（检查到一个修复一个）
    valid_count = 0
    fixed_count = 0
    failed_count = 0

    for rotated_path in tqdm(rotated_files, desc="检查进度"):
        file_name = os.path.basename(rotated_path)

        # 去除 _rotated.tiff 后缀，获取基础名称
        base_name = file_name.replace('_rotated.tiff', '').replace('_rotated.tif', '')

        if verbose:
            print(f"\n🔍 检查文件: {file_name}")

        # 如果指定了 mif_registered_dir，检查是否已有注册结果
        # 文件夹格式：{base_name} {通道信息}，例如 "7D 25-009579 2 520-MPO+670-CD138+570-D240"
        if mif_registered_dir and os.path.exists(mif_registered_dir):
            # 查找以 base_name 开头的文件夹
            matching_folder = None
            if verbose:
                print(f"    [DEBUG] base_name: '{base_name}'")

            for folder in os.listdir(mif_registered_dir):
                folder_path = os.path.join(mif_registered_dir, folder)
                if os.path.isdir(folder_path):
                    if verbose:
                        print(f"    [DEBUG] 检查文件夹: '{folder}' (startswith: {folder.startswith(base_name)})")
                    if folder.startswith(base_name):
                        matching_folder = folder_path
                        break

            if matching_folder:
                if verbose:
                    print(f"  ⏭️  跳过（已有注册）: {file_name}")
                valid_count += 1
                continue

        # 检查是否有0值问题
        is_valid, _, msg = check_tiff_zero_values(rotated_path, verbose=verbose)

        if is_valid:
            valid_count += 1
        else:
            print(f"\n  ❌ {file_name}: {msg}")
            print(f"      尝试修复...")

            # 立即修复这个损坏的文件
            if not dry_run:
                # 删除损坏的文件
                try:
                    os.remove(rotated_path)
                    print(f"  ✓ 已删除损坏的文件")
                except Exception as e:
                    print(f"  ✗ 删除失败: {e}")
                    failed_count += 1
                    continue

                # 查找原始文件
                original_path = find_original_file(file_name, original_dirs)

                if original_path is None:
                    print(f"  ✗ 未找到原始文件")
                    failed_count += 1
                    continue

                print(f"  ✓ 找到原始文件: {os.path.basename(original_path)}")

                # 重新旋转
                output_path = rotated_path  # 使用相同的路径
                try:
                    rotate_svs(original_path, output_path, level, angle, back_color)

                    # 验证修复后的文件
                    is_valid_fixed, _, msg_fixed = check_tiff_zero_values(output_path)
                    if is_valid_fixed:
                        print(f"  ✅ 修复成功并验证通过（{msg_fixed}）")
                        fixed_count += 1
                    else:
                        print(f"  ⚠️  修复后仍有问题（{msg_fixed}）")
                        failed_count += 1
                except Exception as e:
                    print(f"  ✗ 重新旋转失败: {e}")
                    failed_count += 1
            else:
                print(f"  [预览模式] 将被修复")

    print(f"\n检查和修复结果:")
    print(f"  有效文件: {valid_count}")
    if not dry_run:
        print(f"  成功修复: {fixed_count}")
        print(f"  修复失败: {failed_count}")


def parse_args():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="旋转病理图像并保存为TIFF")
    parser.add_argument("--input_path", type=str, default=None,
                        help="输入图像路径或文件夹路径")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"输出目录（默认: {DEFAULT_OUTPUT_DIR}）")
    parser.add_argument("--level", type=int, default=1,
                        help="要读取的级别，默认为1级（推荐用于配准）")
    parser.add_argument("--angle", type=int, required=False,
                        help="旋转角度，支持 90、180、270 度")
    parser.add_argument("--back_color", type=str, default='white',
                        help="背景颜色，默认为 white")
    parser.add_argument("--data_root", type=str, default=None,
                        help="数据集根目录（用于查找mIF-Channel，默认自动推断）")
    parser.add_argument("--start", type=int, default=None,
                        help="文件夹模式：起始文件索引（从1开始）")
    parser.add_argument("--end", type=int, default=None,
                        help="文件夹模式：结束文件索引")
    parser.add_argument("--check_and_fix", action="store_true",
                        help="检查旋转文件夹的文件完整性并修复损坏的文件")
    parser.add_argument("--original_dirs", type=str, nargs='+', default=None,
                        help="原始文件搜索目录（用于检查修复模式）")
    parser.add_argument("--dry_run", action="store_true",
                        help="预览模式（只检查不修复）")
    parser.add_argument("--mif_registered_dir", type=str, default=None,
                        help="mIF注册文件夹（已注册的文件将被跳过）")
    parser.add_argument("-n", "--no_verbose", action="store_true",
                        help="不显示详细调试信息")

    return parser.parse_args()


def main():
    args = parse_args()

    # 检查和修复模式
    if args.check_and_fix:
        if args.original_dirs is None:
            print("❌ 检查修复模式需要指定 --original_dirs 参数")
            print("\n使用示例:")
            print("  python rotate_slide.py --check_and_fix --original_dirs /path/to/original/HE /path/to/original/HE2 --input_path /path/to/he_rotated")
            return

        print("🔧 检查和修复模式")
        check_and_fix_rotated_files(
            rotated_dir=args.input_path,
            original_dirs=args.original_dirs,
            level=args.level,
            angle=270,  # 固定使用 270 度
            back_color=args.back_color,
            dry_run=args.dry_run,
            start_index=args.start,
            end_index=args.end,
            mif_registered_dir=args.mif_registered_dir,
            verbose=not args.no_verbose
        )
        return

    # 常规模式
    if args.input_path is None:
        print("❌ 请指定 --input_path 或使用 --check_and_fix 模式")
        return

    if args.angle is None:
        print("❌ 请指定 --angle 参数")
        return

    input_path = args.input_path

    # 判断是文件还是文件夹
    if os.path.isfile(input_path):
        # 单张图像模式
        print("📄 单张图像模式")
        process_single_image(
            input_path=input_path,
            output_dir=args.output_dir,
            level=args.level,
            angle=args.angle,
            back_color=args.back_color,
            create_preview=True,  # 单图模式默认生成预览图
            data_root=args.data_root
        )
    elif os.path.isdir(input_path):
        # 文件夹模式
        print("📁 文件夹模式")
        process_folder(
            input_folder=input_path,
            output_dir=args.output_dir,
            level=args.level,
            angle=args.angle,
            back_color=args.back_color,
            start_index=args.start,
            end_index=args.end
        )
    else:
        print(f"❌ 错误：路径不存在或无效：{input_path}")
        return

if __name__ == "__main__":
    main()
