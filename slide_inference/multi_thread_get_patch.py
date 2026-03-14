import os
import math
import numpy as np
from PIL import Image
from KFBreader.kfbreader import KFBSlide
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import openslide

Image.MAX_IMAGE_PIXELS = None

def process_patch(args):
    """单个 patch 的处理函数（子进程里调用）"""
    slide_path, i, j, x, y, w, h, patch_size, save_dir = args
    patch_name = f"patch_{i}_{j}.png"
    save_path = os.path.join(save_dir, patch_name)

    # 如果文件已存在则跳过
    if os.path.exists(save_path):
        return "skipped"

    _, ext = os.path.splitext(slide_path)
    if ext == '.kfb':
        slide = KFBSlide(slide_path)
        patch = slide.read_region((x, y), 0, (w, h))
    else:
        slide = openslide.OpenSlide(slide_path)
        patch = slide.read_region((x, y), 0, (w, h))

    # 确保 patch 是 PIL.Image.Image
    if isinstance(patch, np.ndarray):
        patch = Image.fromarray(patch)
    elif not isinstance(patch, Image.Image):
        patch = Image.fromarray(np.array(patch))

    # 补齐到固定大小
    if w != patch_size or h != patch_size:
        new_patch = Image.new("RGB", (patch_size, patch_size), (255, 255, 255))
        new_patch.paste(patch, (0, 0, w, h))  # 用 4 元组明确区域
        patch = new_patch

    patch.save(save_path)
    slide.close()
    return "saved"


def slide_to_patches(slide_path, patch_size=1024, save_dir="patches", num_workers=4):
    os.makedirs(save_dir, exist_ok=True)

    # 读取整体尺寸
    _, ext = os.path.splitext(slide_path)
    if ext == '.kfb':
        slide = KFBSlide(slide_path)
    else:
        slide = openslide.OpenSlide(slide_path)
    slide_w, slide_h = slide.dimensions
    slide.close()

    n_cols = math.ceil(slide_w / patch_size)
    n_rows = math.ceil(slide_h / patch_size)

    tasks = []
    for i in range(n_rows):
        for j in range(n_cols):
            x = j * patch_size
            y = i * patch_size
            w = min(patch_size, slide_w - x)
            h = min(patch_size, slide_h - y)
            tasks.append((slide_path, i, j, x, y, w, h, patch_size, save_dir))

    total = len(tasks)
    print(f"总共有 {total} 个 patches 待处理...")

    counter = 0
    saved_count = 0
    skipped_count = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_patch, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            counter += 1
            if result == "saved":
                saved_count += 1
            else:
                skipped_count += 1

            if counter % 100 == 0 or counter == total:
                print(f"[{counter}/{total}] patches 已处理 (新保存: {saved_count}, 跳过: {skipped_count})", flush=True)

    print(f"✅ All patches done. 新保存: {saved_count}, 跳过: {skipped_count}", flush=True)


# 使用示例
# slide_to_patches(
#     '/nfs5/zsxm/2024.8.12广东省人民医院移动硬盘数据/肝癌/WSI/1552717-3-HE.svs',
#     patch_size=1024,
#     save_dir="/nfs5/lwh/guangdong-liver/1552717-3/patch",
#     num_workers=20
# )