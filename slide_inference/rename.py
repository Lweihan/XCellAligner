import os
import re
from tqdm import tqdm
import math
import openslide
from module.KFBreader.kfbreader import KFBSlide

def rename_patch(slide_path, src_dir, case_id="case"):
    _, ext = os.path.splitext(slide_path)
    if ext == '.kfb':
        slide = KFBSlide(slide_path)
    else:
        slide = openslide.OpenSlide(slide_path)
    slide_w, slide_h = slide.dimensions
    patch_size = 1024
    slide.close()
    n_cols = math.ceil(slide_w / patch_size)
    n_rows = math.ceil(slide_h / patch_size)

    # 匹配 patch_i_j.png
    pattern = re.compile(r"patch_(\d+)_(\d+)\.png")

    files = [f for f in os.listdir(src_dir) if f.endswith(".png")]
    patch_info = []

    for f in files:
        m = pattern.match(f)
        if not m:
            continue
        i, j = int(m.group(1)), int(m.group(2))
        idx = i * n_cols + j
        patch_info.append((idx, f))

    # 按 idx 排序
    patch_info.sort(key=lambda x: x[0])

    # 重命名
    for idx, (patch_idx, fname) in tqdm(enumerate(patch_info), total=len(patch_info)):
        src_path = os.path.join(src_dir, fname)
        dst_fname = f"{case_id}_{patch_idx:05d}_0000.png"  # 用 5 位编号，避免 8000+ 溢出
        dst_path = os.path.join(src_dir, dst_fname)
        os.rename(src_path, dst_path)