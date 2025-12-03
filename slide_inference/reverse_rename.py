import os
import re
import math
from skimage.transform import resize
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import matplotlib.pyplot as plt
from PIL import Image
import openslide
from tqdm import tqdm
from module.KFBreader.kfbreader import KFBSlide

def revserse_rename(slide_path, src_dir, type="patch"):
    _, ext = os.path.splitext(slide_path)
    if ext == '.kfb':
        slide = KFBSlide(slide_path)
    else:
        slide = openslide.OpenSlide(slide_path)
    slide_w, slide_h = slide.dimensions
    patch_size = 1024
    slide.close()
    n_cols = math.ceil(slide_w / patch_size)
    if type == "patch":
        pattern = re.compile(r".*_(\d{5})_0000\.png")
    else:
        pattern = re.compile(r".*_(\d{5})\.png")

    patch_coord = []

    for f in tqdm(os.listdir(src_dir)):
        m = pattern.match(f)
        if not m:
            continue
        patch_idx = int(m.group(1))
        i = patch_idx // n_cols
        j = patch_idx % n_cols
        patch_coord.append((f, i, j))

    # 输出或保存结果
    for f, i, j in tqdm(patch_coord):
        old_path = os.path.join(src_dir, f)
        new_name = f"patch_{i}_{j}.png"
        new_path = os.path.join(src_dir, new_name)
        os.rename(old_path, new_path)