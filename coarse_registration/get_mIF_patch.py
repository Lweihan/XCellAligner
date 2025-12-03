import os
import json
import tifffile
import numpy as np
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from PIL import Image

# ===================== #
# ====== 解析命令行参数 ===== #
# ===================== #

def parse_args():
    parser = argparse.ArgumentParser(description="切分多重荧光图像为 patches")
    parser.add_argument("--input_dir", type=str, help="多重荧光图像所在的文件夹路径")
    parser.add_argument("--save_dir", type=str, help="保存 patch 的文件夹路径")
    parser.add_argument("--patch_size", type=int, default=512, help="每个 patch 的大小，默认为 512")
    return parser.parse_args()

# ===================== #
# ====== 切割 Patch ===== #
# ===================== #

def save_patch(img_name, img, save_subdir, coords, patch_size, save_dir, channel_index, progress_callback=None):
    """保存单张图像的所有 patch"""
    sub_path = os.path.join(save_dir, save_subdir)
    os.makedirs(sub_path, exist_ok=True)
    
    for (x, y) in coords:
        patch = img[y:y+patch_size, x:x+patch_size]
        # 确保图像是正确的数据类型
        if patch.dtype != np.uint8:
            # 归一化到0-255范围并转换为uint8
            patch = ((patch - patch.min()) / (patch.max() - patch.min()) * 255).astype(np.uint8)
        
        # 使用PIL.Image保存PNG图像
        save_path = os.path.join(sub_path, f"mF{channel_index}_x{x}_y{y}.png")
        Image.fromarray(patch).save(save_path)
        if progress_callback:
            progress_callback()

def main():
    # 解析命令行输入的参数
    args = parse_args()
    
    # 检查必需的参数
    if not args.input_dir or not args.save_dir:
        print("错误: 必须提供 --input_dir 和 --save_dir 参数")
        return

    input_dir = args.input_dir
    save_dir = args.save_dir
    patch_size = args.patch_size

    os.makedirs(save_dir, exist_ok=True)

    # 获取输入文件夹中的所有 .tif 文件
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.tif')]
    image_files = sorted(image_files)  # 对文件列表进行排序
    images = {}

    # 读取所有图像并存储在字典中
    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)
        img_name = os.path.splitext(img_file)[0]
        images[img_name] = tifffile.imread(img_path)

    # 获取图像尺寸，假设所有图像的尺寸一致
    h, w = list(images.values())[0].shape[:2]
    
    # 生成所有 patch 的坐标
    coords = []
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            coords.append({"x": x, "y": y})

    # 保存坐标 JSON 文件
    json_path = os.path.join(save_dir, "patch_coords.json")
    with open(json_path, "w") as f:
        json.dump(coords, f, indent=2)
    print(f"✅ 坐标已保存到: {json_path}, 共 {len(coords)} 个 patch")

    # 总 patch 数量
    total_patches = len(coords) * len(images)

    # 使用多线程并行保存每个通道的 patch，并显示总体进度
    with ThreadPoolExecutor(max_workers=len(images)) as executor, \
         tqdm(total=total_patches, desc="Saving Patches", unit="patch") as pbar:

        def update_progress():
            pbar.update(1)

        futures = []
        for channel_index, (key, img) in enumerate(images.items()):
            futures.append(
                executor.submit(
                    save_patch,
                    key, img, key, [(c["x"], c["y"]) for c in coords],
                    patch_size, save_dir, channel_index, update_progress
                )
            )

        # 等待所有任务完成
        for future in as_completed(futures):
            future.result()

    print("✅ 所有通道的 patch 已保存完成！")

if __name__ == "__main__":
    main()