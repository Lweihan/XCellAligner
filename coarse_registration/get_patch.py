import os
from PIL import Image
import numpy as np
import concurrent.futures
from tqdm import tqdm

# 设置PIL允许加载非常大的图像
Image.MAX_IMAGE_PIXELS = None

def is_patch_valid(he_patch: np.array, black_threshold=10, white_threshold=245, std_threshold=5.0, black_area_threshold=0.5):
    """判断 HE patch 是否有效（非纯黑/纯白/无纹理）"""
    mean_val = he_patch.mean()
    std_val = he_patch.std()
    
    # 判断是否是纯黑或纯白
    if mean_val < black_threshold or mean_val > white_threshold:
        return False
    
    # 判断标准差，排除无纹理的区域
    if std_val < std_threshold:
        return False
    
    # 判断黑色区域的比例是否超过阈值
    black_area = np.sum(he_patch < black_threshold) / he_patch.size
    if black_area > black_area_threshold:
        return False
    
    return True


def process_and_save_patches(args):
    """处理单个 patch（HE + 所有 mIF）"""
    he_data, mif_paths, x, y, patch_size, output_dir, sorted_channels = args
    
    # 裁剪 HE patch 并验证
    he_patch = he_data[y:y + patch_size, x:x + patch_size]
    if not is_patch_valid(he_patch):
        return  # 跳过无效区域

    # 保存 HE
    he_name = f"he_x{x}_y{y}.png"
    he_path = os.path.join(output_dir, "he", he_name)
    os.makedirs(os.path.dirname(he_path), exist_ok=True)
    he_patch_img = Image.fromarray(he_patch).resize((512, 512))
    he_patch_img.save(he_path)

    # 保存所有 mIF patches
    for idx, channel in enumerate(sorted_channels):
        mif_patch = mif_paths[channel][y:y + patch_size, x:x + patch_size]
        mif_name = f"mF{idx}_x{x}_y{y}.png"
        mif_dir = os.path.join(output_dir, f"{idx}-{channel}")
        os.makedirs(mif_dir, exist_ok=True)
        mif_patch_img = Image.fromarray(mif_patch).resize((512, 512))
        mif_patch_img.save(os.path.join(mif_dir, mif_name))


def process_whole_slide(he_image_path, mif_dir, output_dir, patch_size=512, max_workers=8):
    # 1. 获取 HE 图像数据
    with Image.open(he_image_path) as he_img:
        he_data = np.array(he_img)

    # 2. 收集并排序 mIF 文件，并读取到内存
    mif_files = [f for f in os.listdir(mif_dir) if f.endswith('.png')]
    mif_channels = []
    mif_data = {}
    for f in mif_files:
        if '-' in f:
            channel = f.split('-', 1)[1].rsplit('.', 1)[0]
        else:
            channel = f.rsplit('.', 1)[0]
        mif_channels.append(channel)
    
    if not mif_files:
        raise ValueError(f"mIF 目录 {mif_dir} 中未找到 PNG 文件！")

    # 按 channel 排序以确保 idx 稳定
    sorted_pairs = sorted(zip(mif_channels, mif_files))
    sorted_channels, sorted_files = zip(*sorted_pairs)
    
    # 读取所有 mIF 图像数据到内存
    for ch, file in zip(sorted_channels, sorted_files):
        with Image.open(os.path.join(mif_dir, file)) as mif_img:
            mif_data[ch] = np.array(mif_img)

    # 3. 构建任务列表（只包含完整 patch）
    tasks = []
    if len(he_data.shape) == 3:  # RGB 图像（height, width, 3）
        height, width, _ = he_data.shape
    else:  # 灰度图像（height, width）
        height, width = he_data.shape
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            if x + patch_size <= width and y + patch_size <= height:
                tasks.append((
                    he_data,  # 使用已加载的 HE 图像数据
                    mif_data,  # 已加载的 mIF 图像数据
                    x, y,
                    patch_size,
                    output_dir,
                    sorted_channels
                ))

    print(f"总 patch 数量（完整）: {len(tasks)}")
    os.makedirs(os.path.join(output_dir, "he"), exist_ok=True)

    # 4. 多线程处理 + tqdm 进度条
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 使用 tqdm 包装 map 结果
        list(tqdm(
            executor.map(process_and_save_patches, tasks),
            total=len(tasks),
            desc="切分图像 Patch",
            unit="patch"
        ))

    print("✅ 图像切分完成！")


# ========================
# 使用示例
# ========================
if __name__ == "__main__":
    he_path = "/c23227/lwh/dataset/复旦中山医院肝内胆管癌/Registed/24S026138/24S026138-he.png"
    mif_folder = "/c23227/lwh/dataset/复旦中山医院肝内胆管癌/Registed/24S026138/mif"
    output_root = "/c23227/lwh/dataset/复旦中山医院肝内胆管癌/patch/24S026138"

    process_whole_slide(
        he_image_path=he_path,
        mif_dir=mif_folder,
        output_dir=output_root,
        patch_size=192,
        max_workers=8
    )
