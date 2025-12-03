import os
import json
import tifffile
import numpy as np
from PIL import Image
from tqdm import tqdm
import SimpleITK as sitk
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import traceback
from threading import Lock

Image.MAX_IMAGE_PIXELS = None

# 全局锁，用于线程安全的打印
print_lock = Lock()

def thread_safe_print(message):
    with print_lock:
        print(message)

# === 单通道 RGB 重采样函数 ===
def resample_rgb_image(image_np, reference_sitk, transform):
    registered_channels = []
    for i in range(3):
        channel = sitk.GetImageFromArray(image_np[..., i])
        channel.CopyInformation(reference_sitk)
        registered = sitk.Resample(channel, reference_sitk, transform, sitk.sitkLinear, 0.0)
        registered_channels.append(sitk.GetArrayFromImage(registered))
    return np.stack(registered_channels, axis=-1).astype(np.uint8)

# === warp RGB 图像用 displacement field ===
def warp_rgb_image(image_np, reference_sitk, transform):
    warped_channels = []
    for i in range(3):
        channel = sitk.GetImageFromArray(image_np[..., i])
        channel.CopyInformation(reference_sitk)
        warped = sitk.Resample(channel, reference_sitk, transform, sitk.sitkLinear, 0.0)
        warped_channels.append(sitk.GetArrayFromImage(warped))
    return np.stack(warped_channels, axis=-1).astype(np.uint8)

# === 配准函数 ===
def register_images_numpy(moving_rgb_np, fixed_rgb_np):
    moving_gray = np.dot(moving_rgb_np[...,:3], [0.2989, 0.5870, 0.1140]) if len(moving_rgb_np.shape) == 3 else moving_rgb_np
    fixed_gray = np.dot(fixed_rgb_np[...,:3], [0.2989, 0.5870, 0.1140]) if len(fixed_rgb_np.shape) == 3 else fixed_rgb_np
    
    moving_gray, fixed_gray = moving_gray / 255.0, fixed_gray / 255.0
    moving_gray, fixed_gray = np.clip(moving_gray, 0, 1), np.clip(fixed_gray, 0, 1)

    moving = sitk.GetImageFromArray((moving_gray * 255).astype(np.uint8))
    fixed = sitk.GetImageFromArray((fixed_gray * 255).astype(np.uint8))

    fixed = sitk.SmoothingRecursiveGaussian(fixed, sigma=1.5)
    moving = sitk.SmoothingRecursiveGaussian(moving, sigma=1.5)

    initial_transform = sitk.AffineTransform(2)
    matrix = [1.05, 0.0, 0.0, 1.00]  # 轻微缩放
    initial_transform.SetMatrix(matrix)

    def get_center(image):
        size = image.GetSize()
        return np.array([size[0] / 2.0, size[1] / 2.0])

    center_moving = get_center(moving)
    center_fixed = get_center(fixed)
    translation = center_fixed - center_moving
    initial_transform.SetTranslation([translation[1], translation[0]])

    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(1.0)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.5, minStep=1e-6, numberOfIterations=500, gradientMagnitudeTolerance=1e-6)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = registration_method.Execute(fixed, moving)

    registered_rigid = resample_rgb_image(moving_rgb_np, fixed, final_transform)

    moving_gray_registered = sitk.Resample(moving, fixed, final_transform, sitk.sitkLinear, 0.0)
    demons = sitk.DemonsRegistrationFilter()
    demons.SetNumberOfIterations(80)
    demons.SetStandardDeviations(1.0)
    displacement_field = demons.Execute(fixed, moving_gray_registered)

    displacement_tx = sitk.DisplacementFieldTransform(displacement_field)
    displacement_tx.SetSmoothingGaussianOnUpdate(0.0, 1.0)

    registered_final = warp_rgb_image(registered_rigid, fixed, displacement_tx)

    return {
        'RegisteredImage': registered_final,
        'RigidResult': registered_rigid,
        'Transformation': final_transform,
        'DisplacementField': displacement_field
    }

def resize_longest_edge(img, max_edge=1024):
    w, h = img.size
    scale = max_edge / max(w, h)
    new_size = (int(w * scale), int(h * scale))
    return img.resize(new_size, Image.LANCZOS)

def resize_to_match(img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    target_h = max(h1, h2)
    target_w = max(w1, w2)

    thread_safe_print(f"匹配尺寸为({str(target_w)}, {str(target_h)})")

    img1_resized = resize(img1, (target_h, target_w), preserve_range=True).astype(img1.dtype)
    img2_resized = resize(img2, (target_h, target_w), preserve_range=True).astype(img2.dtype)

    return img1_resized, img2_resized, target_w, target_h

def map_patch_dapi_to_he(
    dapi_patch_x, dapi_patch_y, patch_width, patch_height,
    he_slide,
    transform,
    dapi_thumb_size, dapi_orig_size,
    he_thumb_size, he_orig_size
):
    """
    修复线程安全问题的映射函数
    """
    try:
        dapi_downsample_x = dapi_orig_size[0] / dapi_thumb_size[0]
        dapi_downsample_y = dapi_orig_size[1] / dapi_thumb_size[1]
        he_downsample_x = he_orig_size[0] / he_thumb_size[0]
        he_downsample_y = he_orig_size[1] / he_thumb_size[1]

        x_thumb = dapi_patch_x / dapi_downsample_x
        y_thumb = dapi_patch_y / dapi_downsample_y
        w_thumb = patch_width / dapi_downsample_x
        h_thumb = patch_height / dapi_downsample_y

        corners = np.array([
            [x_thumb, y_thumb],
            [x_thumb + w_thumb, y_thumb],
            [x_thumb, y_thumb + h_thumb],
            [x_thumb + w_thumb, y_thumb + h_thumb]
        ])

        transformed_corners = []
        for pt in corners:
            # 确保传入的点是浮点数
            transformed = transform.TransformPoint((float(pt[0]), float(pt[1])))
            transformed_corners.append(transformed)
        transformed_corners = np.array(transformed_corners)

        he_corners_level0 = np.empty_like(transformed_corners)
        he_corners_level0[:, 0] = transformed_corners[:, 0] * he_downsample_x
        he_corners_level0[:, 1] = transformed_corners[:, 1] * he_downsample_y

        # 确保边界值在图像范围内
        min_x = int(np.floor(np.min(he_corners_level0[:, 0])))
        min_y = int(np.floor(np.min(he_corners_level0[:, 1])))
        max_x = int(np.ceil(np.max(he_corners_level0[:, 0])))
        max_y = int(np.ceil(np.max(he_corners_level0[:, 1])))

        # 边界检查
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(he_slide.shape[1], max_x)  # 图像宽度
        max_y = min(he_slide.shape[0], max_y)  # 图像高度

        # 如果边界无效，返回一个空的图像
        if min_x >= max_x or min_y >= max_y:
            thread_safe_print(f"⚠️ 无效边界: min_x={min_x}, max_x={max_x}, min_y={min_y}, max_y={max_y}")
            # 返回一个512x512的黑色图像
            empty_img = Image.new('RGB', (512, 512), (0, 0, 0))
            return empty_img, (0, 0, 512, 512)

        # 裁剪图像
        he_patch_img = Image.fromarray(he_slide).crop((min_x, min_y, max_x, max_y))
        
        # 调整大小
        he_patch_img = he_patch_img.resize((512, 512), resample=Image.BILINEAR)

        return he_patch_img, (min_x, min_y, 512, 512)
    
    except Exception as e:
        thread_safe_print(f"❌ map_patch_dapi_to_he 函数执行失败: {str(e)}")
        traceback.print_exc()
        # 返回一个默认的512x512图像
        empty_img = Image.new('RGB', (512, 512), (0, 0, 0))
        return empty_img, (0, 0, 512, 512)

# === 计算图像相似度 ===
def calculate_similarity(image1, image2):
    try:
        # 确保输入是2D数组
        if len(image1.shape) > 2:
            image1 = np.dot(image1[...,:3], [0.2989, 0.5870, 0.1140])
        if len(image2.shape) > 2:
            image2 = np.dot(image2[...,:3], [0.2989, 0.5870, 0.1140])
        
        # 确保图像数据范围在[0,1]之间
        image1 = np.clip(image1, 0, 1)
        image2 = np.clip(image2, 0, 1)
        
        # 计算SSIM，指定data_range
        return ssim(image1, image2, data_range=1.0)
    except Exception as e:
        thread_safe_print(f"❌ SSIM计算失败: {str(e)}")
        return 0.0  # 返回默认相似度值

# === 批量处理并使用多线程 ===
def process_patch_standalone(coord, dapi_pil, he_slide, result, resized_width, resized_height, save_dir, patch_size=512, similarity_threshold=0.8):
    start_time = time.time()
    dapi_patch_x, dapi_patch_y = coord["x"], coord["y"]

    try:
        step_start = time.time()
        dapi_patch = dapi_pil.crop((dapi_patch_x, dapi_patch_y, dapi_patch_x + patch_size, dapi_patch_y + patch_size))
        dapi_np = np.array(dapi_patch)
        thread_safe_print(f"⏱️ 坐标 ({dapi_patch_x}, {dapi_patch_y}) - 图像裁剪耗时: {time.time() - step_start:.4f}s")

        step_start = time.time()
        if np.mean(dapi_np) < 5:  # 如果是全黑，跳过
            thread_safe_print(f"⏱️ 坐标 ({dapi_patch_x}, {dapi_patch_y}) - 图像过滤耗时: {time.time() - step_start:.4f}s")
            return None
        thread_safe_print(f"⏱️ 坐标 ({dapi_patch_x}, {dapi_patch_y}) - 图像过滤耗时: {time.time() - step_start:.4f}s")

        step_start = time.time()
        he_patch_img, _ = map_patch_dapi_to_he(
            dapi_patch_x=dapi_patch_x,
            dapi_patch_y=dapi_patch_y,
            patch_width=patch_size,
            patch_height=patch_size,
            he_slide=he_slide,
            transform=result['Transformation'],
            dapi_thumb_size=(resized_width, resized_height),
            dapi_orig_size=(dapi_pil.width, dapi_pil.height),  # 使用原始尺寸而不是裁剪后的
            he_thumb_size=(resized_width, resized_height),
            he_orig_size=he_slide.shape[:2]
        )
        thread_safe_print(f"⏱️ 坐标 ({dapi_patch_x}, {dapi_patch_y}) - 映射DAPI到HE耗时: {time.time() - step_start:.4f}s")

        step_start = time.time()
        # 计算 DAPI 和 HE 图像的相似性
        dapi_gray = np.dot(dapi_np[...,:3], [0.2989, 0.5870, 0.1140])
        he_gray = np.dot(np.array(he_patch_img)[...,:3], [0.2989, 0.5870, 0.1140])
        thread_safe_print(f"⏱️ 坐标 ({dapi_patch_x}, {dapi_patch_y}) - 灰度转换耗时: {time.time() - step_start:.4f}s")

        step_start = time.time()
        similarity = calculate_similarity(dapi_gray, he_gray)
        thread_safe_print(f"⏱️ 坐标 ({dapi_patch_x}, {dapi_patch_y}) - 相似度计算耗时: {time.time() - step_start:.4f}s")

        if similarity < similarity_threshold:
            thread_safe_print(f"⚠️ 坐标 ({dapi_patch_x}, {dapi_patch_y}) 相似度较低({similarity:.4f})，跳过")
            return None

        step_start = time.time()
        save_path = os.path.join(save_dir, f"he_x{dapi_patch_x}_y{dapi_patch_y}.png")
        he_patch_img.save(save_path)
        thread_safe_print(f"⏱️ 坐标 ({dapi_patch_x}, {dapi_patch_y}) - 保存文件耗时: {time.time() - step_start:.4f}s")
        
        total_time = time.time() - start_time
        thread_safe_print(f"✅ 坐标 ({dapi_patch_x}, {dapi_patch_y}) 处理完成，总耗时: {total_time:.4f}s, 相似度: {similarity:.4f}")
        return save_path
    
    except Exception as e:
        error_msg = f"❌ 坐标 ({dapi_patch_x}, {dapi_patch_y}) 处理失败: {str(e)}\n{traceback.format_exc()}"
        thread_safe_print(error_msg)
        return None

# === 主函数 ===
def main(args):
    # 读取图像
    thread_safe_print("开始加载图像...")
    dapi_img = tifffile.imread(args.dapi_img_path)
    he_slide = tifffile.imread(args.he_slide_path)

    dapi_pil = Image.fromarray(dapi_img)
    he_img = Image.fromarray(he_slide)

    thread_safe_print("开始生成缩略图...")
    dapi_thumbnail = resize_longest_edge(dapi_pil)
    he_thumbnail = resize_longest_edge(he_img)

    fixed_img, moving_img, resized_width, resized_height = resize_to_match(np.array(dapi_thumbnail), np.array(he_thumbnail))

    thread_safe_print("开始图像配准...")
    result = register_images_numpy(moving_img, fixed_img)

    os.makedirs(args.save_dir, exist_ok=True)

    with open(args.json_path, "r") as f:
        coords_list = json.load(f)

    thread_safe_print(f"✅ 读取到 {len(coords_list)} 个坐标点")

    # 使用多线程处理所有 patches，限制最大线程数
    max_workers = min(4, os.cpu_count())  # 减少线程数以避免内存问题
    thread_safe_print(f"🚀 使用 {max_workers} 个线程进行处理")
    
    successful_count = 0
    failed_count = 0
    
    # 将所有参数传递给线程函数
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_coord = {
            executor.submit(process_patch_standalone, coord, dapi_pil, he_slide, result, resized_width, resized_height, args.save_dir): coord 
            for coord in coords_list
        }
        
        # 使用as_completed来处理完成的任务
        for future in tqdm(as_completed(future_to_coord), total=len(coords_list), desc="生成 HE Patch"):
            try:
                result_path = future.result(timeout=120)  # 设置超时时间为120秒
                if result_path:
                    successful_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                coord = future_to_coord[future]
                thread_safe_print(f"❌ 坐标 ({coord['x']}, {coord['y']}) 处理超时或出错: {str(e)}")
                failed_count += 1

    thread_safe_print(f"✅ 所有 patch 已保存到 {args.save_dir}")
    thread_safe_print(f"📊 处理完成: 成功 {successful_count} 个, 失败 {failed_count} 个")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 DAPI 图像到 HE 图像的配准")
    parser.add_argument("--dapi_img_path", required=True, help="DAPI 图像路径")
    parser.add_argument("--he_slide_path", required=True, help="HE 图像路径")
    parser.add_argument("--json_path", required=True, help="坐标 JSON 文件路径")
    parser.add_argument("--save_dir", required=True, help="保存目录路径")

    args = parser.parse_args()

    main(args)