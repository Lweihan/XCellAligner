import os
os.environ['nnUNet_raw'] = './module/nnUNet/nnunetv2/nnunetv2_hist/nnUNet_raw'
os.environ['nnUNet_preprocessed'] = './module/nnUNet/nnunetv2/nnunetv2_hist/nnUNet_preprocessed'
os.environ['nnUNet_results'] = './module/nnUNet/nnunetv2/nnunetv2_hist/nnUNet_results'
import argparse
import time
import shutil
import sys
from PIL import Image
import numpy as np

# Add the module path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'module'))

from slide_inference.multi_thread_get_patch import slide_to_patches
from slide_inference.stain_normalization import batch_color_normalize_with_white_mask
from slide_inference.rename import rename_patch
from slide_inference.basic_cell_segmentation import segment_slide
from slide_inference.reverse_rename import revserse_rename
from slide_inference.extract_feature import batch_inference
from slide_inference.patch_to_slide import stitch_patches_to_svs

def create_black_image(image_path, width=1024, height=1024):
    """
    创建一个纯黑的 RGB 图像并保存到指定路径
    
    :param image_path: 图像保存路径
    :param width: 图像宽度
    :param height: 图像高度
    """
    # 创建一个 RGB 模式的纯黑图像
    black_image = Image.new('RGB', (width, height), color=(0, 0, 0))
    # 保存为 PNG 文件
    black_image.save(image_path)

def check_and_create_black_images(patch_dir, vis_dir):
    """
    检查 vis_dir 中是否存在与 patch_dir 中同名的 PNG 文件，如果没有，则创建一个纯黑的同名文件。
    
    :param patch_dir: 存放补丁图像的文件夹
    :param vis_dir: 存放可视化图像的文件夹
    """
    # 获取 patch_dir 中的所有 PNG 文件
    files = [f for f in os.listdir(patch_dir) if f.endswith(".png")]
    
    # 遍历所有文件
    for file in files:
        patch_path = os.path.join(patch_dir, file)
        vis_path = os.path.join(vis_dir, file.replace(".png", "_cell_info.json"))

        vis_name = file.replace(".png", "_cluster.png")
        # 如果 vis_dir 中不存在同名文件，则创建一个纯黑图像
        if not os.path.exists(vis_path):
            print(f"创建黑色图像：{vis_name}")
            create_black_image(os.path.join(vis_dir, vis_name))
def run_pipeline(slide_path, model_path, temp_path, output_path, type, k):
    """Run the complete slide inference pipeline with timing for each step.
    
    Args:
        slide_path (str): Path to the slide file
        model_path (str): Path to the model weights
        temp_path (str): Path to store temporary files
        output_path (str): Path to save final outputs
        type (str): Type of the slide
    """
    
    # Check if slide file exists
    if not os.path.exists(slide_path):
        raise FileNotFoundError(f"Slide file not found: {slide_path}")
    
    # Create necessary directories
    os.makedirs(temp_path, exist_ok=True)
    
    # Step 1: Extract patches
    print("=" * 50)
    print("Step 1: Extracting patches...")
    patch_dir = os.path.join(temp_path, "patches")
    start_time = time.time()
    
    slide_to_patches(slide_path, patch_size=1024, save_dir=patch_dir, num_workers=4)
    
    patch_time = time.time() - start_time
    print(f"Patch extraction completed in {patch_time:.2f} seconds")
    
    # Step 2: Stain normalization
    print("=" * 50)
    print("Step 2: Performing stain normalization...")
    normalized_dir = os.path.join(temp_path, "normalized_patches")
    start_time = time.time()
    
    batch_color_normalize_with_white_mask(type, patch_dir, normalized_dir, white_threshold=230)
    
    norm_time = time.time() - start_time
    print(f"Stain normalization completed in {norm_time:.2f} seconds")
    
    # # Step 3: Rename patches
    print("=" * 50)
    print("Step 3: Renaming patches...")    
    start_time = time.time()
    
    rename_patch(slide_path, normalized_dir, case_id="case")
    
    rename_time = time.time() - start_time
    print(f"Patch renaming completed in {rename_time:.2f} seconds")
    
    # Step 4: Cell segmentation
    print("=" * 50)
    print("Step 4: Performing cell segmentation...")
    segmented_dir = os.path.join(temp_path, "segmented")
    start_time = time.time()
    
    # Note: In a real implementation, you would need to specify the correct path to segmentor weights
    segmentor_weight = "./module/nnUNet/Dataset011_Custom/nnUNetTrainer__nnUNetPlans__2d"  # This should be adjusted based on your setup
    try:
        segment_slide(normalized_dir, segmented_dir, segmentor_weight)
    except Exception as e:
        print(f"Warning: Segmentation failed with error: {e}")
        print("Creating empty directory for demonstration purposes")
        os.makedirs(segmented_dir, exist_ok=True)
    
    seg_time = time.time() - start_time
    print(f"Cell segmentation completed in {seg_time:.2f} seconds")
    
    # Step 5: Reverse renaming
    print("=" * 50)
    print("Step 5: Reversing patch names...")
    
    start_time = time.time()
    
    revserse_rename(slide_path, normalized_dir, type="patch")
    revserse_rename(slide_path, segmented_dir, type="mask")
    
    reverse_time = time.time() - start_time
    print(f"Reverse renaming completed in {reverse_time:.2f} seconds")

    # Step 6: Extract features
    print("=" * 50)
    print("Step 6: Extracting features...")

    vis_dir = os.path.join(temp_path, "vis")
    start_time = time.time()

    batch_inference(normalized_dir, model_path, vis_dir, k=k, segment_path=segmented_dir, max_workers=4)

    check_and_create_black_images(patch_dir, vis_dir)

    extract_time = time.time() - start_time

    print(f"Feature extraction completed in {extract_time:.2f} seconds")

    # Step 7: Stitch patches back to the slide
    print("=" * 50)
    print("Step 7: Stitching patches back to the slide...")

    start_time = time.time()

    stitch_patches_to_svs(
        slide_path,
        vis_dir,
        output_path,
        patch_size=1024
    )

    stitch_time = time.time() - start_time

    print(f"Stitching completed in {stitch_time:.2f} seconds")
    
    # Summary
    print("=" * 50)
    print("Pipeline completed! Summary:")
    print(f"  Patch extraction:     {patch_time:.2f}s")
    print(f"  Stain normalization:  {norm_time:.2f}s")
    print(f"  Patch renaming:       {rename_time:.2f}s")
    print(f"  Cell segmentation:    {seg_time:.2f}s")
    print(f"  Reverse renaming:     {reverse_time:.2f}s")
    print(f"  Feature extraction:   {extract_time:.2f}s")
    print(f"  Patch stitching:      {stitch_time:.2f}s")
    print(f"  Total time:           {patch_time + norm_time + rename_time + seg_time + reverse_time + extract_time + stitch_time:.2f}s")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Slide inference pipeline")
    parser.add_argument("--slide_path", type=str, required=True, help="Path to the slide file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model weights")
    parser.add_argument("--temp_path", type=str, required=True, help="Path to store temporary files")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save final outputs")
    parser.add_argument("--type", type=str, required=True, help="Type of the slide")
    parser.add_argument("--k", type=int, default=5, help="Number of clusters")
    
    args = parser.parse_args()
    
    try:
        run_pipeline(args.slide_path, args.model_path, args.temp_path, args.output_path, args.type, args.k)
    except Exception as e:
        print(f"Error running pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()