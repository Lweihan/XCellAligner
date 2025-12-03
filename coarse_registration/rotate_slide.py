import openslide
from PIL import Image
import numpy as np
import tifffile as tiff
import argparse

# 允许处理更大的图像
Image.MAX_IMAGE_PIXELS = None

def rotate_svs(input_path, output_path, angle):
    # 打开 SVS 文件
    slide = openslide.OpenSlide(input_path)
    
    # 读取一级图像数据（0级是原始级别的图像）
    level0 = slide.read_region((0, 0), 0, slide.level_dimensions[0]).convert("RGB")
    
    # 旋转图像
    rotated = level0.rotate(angle, expand=True)
    rotated_np = np.array(rotated)

    # 保存为 BigTIFF 格式
    tiff.imwrite(output_path, rotated_np, bigtiff=True)
    slide.close()

    print(f"✅ 保存完成：{output_path}")

def parse_args():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="旋转SVS图像并保存为TIFF")
    parser.add_argument("--input_path", type=str, help="输入SVS图像路径")
    parser.add_argument("--output_path", type=str, help="输出TIFF图像路径")
    parser.add_argument("--angle", type=int, choices=[90, 180, 270], help="旋转角度，支持 90、180、270 度")
    
    return parser.parse_args()

def main():
    args = parse_args()
    rotate_svs(args.input_path, args.output_path, args.angle)

if __name__ == "__main__":
    main()
