import os
import sys
import openslide
from PIL import Image
import numpy as np
import tifffile as tiff
import argparse
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(parent_dir, 'module'))
from KFBreader.kfbreader import KFBSlide

# 允许处理更大的图像
Image.MAX_IMAGE_PIXELS = None

def rotate_svs(input_path, output_path, level, angle, back_color='black'):
    _, ext = os.path.splitext(input_path)
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
        slide = tiff.imread(input_path)
        level0 = Image.fromarray(slide)
    
    if back_color == 'black':
        # 旋转图像
        rotated = level0.rotate(angle, expand=True)
        rotated_np = np.array(rotated)

        # 保存为 BigTIFF 格式
        tiff.imwrite(output_path, rotated_np, bigtiff=True)
    elif back_color == 'white':
        # 创建一个白色背景的图像
        white_bg_img = Image.new('RGB', level0.size, (255, 255, 255))
        white_bg_img.paste(level0, mask=level0.split()[3] if len(level0.split()) > 3 else None)

        # 旋转图像
        rotated = white_bg_img.rotate(angle, expand=True, fillcolor=(255, 255, 255))

        # 将旋转后的图像转换为numpy数组
        rotated_np = np.array(rotated)

        # 保存为 BigTIFF 格式
        tiff.imwrite(output_path, rotated_np, bigtiff=True)
    if ext == '.kfb' or ext == '.mrxs' or ext == '.svs':
        slide.close()

    print(f"✅ 保存完成：{output_path}")

def parse_args():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="旋转SVS图像并保存为TIFF")
    parser.add_argument("--input_path", type=str, help="输入SVS图像路径")
    parser.add_argument("--output_path", type=str, help="输出TIFF图像路径")
    parser.add_argument("--level", type=int, default=0, help="要读取的级别，默认为0级（原始级别）")
    parser.add_argument("--angle", type=int, help="旋转角度，支持 90、180、270 度")
    parser.add_argument("--back_color", type=str, help="背景颜色")
    
    return parser.parse_args()

def main():
    args = parse_args()
    rotate_svs(args.input_path, args.output_path, args.level, args.angle, args.back_color)

if __name__ == "__main__":
    main()
