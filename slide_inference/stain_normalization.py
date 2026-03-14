import os
import cv2
import numpy as np
from PIL import Image, PngImagePlugin
PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024
from wsi_normalizer import ReinhardNormalizer  # 或 ReinhardNormalizer, VahadaneNormalizer, TorchVahadaneNormalizer

def batch_color_normalize_with_white_mask(source, input_folder, output_folder, white_threshold=230):
    """
    批量对文件夹中的PNG图像进行染色标准化处理，并保留原图中接近白色的区域
    
    Args:
        source_path (str): 参考图像路径，用于提取染色特征
        input_folder (str): 输入文件夹路径，包含待处理的PNG图像
        output_folder (str): 输出文件夹路径，保存处理后的图像
        white_threshold (int): 判断接近白色的阈值，默认240 (0-255)
    """
    source_path = './stain_reference/1_2196.png'
    if source == "Adrenal_gland":
        source_path = './stain_reference/3_1449.png'
    elif source == "Bile_duct":
        source_path = './stain_reference/1_1464.png'
    elif source == "Bladder":
        source_path = './stain_reference/1_1670.png'
    elif source == "Breast":
        source_path = '/stain_reference/3_582.png'
    elif source == "Cervix":
        source_path = './stain_reference/1_1784.png'
    elif source == "Colon":
        source_path = './stain_reference/3_2682.png'
    elif source == "Esophagus":
        source_path = './stain_reference/1_1236.png'
    elif source == "HeadNeck":
        source_path = './stain_reference/3_2055.png'
    elif source == "Kidney":
        source_path = './stain_reference/3_2144.png'
    elif source == "Liver":
        source_path = './stain_reference/1_2196.png'
    elif source == "Lung":
        source_path = './stain_reference/2_913.png'
    elif source == "Ovarian":
        source_path = './stain_reference/3_2305.png'
    elif source == "Pancreatic":
        source_path = '/stain_reference/1_1286.png'
    elif source == "Prostate":
        source_path = './stain_reference/3_2360.png'
    elif source == "Skin":
        source_path = './stain_reference/3_2408.png'
    elif source == "Stomach":
        source_path = './stain_reference/1_2500.png'
    elif source == "Testis":
        source_path = './stain_reference/2_1251.png'
    elif source == "Thyroid":
        source_path = './stain_reference/1_1099.png'
    elif source == "Uterus":
        source_path = './stain_reference/3_2581.png'

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 加载参考图像并初始化标准化器
    print("加载参考图像并初始化标准化器...")
    source_image = Image.open(source_path).convert('RGB')
    source_np = np.array(source_image).astype(np.float32) / 255.0
    reinhard_normalizer = ReinhardNormalizer()
    reinhard_normalizer.fit(source_np)
    print("标准化器初始化完成")
    
    # 获取输入文件夹中所有PNG文件
    png_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith('.png'):
                png_files.append(os.path.join(root, file))
    
    print(f"找到 {len(png_files)} 个PNG图像文件")
    
    # 处理每个PNG文件
    processed_count = 0
    for i, target_path in enumerate(png_files, 1):
        try:
            print(f"处理 {i}/{len(png_files)}: {os.path.basename(target_path)}")
            
            # 加载目标图像
            original_image = Image.open(target_path).convert('RGB')
            original_np = np.array(original_image).astype(np.float32)
            
            # 将原始图像转换为标准化格式用于染色
            target_np = original_np / 255.0
            
            # 执行染色标准化
            norm_img = reinhard_normalizer.transform(target_np)
            # 将结果归一化到(0, 255)范围
            norm_img = np.clip(norm_img, 0, 255).astype(np.uint8)
            
            # 找出原图像中接近白色的区域
            # 白色区域定义为R, G, B三个通道都大于阈值的像素
            white_mask = np.all(original_np > white_threshold, axis=2)
            
            # 将原图中接近白色的区域复制到标准化后的图像上
            result_img = norm_img.copy()
            result_img[white_mask] = original_np[white_mask].astype(np.uint8)
            
            # 生成输出路径
            relative_path = os.path.relpath(target_path, input_folder)
            output_path = os.path.join(output_folder, relative_path)
            
            # 确保输出路径的目录存在
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存处理后的图像
            result_image = Image.fromarray(result_img)
            result_image.save(output_path)
            
            processed_count += 1
            print(f"  已保存: {output_path}")
            
        except Exception as e:
            print(f"  错误: 处理 {target_path} 时出错 - {str(e)}")
    
    print(f"\n处理完成！成功处理 {processed_count}/{len(png_files)} 个文件")
    print(f"输出文件夹: {output_folder}")


# def main():
#     # 设置参数
#     # 参考图像路径 - 用于提取染色特征的图像
#     source_path = '/nfs5/lwh/cell-seg-dataset-processed/pannuke/images/3_2227.png'
    
#     # 输入文件夹路径 - 包含待处理的PNG图像
#     input_folder = "/nfs5/lwh/guangdong-liver/1622496-8/patch"
    
#     # 输出文件夹路径 - 保存处理后的图像
#     output_folder = "/nfs5/lwh/guangdong-liver/1622496-8/patch-stain"
    
#     # 检查输入路径是否存在
#     if not os.path.exists(source_path):
#         print(f"错误: 参考图像 '{source_path}' 不存在")
#         return
    
#     if not os.path.isdir(input_folder):
#         print(f"错误: 输入文件夹 '{input_folder}' 不存在")
#         return
    
#     # 执行批量处理
#     batch_color_normalize_with_white_mask(source_path, input_folder, output_folder)