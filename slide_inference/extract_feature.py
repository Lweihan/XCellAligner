import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict
from torchvision import transforms
import glob
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from utils import load_cellpose_model, preprocess_image, build_cell_features
from models import TransformerEncoder

# Try to import ctranpath, install if not available
try:
    from module.TransPath.ctran import ctranspath
except ImportError:
    raise ImportError("CTransPath module not found. Please install it from https://github.com/Xiyue-Wang/TransPath")

# 全局变量用于存储模型和设备
_global_device = None
_global_ctranspath_model = None
_global_cellpose_model = None
_global_model = None
_model_lock = threading.Lock()

def initialize_models_once(model_path, device):
    """初始化模型，确保只初始化一次"""
    global _global_device, _global_ctranspath_model, _global_cellpose_model, _global_model
    
    with _model_lock:
        if _global_device is None:
            _global_device = device
            
        if _global_ctranspath_model is None:
            print("加载 CTransPath 模型...")
            _global_ctranspath_model = ctranspath()
            _global_ctranspath_model.to(_global_device)
            _global_ctranspath_model.eval()
            
        if _global_cellpose_model is None:
            print("加载 Cellpose 模型...")
            _global_cellpose_model = load_cellpose_model(model_type='cyto', gpu=torch.cuda.is_available())
            
        if _global_model is None:
            # 初始化我们的transformer模型 (输出维度仍然是8)
            input_dim = 1000  # CTransPath特征维度
            hidden_dim = 512
            n_heads = 4
            num_layers = 6
            output_dim = 8
            max_cells = 255
            
            _global_model = TransformerEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                num_layers=num_layers,
                output_dim=output_dim,
                max_cells=max_cells
            ).to(_global_device)
            
            # 加载训练好的模型权重
            print("加载训练好的模型权重...")
            _global_model.load_state_dict(torch.load(model_path, map_location=_global_device))
            _global_model.eval()
    
    return _global_device, _global_ctranspath_model, _global_cellpose_model, _global_model

def save_features_to_disk(features, image_name, output_dir):
    """
    将特征保存为 NumPy 文件
    
    Args:
        features: 特征数组
        image_name: 图像名称
        output_dir: 输出目录
    """
    features_file = os.path.join(output_dir, f"features_{image_name}.npy")
    np.save(features_file, np.array(features))
    print(f"已保存特征到: {features_file}")

def extract_cell_features_for_inference(image_path, cellpose_model, ctranspath_model, device):
    """
    提取细胞特征用于推理（不需要标签图）
    
    Args:
        image_path: 图像路径
        cellpose_model: Cellpose模型
        ctranspath_model: CTransPath模型
        device: 设备
    
    Returns:
        tuple: (cell_features, masks, original_image)
               cell_features: 特征列表
               masks: 分割掩码
               original_image: 原始图像
    """
    img = preprocess_image(image_path)
    
    # 使用Cellpose进行细胞分割
    masks, flows, styles = cellpose_model.eval(img, diameter=None, channels=[0, 0])

    # 使用CTransPath模型进行编码
    ctranspath_model.eval()  # 设置为评估模式

    # CTransPath模型的预处理函数
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 提取每个细胞的特征
    cell_features = []
    unique_masks = np.unique(masks)
    
    for label in unique_masks:
        if label == 0:
            continue  # 跳过背景
            
        # 获取细胞区域
        cell_mask = masks == label
        cell_img = img * cell_mask[..., None]  # 通过掩码提取细胞区域
        cell_img_pil = Image.fromarray(cell_img)
        
        # 预处理并输入CTransPath模型
        cell_img_tensor = preprocess(cell_img_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # 使用 CTransPath 提取特征
            features = ctranspath_model(cell_img_tensor)  # 得到细胞特征向量
        
        cell_features.append(features.squeeze().cpu().numpy())  # 获取特征并存储

    return cell_features, masks, img

def get_dominant_class_in_mask(segment_mask, cell_mask):
    """
    获取细胞区域内的主要类别
    
    Args:
        segment_mask: 语义分割结果 (H, W)
        cell_mask: 单个细胞的掩码 (H, W)
        
    Returns:
        dominant_class: 主要类别编号
    """
    # 获取细胞区域内的像素值
    masked_values = segment_mask[cell_mask]
    
    # 如果没有像素值，则返回0（背景）
    if len(masked_values) == 0:
        return 0
    
    # 统计各个类别的出现次数
    unique_values, counts = np.unique(masked_values, return_counts=True)
    
    # 返回出现次数最多的类别
    dominant_class = unique_values[np.argmax(counts)]
    return dominant_class

def get_cell_centroids(masks):
    """
    计算每个细胞的质心坐标
    
    Args:
        masks: 细胞分割掩码
        
    Returns:
        centroids: 每个细胞的质心坐标列表 [(x, y), ...]
    """
    centroids = []
    unique_masks = np.unique(masks)
    
    for label in unique_masks:
        if label == 0:
            continue  # 跳过背景
            
        # 获取细胞区域
        cell_mask = masks == label
        
        # 计算质心
        coords = np.where(cell_mask)
        centroid_y = int(np.mean(coords[0]))  # 行号是y坐标
        centroid_x = int(np.mean(coords[1]))  # 列号是x坐标
        centroids.append((centroid_x, centroid_y))
        
    return centroids

def visualize_clusters(masks, cluster_labels, save_path, k):
    """
    可视化聚类结果 (不依赖原始图像)
    
    Args:
        masks: 细胞分割掩码
        cluster_labels: 聚类标签
        save_path: 结果保存路径
        k: 聚类数
    """
    # 创建黑色背景图像
    mask_shape = masks.shape
    result_img = np.zeros((mask_shape[0], mask_shape[1], 3), dtype=np.uint8)
    
    # 创建颜色映射
    colors = plt.cm.get_cmap('tab10', k)
    
    # 为每个聚类分配颜色
    unique_masks = np.unique(masks)
    cell_idx = 0
    
    for label in unique_masks:
        if label == 0:
            continue  # 跳过背景
            
        # 获取细胞区域
        cell_mask = masks == label
        
        # 为该细胞分配颜色
        cluster_id = cluster_labels[cell_idx]
        color = (np.array(colors(cluster_id)[:3]) * 255).astype(np.uint8)
        
        # 应用颜色到结果图像
        for c in range(3):  # RGB三个通道
            result_img[cell_mask, c] = color[c]
            
        cell_idx += 1
    
    # 保存结果
    result_pil = Image.fromarray(result_img)
    result_pil.save(save_path)
    print(f"结果已保存到: {save_path}")

def pad_features_to_max_cells(features, max_cells=255):
    """
    将特征数组填充到最大细胞数
    
    Args:
        features: 特征数组 [num_cells, feature_dim]
        max_cells: 最大细胞数
        
    Returns:
        padded_features: 填充后的特征 [max_cells, feature_dim]
        mask: 掩码 [max_cells] 1表示有效位置，0表示填充位置
    """
    num_cells, feature_dim = features.shape
    padded_features = np.zeros((max_cells, feature_dim))
    padded_features[:num_cells] = features
    mask = np.zeros(max_cells)
    mask[:num_cells] = 1
    return padded_features, mask

def process_single_image(image_path, model_path, device, k=5, segment_mask=None):
    """
    处理单张图像（线程安全版本）
    
    Args:
        image_path: 输入图像路径
        model_path: 训练好的模型路径
        device: 设备
        k: 聚类数
        segment_mask: 语义分割结果（可选）
    """
    # 初始化模型（仅首次调用时实际初始化）
    device, ctranspath_model, cellpose_model, model = initialize_models_once(model_path, device)
    
    # 提取细胞特征
    print(f"提取细胞特征: {os.path.basename(image_path)}")
    cell_features, masks, original_image = extract_cell_features_for_inference(
        image_path, cellpose_model, ctranspath_model, device)
    
    if len(cell_features) == 0:
        print(f"未检测到细胞: {os.path.basename(image_path)}")
        return False
    
    print(f"共检测到 {len(cell_features)} 个细胞")
    
    # 准备特征数据并填充到固定长度 (使用原始1000维特征)
    features_array = np.array(cell_features)
    padded_features, mask = pad_features_to_max_cells(features_array)
    
    # 添加batch维度
    padded_features = torch.tensor(padded_features, dtype=torch.float32).unsqueeze(0).to(device)  # [1, max_cells, 1000]
    mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(device)  # [1, max_cells]
    
    # 使用模型进行推理
    print("使用模型进行推理...")
    with torch.no_grad():
        cls_output, model_output, logits = model(padded_features, mask_tensor)
    
    # 提取有效的细胞特征（去除填充部分）
    valid_features = model_output[0, :len(cell_features)].cpu().numpy()  # [num_cells, output_dim]
    
    # 如果提供了语义分割结果，则为每个细胞特征添加类别信息
    if segment_mask is not None:
        # 为每个细胞计算主要类别
        cell_classes = []
        unique_masks = np.unique(masks)
        valid_cell_count = 0
        
        for label in unique_masks:
            if label == 0:
                continue  # 跳过背景
                
            cell_mask = masks == label
            dominant_class = get_dominant_class_in_mask(segment_mask, cell_mask)
            cell_classes.append(dominant_class)
            valid_cell_count += 1
        
        # 将类别信息添加到模型输出的特征中
        cell_classes_array = np.array(cell_classes).reshape(-1, 1)
        
        # 合并模型输出特征和类别特征
        enhanced_features = np.concatenate([valid_features, cell_classes_array], axis=1)
        print(f"已为 {valid_cell_count} 个细胞添加语义类别特征")
    else:
        # 如果没有提供语义分割结果，只使用模型输出的特征
        enhanced_features = valid_features
    
    return masks, enhanced_features, os.path.basename(image_path), image_path

def batch_inference(input_folder, model_path, output_folder, k=5, max_workers=2):
    """
    批量推理主函数
    
    Args:
        input_folder: 输入图像文件夹路径
        model_path: 训练好的模型路径
        output_folder: 结果保存文件夹路径
        k: 聚类数
        segment_path: 语义分割结果文件夹路径（可选）
        max_workers: 最大线程数
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    temp_features_dir = os.path.join(output_folder, "temp_features")
    os.makedirs(temp_features_dir, exist_ok=True)
    
    # 获取所有图像文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    image_files = []
    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, extension)))
        image_files.extend(glob.glob(os.path.join(input_folder, extension.upper())))
    
    if not image_files:
        print(f"在 {input_folder} 中没有找到图像文件")
        return
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 处理每个图像并保存特征
    processed_images = []
    image_paths = []
    image_masks = []  # 保存每张图像的mask用于计算质心
    
    # 使用线程池处理图像
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务
        future_to_image = {}
        for i, image_path in enumerate(image_files):
            print(f"[{i+1}/{len(image_files)}] 准备处理图像: {os.path.basename(image_path)}")
            
            # 提交任务到线程池
            future = executor.submit(process_single_image, image_path, model_path, device, k, segment_mask)
            future_to_image[future] = (image_path, segment_mask)
        
        # 获取结果
        completed_count = 0
        for future in as_completed(future_to_image):
            image_path, segment_mask = future_to_image[future]
            try:
                result = future.result(timeout=120)  # 设置超时时间为120秒
                if result:
                    masks, enhanced_features, image_name, orig_image_path = result
                    
                    # 保存特征到磁盘
                    save_features_to_disk(enhanced_features, image_name, temp_features_dir)
                    processed_images.append(image_name)
                    image_paths.append(orig_image_path)
                    image_masks.append(masks)  # 保存mask用于后续计算质心
                    
                    completed_count += 1
                    print(f"[{completed_count}/{len(image_files)}] 已保存 {image_name} 的特征")
                else:
                    completed_count += 1
                    print(f"[{completed_count}/{len(image_files)}] 跳过图像 {os.path.basename(image_path)}")
            except Exception as e:
                completed_count += 1
                print(f"[{completed_count}/{len(image_files)}] 处理图像 {os.path.basename(image_path)} 时出错: {str(e)}")
                continue
    
    print("所有图像特征已保存到磁盘")
    
    # 从磁盘加载所有特征进行聚类
    print("正在从磁盘加载所有特征...")
    all_features = []
    feature_files = [os.path.join(temp_features_dir, f) for f in os.listdir(temp_features_dir) if f.startswith("features_")]
    all_features = []

    for feature_file in feature_files:
        batch_features = np.load(feature_file)

        # 只处理二维特征
        if batch_features.ndim != 2:
            print(f"跳过非二维特征文件: {feature_file}, shape={batch_features.shape}")
            continue

        feature_dim = batch_features.shape[1]

        # 8 维 -> 补 0 成 9 维
        if feature_dim == 8:
            pad = np.zeros((batch_features.shape[0], 1), dtype=batch_features.dtype)
            batch_features = np.concatenate([batch_features, pad], axis=1)

        # 9 维 -> 直接使用
        elif feature_dim == 9:
            pass

        # 其他维度 -> 跳过
        else:
            print(f"跳过异常维度特征文件: {feature_file}, shape={batch_features.shape}")
            continue

        all_features.append(batch_features)
    
    if len(all_features) == 0:
        print("没有找到任何特征文件")
        return
    
    all_features = np.concatenate(all_features, axis=0)
    print(f"加载所有特征完成，特征总数：{len(all_features)}")
    
    # K-means 聚类
    print(f"使用K-means进行 {k} 类聚类...")
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(all_features)
    
    # 为每个图像生成聚类可视化结果和细胞信息
    cluster_idx = 0
    all_cell_info = []  # 存储所有细胞的信息
    for i, image_name in enumerate(processed_images):
        try:
            # 加载之前保存的特征
            features_file = os.path.join(temp_features_dir, f"features_{image_name}.npy")
            
            if not os.path.exists(features_file):
                print(f"缺少 {image_name} 的特征文件")
                continue
            
            enhanced_features = np.load(features_file)
            
            base_name = os.path.splitext(image_name)[0]
            
            # 获取当前图像的聚类标签
            current_clusters = clusters[cluster_idx:cluster_idx+len(enhanced_features)]
            cluster_idx += len(enhanced_features)
            
            # 计算每个细胞的质心坐标
            centroids = get_cell_centroids(image_masks[i])
            
            # 保存每个细胞的信息：图像名、质心坐标(x,y)、聚类类别
            for j, (centroid, cluster_label) in enumerate(zip(centroids, current_clusters)):
                cell_info = {
                    "image_name": image_name,
                    "cell_id": j,
                    "centroid_x": int(centroid[0]),
                    "centroid_y": int(centroid[1]),
                    "cluster": int(cluster_label)
                }
                all_cell_info.append(cell_info)
            
            # 生成输出文件路径
            save_path = os.path.join(output_folder, f"{base_name}_cluster.png")
            
            # 可视化结果（不使用原始图像）
            print(f"可视化聚类结果: {image_name}...")
            visualize_clusters(image_masks[i], current_clusters, save_path, k)
            
        except Exception as e:
            print(f"处理图像 {image_name} 的聚类结果时出错: {str(e)}")
            continue
    
    # 保存所有细胞的信息到一个总的JSON文件
    all_cell_info_path = os.path.join(output_folder, "all_cells_info.json")
    with open(all_cell_info_path, "w", encoding="utf-8") as f:
        json.dump(all_cell_info, f, ensure_ascii=False, indent=2)
    print(f"所有细胞信息已保存到: {all_cell_info_path}")
    
    print("批量推理完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量推理细胞图像")
    parser.add_argument("--input_folder", type=str, required=True, help="输入图像文件夹路径")
    parser.add_argument("--model_path", type=str, required=True, help="训练好的模型路径")
    parser.add_argument("--output_folder", type=str, required=True, help="结果保存文件夹路径")
    parser.add_argument("--k", type=int, default=5, help="聚类数")
    parser.add_argument("--segment_path", type=str, default=None, help="语义分割结果文件夹路径（可选）")
    parser.add_argument("--max_workers", type=int, default=2, help="最大线程数")
    
    args = parser.parse_args()
    
    batch_inference(args.input_folder, args.model_path, args.output_folder, args.k, args.segment_path, args.max_workers)