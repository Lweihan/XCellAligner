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
from torchvision.ops import roi_align
from utils import load_cellpose_model, preprocess_image, build_cell_features
from models import TransformerEncoder

# Try to import ctranpath, install if not available
try:
    from module.TransPath.ctran import ctranspath
except ImportError:
    raise ImportError("CTransPath module not found. Please install it from https://github.com/Xiyue-Wang/TransPath")

def extract_features_using_roi_align(image_path, cellpose_model, ctranspath_model, device, max_cells=255):
    """
    使用Cellpose的分割结果作为region proposal，使用ROI Align提取每个细胞的特征
    Args:
        image_path: 输入图像路径
        cellpose_model: 细胞分割模型
        ctranspath_model: CTransPath模型
        device: 设备（CPU/GPU）
        max_cells: 最大细胞数
    Returns:
        tuple: (cell_features, masks, original_image)
               cell_features: 提取的细胞特征
               masks: 细胞掩码
               original_image: 原始图像
    """
    img = preprocess_image(image_path)
    
    # 使用Cellpose进行细胞分割，得到细胞掩码
    masks, _, _ = cellpose_model.eval(img, diameter=None, channels=[0, 0])
    
    # 使用CTransPath提取全局特征
    ctranspath_model.eval()
    # 将img从numpy.ndarray转为PIL图像
    image = Image.fromarray(img)
    preprocess = transforms.Compose([ 
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    cell_img_tensor = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # 提取整张图的特征
        global_features = ctranspath_model(cell_img_tensor)
    
    # 确保global_features有空间维度
    if global_features.dim() == 2:  # 如果是特征向量，则需要重塑
        N, C = global_features.shape
        global_features = global_features.view(N, C, 1, 1)  # 将其重塑为[N, C, 1, 1]形状
    
    # 对每个细胞区域使用ROI Align
    cell_features = []
    unique_masks = np.unique(masks)
    
    for label in unique_masks:
        if label == 0:
            continue  # 跳过背景
        
        # 获取细胞区域
        cell_mask = masks == label
        cell_mask_coords = np.column_stack(np.where(cell_mask))  # 获取细胞区域坐标
        
        # 使用ROI Align提取特征
        roi_boxes = np.array([[
            0,  # batch_index
            min(cell_mask_coords[:, 1]),  # xmin
            min(cell_mask_coords[:, 0]),  # ymin
            max(cell_mask_coords[:, 1]),  # xmax
            max(cell_mask_coords[:, 0])   # ymax
        ]])
        
        # 转为tensor并归一化为[0, 1]的范围
        roi_boxes = torch.tensor(roi_boxes, dtype=torch.float32).to(device)  # [num_rois, 5]
        
        # Rescale global features to match image size (height, width)
        global_features_resized = torch.nn.functional.interpolate(global_features, size=(image.size[1], image.size[0]), mode="bilinear", align_corners=False)
        
        with torch.no_grad():
            # 使用ROI Align提取特征，output_size = (1, 1) 保证每个细胞区域得到一个 1000 维特征向量
            roi_aligned_features = roi_align(global_features_resized, roi_boxes, output_size=(1, 1))  # (num_rois, 1000, 1, 1)
            
            # 这里是去除空间维度，保留 1000 维特征
            cell_features.append(roi_aligned_features.squeeze().cpu().detach().numpy())  # 转换为numpy数组
    
    return cell_features, masks, img

def visualize_clusters(image, masks, cluster_labels, save_path, k):
    """
    可视化聚类结果
    
    Args:
        image: 原始图像
        masks: 细胞分割掩码
        cluster_labels: 聚类标签
        save_path: 结果保存路径
        k: 聚类数
    """
    # 创建颜色映射
    colors = plt.cm.get_cmap('tab10', k)
    
    # 创建结果图像
    result_img = image.copy()
    
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
            result_img[cell_mask, c] = result_img[cell_mask, c] * 0.5 + color[c] * 0.5
            
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

def inference_with_roi_align(image_path, model_path, save_path, k=5):
    """
    推理主函数，使用ROI Align提取细胞特征进行聚类
    Args:
        image_path: 输入图像路径
        model_path: 训练好的模型路径
        save_path: 结果保存路径
        k: 聚类数
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    print("加载 Cellpose 模型...")
    cellpose_model = load_cellpose_model(model_type='cyto', gpu=torch.cuda.is_available())
    
    print("加载 CTransPath 模型...")
    ctranspath_model = ctranspath()
    ctranspath_model.to(device)
    
    # 加载训练好的Transformer模型
    model = TransformerEncoder(
        input_dim=1000,
        hidden_dim=512,
        n_heads=4,
        num_layers=6,
        output_dim=8,
        max_cells=255
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 提取细胞特征
    print("提取细胞特征...")
    cell_features, masks, original_image = extract_features_using_roi_align(
        image_path, cellpose_model, ctranspath_model, device)
    
    if len(cell_features) == 0:
        print("未检测到细胞，请检查输入图像。")
        return
    
    print(f"共检测到 {len(cell_features)} 个细胞")
    
    # 准备特征数据并填充到固定长度
    features_array = np.array(cell_features)
    padded_features, mask = pad_features_to_max_cells(features_array, max_cells=255)
    
    # 添加batch维度
    padded_features = torch.tensor(padded_features, dtype=torch.float32).unsqueeze(0).to(device)  # [1, max_cells, input_dim]
    mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(device)  # [1, max_cells]
    
    # 使用模型进行推理
    print("使用模型进行推理...")
    with torch.no_grad():
        cls_output, model_output, logits = model(padded_features, mask_tensor)
    
    # 提取有效的细胞特征（去除填充部分）
    valid_features = model_output[0, :len(cell_features)].cpu().numpy()  # [num_cells, output_dim]
    valid_logits = logits[0, :len(cell_features)].cpu().numpy()

    # 使用K-means聚类
    print(f"使用K-means进行 {k} 类聚类...")
    kmeans = KMeans(n_clusters=k, random_state=42)
    enhanced_features = build_cell_features(original_image, masks, valid_features)
    cluster_labels = kmeans.fit_predict(enhanced_features)
    
    # 可视化结果
    print("可视化聚类结果...")
    visualize_clusters(original_image, masks, cluster_labels, save_path, k)
    
    print("推理完成。")

# 运行推理
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='基于ROI Align提取细胞特征的推理')
    parser.add_argument('--image_path', type=str, required=True, help='输入图像路径')
    parser.add_argument('--model_path', type=str, required=True, help='训练好的模型路径')
    parser.add_argument('--save_path', type=str, required=True, help='结果保存路径')
    parser.add_argument('--k', type=int, default=5, help='聚类数')
    
    args = parser.parse_args()
    
    inference_with_roi_align(
        image_path=args.image_path,
        model_path=args.model_path,
        save_path=args.save_path,
        k=args.k
    )