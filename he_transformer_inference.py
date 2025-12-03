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

from utils import load_cellpose_model, preprocess_image, build_cell_features
from models import TransformerEncoder

# Try to import ctranpath, install if not available
try:
    from module.TransPath.ctran import ctranspath
except ImportError:
    raise ImportError("CTransPath module not found. Please install it from https://github.com/Xiyue-Wang/TransPath")

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

def inference(image_path, model_path, save_path, k=5):
    """
    推理主函数
    
    Args:
        image_path: 输入图像路径
        model_path: 训练好的模型路径
        save_path: 结果保存路径
        k: 聚类数
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化模型
    print("加载 Cellpose 模型...")
    cellpose_model = load_cellpose_model(model_type='cyto', gpu=torch.cuda.is_available())
    
    print("加载 CTransPath 模型...")
    ctranspath_model = ctranspath()
    ctranspath_model.to(device)
    
    # 初始化我们的transformer模型
    input_dim = 1000
    hidden_dim = 512
    n_heads = 4
    num_layers = 6
    output_dim = 8
    max_cells = 255
    
    model = TransformerEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        num_layers=num_layers,
        output_dim=output_dim,
        max_cells=max_cells
    ).to(device)
    
    # 加载训练好的模型权重
    print("加载训练好的模型权重...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 提取细胞特征
    print("提取细胞特征...")
    cell_features, masks, original_image = extract_cell_features_for_inference(
        image_path, cellpose_model, ctranspath_model, device)
    
    if len(cell_features) == 0:
        print("未检测到细胞，请检查输入图像。")
        return
    
    print(f"共检测到 {len(cell_features)} 个细胞")
    
    # 准备特征数据并填充到固定长度
    features_array = np.array(cell_features)
    padded_features, mask = pad_features_to_max_cells(features_array, max_cells)
    
    # 添加batch维度
    padded_features = torch.tensor(padded_features, dtype=torch.float32).unsqueeze(0).to(device)  # [1, max_cells, input_dim]
    mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(device)  # [1, max_cells]
    
    # 使用模型进行推理
    print("使用模型进行推理...")
    with torch.no_grad():
        cls_output, model_output = model(padded_features, mask_tensor)
    
    # 提取有效的细胞特征（去除填充部分）
    valid_features = model_output[0, :len(cell_features)].cpu().numpy()  # [num_cells, output_dim]
    
    # 使用K-means聚类
    print(f"使用K-means进行 {k} 类聚类...")
    kmeans = KMeans(n_clusters=k, random_state=42)
    enhanced_features = build_cell_features(original_image, masks, valid_features)
    cluster_labels = kmeans.fit_predict(enhanced_features)
    
    # 可视化结果
    print("可视化聚类结果...")
    visualize_clusters(original_image, masks, cluster_labels, save_path, k)
    
    # 输出每个细胞的聚类结果
    print("\n细胞聚类结果:")
    for i, label in enumerate(cluster_labels):
        print(f"细胞 {i+1}: 聚类 {label+1}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HE-Transformer 推理')
    parser.add_argument('--image_path', type=str, required=True, 
                        help='输入图像路径')
    parser.add_argument('--model_path', type=str, required=True,
                        help='训练好的模型路径')
    parser.add_argument('--save_path', type=str, required=True,
                        help='结果保存路径')
    parser.add_argument('--k', type=int, default=5,
                        help='聚类数')
    
    args = parser.parse_args()
    
    inference(
        image_path=args.image_path,
        model_path=args.model_path,
        save_path=args.save_path,
        k=args.k
    )