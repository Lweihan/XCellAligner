import torch, torchvision
import torch.nn as nn
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from cellpose import models
from PIL import Image
from module.TransPath.ctran import ctranspath

def load_cellpose_model(model_type='cyto2', device=None):
    from cellpose import models
    if device is not None and 'cuda' in str(device):
        gpu = True
    else:
        gpu = False
        device = None  # CellposeModel expects device=None for CPU
    return models.CellposeModel(
        gpu=gpu,
        pretrained_model=model_type,
        device=device
    )

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    return img_np

def extract_cell_features(image_path, label_path, cellpose_model, ctranspath_model, device):
    """
    提取细胞特征，并根据标签图为每个细胞分配类别
    
    Args:
        image_path: 图像路径
        label_path: 标签图像路径
        cellpose_model: Cellpose模型
        ctranspath_model: CTransPath模型
        device: 设备
    
    Returns:
        tuple: (cell_features, cell_labels, masks)
               cell_features: 特征列表
               cell_labels: 每个细胞对应的类别标签
               masks: 分割掩码
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
    
    # 加载标签图像
    label_img = Image.open(label_path).convert('L')
    label_np = np.array(label_img)
    
    # 提取每个细胞的特征
    cell_features = []
    cell_labels = []  # 存储每个细胞的类别标签
    unique_masks = np.unique(masks)
    
    for label in unique_masks:
        if label == 0:
            continue  # 跳过背景
            
        # 获取细胞区域
        cell_mask = masks == label
        cell_img = img * cell_mask[..., None]  # 通过掩码提取细胞区域
        cell_img_pil = Image.fromarray(cell_img)
        
        # 计算该细胞的主要类别标签
        # 找到与细胞区域重叠最多的类别
        overlap_counts = {}
        total_cell_pixels = np.sum(cell_mask)
        
        if total_cell_pixels > 0:
            # 在细胞掩码区域内统计各类别像素数量
            for y in range(cell_mask.shape[0]):
                for x in range(cell_mask.shape[1]):
                    if cell_mask[y, x]:  # 如果是细胞区域
                        pixel_label = label_np[y, x]  # 获取对应标签值
                        if pixel_label not in overlap_counts:
                            overlap_counts[pixel_label] = 0
                        overlap_counts[pixel_label] += 1
            
            # 找到占比最大的类别作为该细胞的标签
            if overlap_counts:
                major_label = max(overlap_counts, key=overlap_counts.get)
                # 只有当主要类别不是背景(0)且占比较大时才保留
                if major_label != 0 and overlap_counts[major_label] > 0.5 * total_cell_pixels:
                    # 预处理并输入CTransPath模型
                    cell_img_tensor = preprocess(cell_img_pil).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        # 使用 CTransPath 提取特征
                        features = ctranspath_model(cell_img_tensor)  # 得到细胞特征向量
                    
                    cell_features.append(features.squeeze().cpu().numpy())  # 获取特征并存储
                    cell_labels.append(major_label)  # 存储类别标签

    return cell_features, cell_labels, masks

def build_cell_features(original_image, masks, valid_features):
    """
    构建增强细胞特征：
        - 模型输出特征 valid_features
        - 细胞面积
        - RGB 平均颜色
        - 灰度平均强度

    Args:
        original_image: numpy array [H, W, 3], uint8
        masks: numpy array [H, W], int, 每个细胞一个 ID
        valid_features: numpy array [num_cells, feature_dim]

    Returns:
        enhanced_features: numpy array [num_cells, feature_dim + 5]
    """
    
    original_image = original_image.astype(np.float32)
    unique_ids = np.unique(masks)
    
    # 排除背景
    cell_ids = unique_ids[unique_ids != 0]
    num_cells = len(cell_ids)

    feature_dim = valid_features.shape[1]
    enhanced_features = np.zeros((num_cells, feature_dim + 5), dtype=np.float32)

    # 转灰度
    gray = (
        0.299 * original_image[:, :, 0] +
        0.587 * original_image[:, :, 1] +
        0.114 * original_image[:, :, 2]
    )

    for idx, cid in enumerate(cell_ids):
        cell_mask = masks == cid

        area = np.sum(cell_mask)
        mean_color = np.mean(original_image[cell_mask], axis=0)  # (3,)
        mean_gray = np.mean(gray[cell_mask])
        handcrafted = np.array([
            area,
            mean_color[0], mean_color[1], mean_color[2],
            mean_gray
        ], dtype=np.float32)

        enhanced_features[idx] = np.concatenate([
            valid_features[idx],
            handcrafted
        ])

    return enhanced_features