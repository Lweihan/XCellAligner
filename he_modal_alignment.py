import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
import torch.nn.functional as F
import re

# 导入必要的模块
from utils import load_cellpose_model, extract_cell_features
from models import TransformerEncoder
from loss import InfoNCELoss, compute_matching_loss

# 尝试导入ctranspath
try:
    from module.TransPath.ctran import ctranspath
except ImportError:
    raise ImportError("CTransPath module not found. Please install it from https://github.com/Xiyue-Wang/TransPath")

class HEModalAlignment:
    def __init__(self, he_model_path, modal_model_path, he_data_root, mif_data_root):
        """
        初始化HE和mIF模态对齐模型
        
        Args:
            he_model_path (str): HE Transformer模型权重路径
            modal_model_path (str): Modal Transformer模型权重路径
            he_data_root (str): HE图像文件夹路径
            mif_data_root (str): mIF图像文件夹路径
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 加载模型
        self.he_model = self.load_he_model(he_model_path)
        self.modal_model = self.load_modal_model(modal_model_path)
        
        # 初始化其他组件
        self.info_nce_loss = InfoNCELoss()
        
        # 加载数据
        self.he_data_root = he_data_root
        self.mif_data_root = mif_data_root

    def load_he_model(self, model_path):
        """加载HE Transformer模型"""
        print("Loading Cellpose model...")
        cellpose_model = load_cellpose_model(model_type='cyto', gpu=torch.cuda.is_available())
        
        print("Loading CTransPath model...")
        ctranspath_model = ctranspath()
        ctranspath_model.to(self.device)
        
        # HE Transformer模型参数
        input_dim = 1000  # CTransPath输出特征维度
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
        ).to(self.device)
        
        # 加载预训练权重
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded HE model from {model_path}")
        else:
            print(f"HE model checkpoint not found at {model_path}, using randomly initialized model")
            
        return model

    def load_modal_model(self, model_path):
        """加载Modal Transformer模型"""
        # 需要确定输入维度，这里暂时设为一个默认值
        input_dim = 7  # 假设有4个通道，实际应该根据数据动态确定
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
        ).to(self.device)
        
        # 加载预训练权重
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded Modal model from {model_path}")
        else:
            print(f"Modal model checkpoint not found at {model_path}, using randomly initialized model")
            
        return model

    def extract_he_features(self, image_path):
        """
        从HE图像中提取细胞特征（不需要标签）
        
        Args:
            image_path (str): HE图像路径
            
        Returns:
            tuple: (features, masks) 提取的特征和掩码
        """
        # 加载Cellpose和CTransPath模型（如果尚未加载）
        cellpose_model = load_cellpose_model(model_type='cyto', gpu=torch.cuda.is_available())
        ctranspath_model = ctranspath()
        ctranspath_model.to(self.device)
        
        try:
            # 预处理图像
            img = Image.open(image_path).convert('RGB')
            img_np = np.array(img)
            
            # 使用Cellpose进行细胞分割
            masks, flows, styles = cellpose_model.eval(img_np, diameter=None, channels=[0, 0])

            # 使用CTransPath模型进行编码
            ctranspath_model.eval()  # 设置为评估模式

            # CTransPath模型的预处理函数
            from torchvision import transforms
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
                cell_img = img_np * cell_mask[..., None]  # 通过掩码提取细胞区域
                cell_img_pil = Image.fromarray(cell_img)
                
                # 预处理并输入CTransPath模型
                cell_img_tensor = preprocess(cell_img_pil).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    # 使用 CTransPath 提取特征
                    features = ctranspath_model(cell_img_tensor)  # 得到细胞特征向量
                
                cell_features.append(features.squeeze().cpu().numpy())  # 获取特征并存储

            # 转换为张量并填充到固定大小
            max_cells = 255
            if len(cell_features) > 0:
                # 限制最大细胞数
                if len(cell_features) > max_cells:
                    cell_features = cell_features[:max_cells]
                
                # 填充或截断到max_cells
                padded_features = np.zeros((max_cells, 1000))  # CTransPath特征维度是1000
                mask = np.zeros(max_cells)
                
                num_cells = len(cell_features)
                for i in range(num_cells):
                    padded_features[i] = cell_features[i]
                    mask[i] = 1
                
                # 转换为PyTorch张量
                features_tensor = torch.FloatTensor(padded_features).unsqueeze(0)  # 添加batch维度
                mask_tensor = torch.FloatTensor(mask).unsqueeze(0)
                
                return features_tensor, mask_tensor
            else:
                print(f"No cells detected in {image_path}")
                return None, None
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None, None

    def extract_modal_features(self, mif_image_paths):
        """
        从mIF图像中提取特征
        
        Args:
            mif_image_paths (list): mIF图像路径列表
            
        Returns:
            tuple: (features, masks) 提取的特征和掩码
        """
        # 使用CellDensityExtractor提取密度特征
        try:
            from module.ModalEncoder.cell_density_extractor import CellDensityExtractor
            
            extractor = CellDensityExtractor()
            
            # 加载图像
            images = []
            for path in mif_image_paths:
                img = Image.open(path)
                images.append(np.array(img))
            
            # 提取密度矩阵 (cells, channels)
            density_matrix = extractor.process_image_pair(images, [0] + [1] * (len(images) - 1))
            
            # 转换为张量并填充到固定大小
            max_cells = 255
            num_cells = density_matrix.shape[0]
            
            if num_cells > 0:
                # 填充或截断到max_cells
                if num_cells > max_cells:
                    density_tensor = torch.FloatTensor(density_matrix[:max_cells, :])
                    mask = torch.ones(max_cells)
                else:
                    padding_size = max_cells - num_cells
                    density_tensor = torch.FloatTensor(np.pad(density_matrix, ((0, padding_size), (0, 0)), mode='constant'))
                    mask = torch.cat([torch.ones(num_cells), torch.zeros(padding_size)])
                
                # 添加batch维度
                features_tensor = density_tensor.unsqueeze(0)
                mask_tensor = mask.unsqueeze(0)
                
                return features_tensor, mask_tensor
            else:
                print("No cells detected in mIF images")
                return None, None
        except Exception as e:
            print(f"Error extracting modal features: {e}")
            return None, None

    def compute_contrastive_loss(self, paired_features, unpaired_features_list):
        # print(len(paired_features))
        # print(len(unpaired_features_list))

        # print("\n=== DEBUG SHAPES ===")
        # print("len(paired_features):", len(paired_features))
        # print("len(unpaired_features_list):", len(unpaired_features_list))

        for i, (h, m) in enumerate(paired_features):
            print(f"paired {i}: he={h.shape}, mif={m.shape}")

        for i, (h, m) in enumerate(unpaired_features_list):
            print(f"unpaired {i}: he={h.shape}, mif={m.shape}")

        all_pairs = paired_features + unpaired_features_list
        all_labels = [1] * len(paired_features) + [0] * len(unpaired_features_list)

        if len(all_pairs) == 0:
            return torch.tensor(0.0, device=self.device)

        # 限制负样本数量
        max_negative_samples = 32
        if len(unpaired_features_list) > max_negative_samples:
            import random
            neg_indices = random.sample(range(len(unpaired_features_list)), max_negative_samples)
            unpaired_features_list = [unpaired_features_list[i] for i in neg_indices]
            all_pairs = paired_features + unpaired_features_list
            all_labels = [1] * len(paired_features) + [0] * len(unpaired_features_list)

        # 收集所有特征
        all_he_features = [he for he, _ in all_pairs]
        all_mif_features = [mif for _, mif in all_pairs]

        if len(all_he_features) == 0:
            return torch.tensor(0.0, device=self.device)

        he_features = torch.cat(all_he_features, dim=0)
        mif_features = torch.cat(all_mif_features, dim=0)
        labels = torch.LongTensor(all_labels).to(self.device)

        # ---- 关键修复：保证相似度永远是 1D 向量 ----
        cosine_sim = F.cosine_similarity(he_features, mif_features, dim=-1).view(-1)

        # 掩码
        pos_mask = (labels == 1)
        neg_mask = (labels == 0)

        pos_loss = torch.tensor(0.0, device=self.device)
        neg_loss = torch.tensor(0.0, device=self.device)

        # 正样本
        if pos_mask.any():
            pos_sim = cosine_sim[pos_mask]
            if pos_sim.numel() > 0:
                pos_loss = (1 - pos_sim).pow(2).mean()

        # 负样本
        if neg_mask.any():
            neg_sim = cosine_sim[neg_mask]
            if neg_sim.numel() > 0:
                neg_loss = neg_sim.pow(2).mean()

        return 0.5 * pos_loss + 0.5 * neg_loss


    def align_he_modal(self, paired_data, unpaired_data, optimizer):
        """
        对齐HE和mIF模态，并计算 Hungarian matching loss + 对比学习 loss
        
        Args:
            paired_data: 配对数据列表 [{he_image, mif_images}, ...]
            unpaired_data: 非配对数据列表 [{he_image, mif_images}, ...]
            optimizer: 优化器
            
        Returns:
            dict: 包含损失值的字典
        """
        paired_features = []         # 图像级特征（平均池化）
        he_feats_for_matching = []   # Transformer输出细胞级特征，用于 Hungarian
        mif_feats_for_matching = []

        # --- 处理配对数据 ---
        for data in paired_data:
            he_features, he_mask = self.extract_he_features(data['he_image'])
            mif_features, mif_mask = self.extract_modal_features(data['mif_images'])

            if he_features is None or mif_features is None:
                continue

            he_features = he_features.to(self.device)
            he_mask = he_mask.to(self.device)
            mif_features = mif_features.to(self.device)
            mif_mask = mif_mask.to(self.device)

            # Transformer输出
            _, he_output = self.he_model(he_features, he_mask)         # [1, max_cells, feat_dim]
            with torch.no_grad():
                self.modal_model.eval()
                _, mif_output = self.modal_model(mif_features, mif_mask) # [1, max_cells, feat_dim]

            he_valid = he_mask.bool()[0]
            mif_valid = mif_mask.bool()[0]

            he_valid_feats = he_output[0][he_valid]      # [num_cells, feat_dim]
            mif_valid_feats = mif_output[0][mif_valid]   # [num_cells, feat_dim]

            if he_valid_feats.shape[0] == 0 or mif_valid_feats.shape[0] == 0:
                continue

            # 图像级平均池化
            he_pooled = he_valid_feats.mean(dim=0, keepdim=True)
            mif_pooled = mif_valid_feats.mean(dim=0, keepdim=True)
            paired_features.append((he_pooled, mif_pooled))

            # 保存细胞级特征用于 Hungarian loss
            he_feats_for_matching.append(he_valid_feats)
            mif_feats_for_matching.append(mif_valid_feats)

        # --- 处理非配对数据 ---
        unpaired_features_list = []
        for data in unpaired_data:
            he_features, he_mask = self.extract_he_features(data['he_image'])
            mif_features, mif_mask = self.extract_modal_features(data['mif_images'])

            if he_features is None or mif_features is None:
                continue

            he_features = he_features.to(self.device)
            he_mask = he_mask.to(self.device)
            mif_features = mif_features.to(self.device)
            mif_mask = mif_mask.to(self.device)

            _, he_output = self.he_model(he_features, he_mask)
            with torch.no_grad():
                self.modal_model.eval()
                _, mif_output = self.modal_model(mif_features, mif_mask)

            he_valid = he_mask.bool()[0]
            mif_valid = mif_mask.bool()[0]

            he_valid_feats = he_output[0][he_valid]
            mif_valid_feats = mif_output[0][mif_valid]

            if he_valid_feats.shape[0] == 0 or mif_valid_feats.shape[0] == 0:
                continue

            he_pooled = he_valid_feats.mean(dim=0, keepdim=True)
            mif_pooled = mif_valid_feats.mean(dim=0, keepdim=True)
            unpaired_features_list.append((he_pooled, mif_pooled))

        # --- 对比学习 loss ---
        if len(paired_features) == 0 and len(unpaired_features_list) == 0:
            return {"contrastive_loss": 0.0, "matching_loss": 0.0}

        contrastive_loss = self.compute_contrastive_loss(paired_features, unpaired_features_list)

        # --- Hungarian matching loss ---
        # 简单实现：使用 MSE + Hungarian 匹配
        matching_loss = torch.tensor(0.0, device=self.device)
        if len(he_feats_for_matching) > 0:
            for he_cells, mif_cells in zip(he_feats_for_matching, mif_feats_for_matching):
                # 计算 pairwise cosine distance matrix
                dist_matrix = 1 - F.cosine_similarity(
                    he_cells.unsqueeze(1), mif_cells.unsqueeze(0), dim=-1
                )  # [num_he_cells, num_mif_cells]

                # 匈牙利匹配（linear_sum_assignment）
                import scipy.optimize
                dist_np = dist_matrix.detach().cpu().numpy()
                row_ind, col_ind = scipy.optimize.linear_sum_assignment(dist_np)

                matched_he = he_cells[row_ind]
                matched_mif = mif_cells[col_ind]

                # 匹配的 MSE 作为 loss
                matching_loss += F.mse_loss(matched_he, matched_mif)

        # --- 总 loss & backward ---
        total_loss = contrastive_loss + matching_loss

        self.he_model.train()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        return {
            "contrastive_loss": contrastive_loss.item(),
            "matching_loss": matching_loss.item()
        }

    def parse_filename(self, filename):
        """
        解析文件名获取坐标信息
        
        Args:
            filename (str): 文件名
            
        Returns:
            dict: 包含坐标信息的字典
        """
        # 解析HE文件名格式: he_x{X}_y{Y}.ext
        he_match = re.match(r'he_x(\d+)_y(\d+)\.\w+', filename)
        if he_match:
            return {
                'type': 'he',
                'x': int(he_match.group(1)),
                'y': int(he_match.group(2))
            }
        
        # 解析mIF文件名格式: mIF{N}_x{X}_y{Y}.ext
        mif_match = re.match(r'mF\d+_x(\d+)_y(\d+)\.\w+', filename)
        if mif_match:
            return {
                'type': 'mif',
                'x': int(mif_match.group(1)),
                'y': int(mif_match.group(2))
            }
        
        return None

    def get_matching_files(self):
        """
        根据文件名坐标匹配HE和mIF图像
        
        Returns:
            tuple: 包含匹配文件信息、HE图像列表、mIF图像列表和通道列表的元组
        """
        # 获取HE图像文件
        he_image_dir = self.he_data_root
        he_images = {}
        for f in os.listdir(he_image_dir):
            if f.startswith('he_x') and f.endswith(('.png', '.jpg', '.jpeg')):
                coords = self.parse_filename(f)
                if coords:
                    key = (coords['x'], coords['y'])
                    he_images[key] = os.path.join(he_image_dir, f)
        
        # 获取mIF通道目录
        mif_channels = [d for d in os.listdir(self.mif_data_root) 
                       if os.path.isdir(os.path.join(self.mif_data_root, d))]
        
        # 获取mIF图像文件
        mif_images = {}
        for channel in mif_channels:
            channel_path = os.path.join(self.mif_data_root, channel)
            for f in os.listdir(channel_path):
                if f.startswith('mF') and f.endswith(('.png', '.jpg', '.jpeg')):
                    coords = self.parse_filename(f)
                    if coords:
                        key = (coords['x'], coords['y'])
                        if key not in mif_images:
                            mif_images[key] = []
                        mif_images[key].append(os.path.join(channel_path, f))
        
        print(f"Found {len(he_images)} HE images, {len(mif_images)} mIF images")
        # 匹配HE和mIF图像
        matched_files = []
        all_mif_files = []
        for key in he_images:
            if key in mif_images:
                # 确保所有mIF通道都有图像
                if len(mif_images[key]) == len(mif_channels):
                    matched_files.append({
                        'he_image': he_images[key],
                        'mif_images': mif_images[key]
                    })
            # 收集所有mIF文件用于构建非配对数据
            if key in mif_images:
                all_mif_files.extend(mif_images[key])
        
        print(f"Matched {len(matched_files)} HE-mIF image pairs")
        return matched_files, list(he_images.values()), all_mif_files, mif_channels

    def create_unpaired_data(self, he_images, mif_files, mif_channels, num_unpaired_per_he=32):
        """
        创建大量 non-paired (HE ↔ mIF) 样本，用于对比学习。

        Args:
            he_images (list): 所有 HE 图像路径
            mif_files (list): 所有 mIF 图像
            mif_channels (list): mIF 通道列表
            num_unpaired_per_he (int): 每个 HE 图像随机生成多少组 unpaired 组合

        Returns:
            list: 长度约 = len(he_images) * num_unpaired_per_he
        """
        import random
        unpaired_data = []

        for he in he_images:
            for _ in range(num_unpaired_per_he):
                # 随机为每个通道选一个 mIF 文件
                selected_mif = [random.choice(mif_files) for _ in mif_channels]

                unpaired_data.append({
                    "he_image": he,
                    "mif_images": selected_mif
                })

        print(f"[INFO] Created {len(unpaired_data)} unpaired HE–mIF combinations.")
        return unpaired_data


    def train(self, epochs=100, learning_rate=1e-4, save_path="he_modal_alignment_model.pth"):
        """
        训练HE和mIF模态对齐模型
        
        Args:
            epochs (int): 训练轮数
            learning_rate (float): 学习率
            save_path (str): 模型保存路径
        """
        # 创建优化器 (只优化he_model)
        optimizer = optim.Adam(self.he_model.parameters(), lr=learning_rate)
        
        # 获取匹配的文件
        matched_files, he_images, mif_files, mif_channels = self.get_matching_files()
        
        if len(matched_files) == 0:
            print("No matching HE-mIF image pairs found!")
            return
        
        # 创建非配对数据
        unpaired_files = self.create_unpaired_data(he_images, mif_files, mif_channels)
        
        # 训练循环
        best_loss = float('inf')
        
        for epoch in range(epochs):
            total_contrastive_loss = 0
            total_matching_loss = 0
            num_batches = 0
            
            # 遍历匹配的文件对
            progress_bar = tqdm(range(0, len(matched_files), 16), desc=f'Epoch {epoch+1}/{epochs}')  # 每批次16个配对样本
            for i in progress_bar:
                # 获取当前批次的配对数据
                batch_paired = matched_files[i:i+16]

                # 随机选择非配对数据（与配对数据数量相同）
                import random
                batch_unpaired = random.sample(unpaired_files, min(len(unpaired_files), len(batch_paired)))
                
                # 执行对齐
                losses = self.align_he_modal(batch_paired, batch_unpaired, optimizer)
                
                total_matching_loss += losses["matching_loss"]
                total_contrastive_loss += losses["contrastive_loss"]
                num_batches += 1
                
                # 更新进度条
                progress_bar.set_postfix({
                    'contrastive_loss': losses["contrastive_loss"],
                    'matching_loss': losses["matching_loss"],
                })
            
            # 计算平均损失
            if num_batches > 0:
                avg_contrastive_loss = total_contrastive_loss / num_batches
                
                print(f'Epoch [{epoch+1}/{epochs}], Avg Contrastive Loss: {avg_contrastive_loss:.4f}')
                
                # 保存最佳模型
                if avg_contrastive_loss < best_loss:
                    best_loss = avg_contrastive_loss
                    torch.save(self.he_model.state_dict(), save_path)
                    print(f"Saved best HE model with loss: {best_loss:.4f}")

def main():
    parser = argparse.ArgumentParser(description='HE and mIF Modal Alignment')
    parser.add_argument('--he_model_path', type=str, required=True, 
                        help='Path to HE Transformer model weights')
    parser.add_argument('--modal_model_path', type=str, required=True,
                        help='Path to Modal Transformer model weights')
    parser.add_argument('--he_data_root', type=str, required=True,
                        help='Path to HE image folder')
    parser.add_argument('--mif_data_root', type=str, required=True,
                        help='Path to mIF image folder containing n subfolders for different channels')
    parser.add_argument('--save_path', type=str, default='he_modal_alignment_model.pth',
                        help='Path to save the aligned model')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    
    args = parser.parse_args()
    
    # 创建对齐模型
    alignment_model = HEModalAlignment(
        he_model_path=args.he_model_path,
        modal_model_path=args.modal_model_path,
        he_data_root=args.he_data_root,
        mif_data_root=args.mif_data_root
    )
    
    # 开始训练
    alignment_model.train(
        epochs=args.epochs,
        learning_rate=args.lr,
        save_path=args.save_path
    )

if __name__ == "__main__":
    main()