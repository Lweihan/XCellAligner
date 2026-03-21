import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from skimage import io
from typing import List, Tuple, Optional, Union

# 假设这些模块在你的环境中可用，如果路径不同请调整
from utils import load_cellpose_model 
from XCellFormer import XCellFormer
try:
    from module.TransPath.ctran import ctranspath
except ImportError:
    raise ImportError("CTransPath module not found.")

class CellInferenceEngine:
    def __init__(self, 
                 cellpose_model_path: Optional[str] = None,
                 ctranspath_checkpoint: Optional[str] = None,
                 xcell_checkpoint: Optional[str] = None,
                 xcell_config: dict = None,
                 device: str = None):
        """
        初始化推理引擎。
        
        Args:
            cellpose_model_path: Cellpose模型类型或路径 (e.g., 'cyto')
            ctranspath_checkpoint: CTransPath的权重路径 (如果需要加载特定权重)
            xcell_checkpoint: XCellTransformer的权重路径 (.bin 或 .pth)
            xcell_config: XCellTransformer的配置字典
            device: 推理设备 ('cuda' 或 'cpu')
        """
        torch.manual_seed(0)
        np.random.seed(0)
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"[InferenceEngine] Initializing on device: {self.device}")

        # 1. 加载 Cellpose
        print("[InferenceEngine] Loading Cellpose...")
        try:
            self.cellpose_model = load_cellpose_model(model_type=cellpose_model_path or 'cyto', device=self.device)
            # 存储模型类型用于多GPU处理
            self.cellpose_model_type = cellpose_model_path or 'cyto'
        except ImportError:
            # 备用方案：直接导入 cellpose (需安装 cellpose 包)
            from cellpose import models
            self.cellpose_model = models.Cellpose(model_type='cyto', gpu=(self.device.type=='cuda'))
            self.cellpose_model_type = 'cyto'

        # 2. 加载 CTransPath
        print("[InferenceEngine] Loading CTransPath...")
        try:
            from module.TransPath.ctran import ctranspath
            self.ctranspath_model = ctranspath()
            if ctranspath_checkpoint and os.path.exists(ctranspath_checkpoint):
                # 如果有特定权重则加载，否则使用默认初始化权重
                checkpoint = torch.load(ctranspath_checkpoint, map_location=self.device)
                if 'state_dict' in checkpoint:
                    self.ctranspath_model.load_state_dict(checkpoint['state_dict'], strict=False)
                else:
                    self.ctranspath_model.load_state_dict(checkpoint, strict=False)
            
            self.ctranspath_model.to(self.device)
            # 移除分类头，只保留特征提取
            if hasattr(self.ctranspath_model, 'head'):
                self.ctranspath_model.head = nn.Identity()
            self.ctranspath_model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load CTransPath: {e}")

        # 3. 加载 XCellTransformer (主模型)
        print("[InferenceEngine] Loading XCellTransformer...")
        if xcell_config is None:
            # 默认配置，需根据实际训练时的配置调整
            xcell_config = {
                "input_dim": 768,
                "hidden_dim": 256,
                "n_heads": 8,
                "num_layers": 4,
                "output_dim": 20, # 分类类别数
                "use_large_vit": True,
                "vit_weights_path": '/c23227/lwh/research/eccv_experiments/vith_weight' # 示例路径
            }
        
        self.extract_feature_model = XCellFormer(**xcell_config)
        
        if xcell_checkpoint and os.path.exists(xcell_checkpoint):
            checkpoint = torch.load(xcell_checkpoint, map_location=self.device)
            if 'state_dict' in checkpoint:
                self.extract_feature_model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                self.extract_feature_model.load_state_dict(checkpoint, strict=False)
            print(f"[InferenceEngine] Loaded weights from {xcell_checkpoint}")
        
        self.extract_feature_model.to(self.device)
        self.extract_feature_model.eval()

        # 预处理变换 (CTransPath 要求)
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Sobel 算子用于计算周长 (固定在 GPU)
        self.sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        self.sobel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)

    def _preprocess_image(self, img_np: np.ndarray) -> np.ndarray:
        """清洗图像以确保符合 Cellpose 和 RGB 要求"""
        if img_np.ndim == 2:
            img_np = np.stack([img_np, img_np, img_np], axis=-1)
        elif img_np.ndim == 3:
            if img_np.shape[2] == 1:
                img_np = np.repeat(img_np, 3, axis=2)
            elif img_np.shape[2] == 4:
                img_np = img_np[:, :, :3]
            elif img_np.shape[2] > 4:
                img_np = img_np[:, :, :3]
        
        if img_np.dtype != np.uint8:
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)
        return img_np

    def _extract_cell_features(self, img_np: np.ndarray, masks: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        针对每个细胞提取 CTransPath 特征。
        返回: (cell_features_list, masks_array)
        """
        import time
        
        cell_features = []
        # Create a new mask to guarantee contiguous labels matching the feature list
        new_masks = np.zeros_like(masks, dtype=np.int32)
        valid_label_idx = 1
        
        masks_tensor = torch.from_numpy(masks).to(self.device).to(torch.int32)
        unique_labels = torch.unique(masks_tensor)
        unique_labels = unique_labels[unique_labels != 0]

        if len(unique_labels) == 0:
            print(f"[{time.strftime('%H:%M:%S')}] 未检测到细胞")
            return [], new_masks

        print(f"[{time.strftime('%H:%M:%S')}] 开始处理 {len(unique_labels)} 个细胞的特征提取")
        
        # 计算全局最大面积用于归一化
        max_area = (masks_tensor > 0).sum().float()
        if max_area == 0: max_area = torch.tensor(1.0, device=self.device)

        for idx, label in enumerate(unique_labels):
            if idx % 100 == 0:  # 每100个细胞输出一次进度
                print(f"[{time.strftime('%H:%M:%S')}] 处理细胞进度: {idx}/{len(unique_labels)}")
            cell_mask = (masks_tensor == label).float()
            cell_area = cell_mask.sum()
            
            # 计算周长 (Sobel)
            cell_mask_unsq = cell_mask.unsqueeze(0).unsqueeze(0)
            grad_x = F.conv2d(cell_mask_unsq, self.sobel_x, padding=1)
            grad_y = F.conv2d(cell_mask_unsq, self.sobel_y, padding=1)
            cell_perimeter = torch.sqrt(grad_x**2 + grad_y**2).sum()
            
            roundness = 4 * np.pi * (cell_area / (cell_perimeter**2 + 1e-6))
            normalized_area = cell_area * 1e4 / (max_area + 1e-6)
            normalized_perimeter = cell_perimeter * 1e4 / (max_area + 1e-6)
            normalized_roundness = roundness * 1e4

            # 提取细胞 ROI
            # 注意：原代码逻辑是 mask * img，这会导致背景变黑。
            cell_region_np = (img_np * cell_mask.cpu().numpy().astype(np.uint8)[:, :, None]).astype(np.uint8)
            
            if cell_region_np.sum() == 0:
                continue
            
            cell_img_pil = Image.fromarray(cell_region_np)
            
            try:
                input_tensor = self.preprocess(cell_img_pil).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    output = self.ctranspath_model(input_tensor)
                
                if output.dim() == 4 and output.shape[-1] == 768:
                    output_permuted = output.permute(0, 3, 1, 2) # [1, 768, 7, 7]
                    feat_pooled = F.adaptive_avg_pool2d(output_permuted, (1, 1)) #[1, 768, 1, 1]
                    cell_img_feat = feat_pooled.squeeze(-1).squeeze(-1).squeeze(0) # [768] (保持为 Tensor 方便计算)
                    
                    # ==========================================
                    # 新增：形态学特征调制 (Morphology Modulation)
                    # ==========================================
                    
                    # 1. 获取形态学标量值 (.item() 转为 python float)
                    m_area = normalized_area.item()
                    m_peri = normalized_perimeter.item()
                    m_round = normalized_roundness.item()
                    
                    # 2. 构建形态学特征向量 [Area, Perimeter, Roundness]
                    morphology_vector = np.array([m_area, m_peri, m_round])
                    
                    # 3. 生成扩展向量 (Tile & Slice)
                    # 逻辑：将长度为 3 的向量重复，直到长度 >= 768，然后截取前 768 位
                    feat_dim = cell_img_feat.shape[0] # 通常是 768
                    repeat_times = int(np.ceil(feat_dim / 3.0))
                    
                    expanded_morphology = np.tile(morphology_vector, repeat_times)[:feat_dim]
                    
                    # 4. 执行逐元素相乘 (调制)
                    # 将 numpy 转为 tensor 并与 gpu 上的特征相乘
                    modulation_tensor = torch.from_numpy(expanded_morphology).float().to(self.device)
                    
                    # 【关键操作】：视觉特征 * 形态学因子
                    modulated_feat = cell_img_feat * modulation_tensor
                    
                    # 5. 转回 numpy 并保存
                    final_feat = modulated_feat.cpu().numpy()
                    
                    cell_features.append(final_feat)
                    # 将该细胞在新 mask 中设为递增且连续的 ID，保证与 feature index 一一对应!
                    new_masks[masks == label.item()] = valid_label_idx
                    valid_label_idx += 1
                    
                else:
                    # 如果输出维度不对，跳过
                    continue
            except Exception as e:
                print(f"Warning: Feature extraction failed for cell {label}: {e}")
                continue

        return cell_features, new_masks

    def _pad_features(self, features: List[np.ndarray], max_cells: int = 512) -> Tuple[torch.Tensor, torch.Tensor]:
        """填充特征到固定长度"""
        if not features:
            # 如果没有检测到细胞，返回全零
            return (torch.zeros(1, max_cells, 768, device=self.device), 
                    torch.zeros(1, max_cells, device=self.device))
        
        features_array = np.array(features)
        num_cells = features_array.shape[0]
        target_cells = max_cells if num_cells <= max_cells else num_cells
        
        padded_features = np.zeros((target_cells, features_array.shape[1]))
        padded_features[:num_cells] = features_array
        
        mask = np.zeros(target_cells)
        mask[:num_cells] = 1.0
        
        x_tensor = torch.tensor(padded_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        return x_tensor, mask_tensor

    def predict(self,
                image_source: Union[str, np.ndarray, Image.Image], 
                max_cells: int = 512,
                cell_features_list=None) -> dict:
        """
        执行单张图像的推理。
        
        Args:
            image_source: 图片路径 (str), numpy数组, 或 PIL Image
            max_cells: 最大处理细胞数量 (用于padding)
            
        Returns:
            dict: {
                'cls_output': 全局分类结果 (numpy array),
                'cell_logits': 每个细胞的 logits (numpy array, 长度为实际细胞数),
                'num_cells': 检测到的细胞数量,
                'masks': 分割掩码 (numpy array)
            }
        """
        if cell_features_list is None:
            import time
            
            # 1. 加载并清洗图像
            print(f"[{time.strftime('%H:%M:%S')}] 开始加载图像...")
            start_time = time.time()
            
            if isinstance(image_source, str):
                print(f"[{time.strftime('%H:%M:%S')}] 从文件加载图像: {image_source}")
                img_np = io.imread(image_source)
                img_pil = Image.open(image_source).convert("RGB")
                print(f"[{time.strftime('%H:%M:%S')}] 图像尺寸: {img_np.shape}")
            elif isinstance(image_source, np.ndarray):
                img_np = image_source
                img_pil = Image.fromarray(img_np).convert("RGB") if img_np.ndim == 3 else Image.fromarray(img_np).convert("L").convert("RGB")
            elif isinstance(image_source, Image.Image):
                img_pil = image_source.convert("RGB")
                img_np = np.array(img_pil)
            else:
                raise ValueError("Unsupported image source type")
    
            print(f"[{time.strftime('%H:%M:%S')}] 图像加载完成，耗时: {time.time() - start_time:.2f}秒")
            
            print(f"[{time.strftime('%H:%M:%S')}] 开始预处理图像...")
            preprocess_start = time.time()
            img_np_clean = self._preprocess_image(img_np)
            print(f"[{time.strftime('%H:%M:%S')}] 图像预处理完成，耗时: {time.time() - preprocess_start:.2f}秒")
    
            # 2. Cellpose 分割
            print(f"[{time.strftime('%H:%M:%S')}] 开始 Cellpose 细胞分割...")
            cellpose_start = time.time()
            try:
                # channels=[0,0] 表示 RGB 灰度图或者让模型自动判断，通常 RGB 用 [0,0] 或 [1,2] 取决于模型
                # 原代码建议 [0,0]
                masks, flows, styles = self.cellpose_model.eval(img_np_clean, diameter=18, channels=[0, 0])
                print(f"[{time.strftime('%H:%M:%S')}] Cellpose 分割完成，耗时: {time.time() - cellpose_start:.2f}秒")
                print(f"[{time.strftime('%H:%M:%S')}] 检测到细胞数量: {len(np.unique(masks)) - 1}")
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] Cellpose inference error: {e}")
                masks = np.zeros_like(img_np_clean[:,:,0], dtype=np.int32)
    
            # 3. 提取细胞特征
            print(f"[{time.strftime('%H:%M:%S')}] 开始提取细胞特征...")
            feature_start = time.time()
            cell_features_list, _ = self._extract_cell_features(img_np_clean, masks)
            print(f"[{time.strftime('%H:%M:%S')}] 细胞特征提取完成，耗时: {time.time() - feature_start:.2f}秒")

        if len(cell_features_list) > 0:
            print(cell_features_list[0][:10], len(cell_features_list[0]))
        if len(cell_features_list) > 1:
            print(cell_features_list[1][:10], len(cell_features_list[1]))
        
        if len(cell_features_list) == 0:
            print("No cells detected or features extracted.")
            # 返回空结果或默认值
            return {
                'cls_output': None,
                'cell_logits': None,
                'num_cells': 0,
                'masks': masks
            }

        # 4. Padding & Tensor 转换
        x_tensor, mask_tensor = self._pad_features(cell_features_list, max_cells=max_cells)

        # 5. 主模型推理 (XCellTransformer)
        try:
            with torch.no_grad():
                # 注意：原代码中 extract_feature_model 需要 raw_images (PIL 或 Tensor)
                # 这里传入 PIL 对象，具体看 XCellTransformer 内部如何处理 raw_images
                # 如果内部需要 Tensor，请在此处转换: transforms.ToTensor()(img_pil).unsqueeze(0).to(self.device)
                cls_output, model_output, logits, attn_weights = self.extract_feature_model(
                    raw_images=img_pil, 
                    x=x_tensor, 
                    mask=mask_tensor
                )
            
            # 6. 后处理输出
            # logits 形状通常是 [Batch, Max_Cells, Classes] 或 [Batch, Max_Cells]
            # 原代码取 logits[0, :len(cell_features_list)]
            actual_num_cells = len(cell_features_list)
            
            cls_out_np = cls_output.cpu().numpy()
            # 确保 logits 切片正确
            if logits.dim() == 2: # [Batch, Max_Cells] (例如回归任务或二分类标量)
                cell_logits_np = logits[0, :actual_num_cells].cpu().numpy()
            elif logits.dim() == 3: # [Batch, Max_Cells, Num_Classes]
                cell_logits_np = logits[0, :actual_num_cells, :].cpu().numpy()
            else:
                cell_logits_np = logits.cpu().numpy()

            return {
                'cls_output': cls_out_np,
                'cell_logits': cell_logits_np,
                'num_cells': actual_num_cells,
                'masks': masks
            }

        except Exception as e:
            print(f"Model inference error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'cls_output': None,
                'cell_logits': None,
                'num_cells': 0,
                'masks': masks,
                'error': str(e)
            }

    def predict_batch(self, image_paths: List[str], max_cells: int = 512) -> List[dict]:
        """批量推理"""
        results = []
        for path in image_paths:
            res = self.predict(path, max_cells=max_cells)
            results.append(res)
        return results