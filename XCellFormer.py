import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Optional, Union
from transformers import ViTFeatureExtractor, ViTModel

# ------------------------
# 配置：HuggingFace ViT-Huge
# ------------------------
VIT_H_MODEL_NAME = "google/vit-huge-patch14-224-in21k"

# 标准 ImageNet 预处理 (与 HF ViT 训练时一致)
# 注意：HF 的 processor 内部也是做 Resize 256 -> CenterCrop 224 -> Normalize
VIT_PREPROCESS_CONFIG = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "size": 224
}

class XCellFormer(nn.Module):
    def __init__(
        self,
        input_dim,          # 细胞特征维度
        hidden_dim,         # Small Branch 隐藏层维度
        n_heads,
        num_layers,
        output_dim=7,
        max_cells=255,
        num_cls_latents=32,
        use_large_vit=True,
        vit_weights_path=None,
        device='cuda'
    ):
        super().__init__()
        self.max_cells = max_cells
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_cls_latents = num_cls_latents
        self.device = device
        
        # 确保维度能被头数整除
        self.adjusted_output_dim = (hidden_dim // n_heads) * n_heads
        if self.adjusted_output_dim != hidden_dim:
            print(f"[Warning] hidden_dim ({hidden_dim}) adjusted to {self.adjusted_output_dim} for multi-head attention.")

        # --- Small Transformer Branch (细胞级 - 保持不变) ---
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, 
                nhead=n_heads, 
                batch_first=False,
                dropout=0.1,
                activation='gelu'
            ),
            num_layers=num_layers
        )
        self.cls_to_kv = nn.Linear(hidden_dim, num_cls_latents * self.adjusted_output_dim)
        self.cell_to_q = nn.Linear(hidden_dim, self.adjusted_output_dim)
        
        self.reverse_cross_attention = nn.MultiheadAttention(
            embed_dim=self.adjusted_output_dim, 
            num_heads=n_heads, 
            dropout=0.1, 
            batch_first=False
        )
        self.pre_norm = nn.LayerNorm(self.adjusted_output_dim)
        self.post_norm = nn.LayerNorm(self.adjusted_output_dim)
        self.final_output = nn.Linear(self.adjusted_output_dim, output_dim)

        # --- Large ViT Branch (全局级 - 使用 Transformers) ---
        if use_large_vit:
            print(f"🚀 Loading HuggingFace ViT-Huge: {VIT_H_MODEL_NAME}")
            
            self.feature_extractor = ViTFeatureExtractor.from_pretrained(vit_weights_path, local_files_only=True)
            self.vit_huge = ViTModel.from_pretrained(vit_weights_path, local_files_only=True)
            
            config = self.vit_huge.config
            self.vit_output_dim = config.hidden_size
            
            print(f"✅ ViT Loaded. Hidden Dim: {self.vit_output_dim}")
            
            # 推理时通常冻结，如果需微调请注释掉
            self.vit_huge.eval()
            # 若要微调 ViT:
            # for param in self.vit_huge.parameters():
            #     param.requires_grad = True
            
        else:
            self.vit_huge = None
            self.vit_output_dim = 0

    def forward(self, raw_images, x, mask):
        """
        Args:
            raw_images: [B, 3, 224, 224] Tensor. 
                        必须是经过归一化的 Tensor (mean=[0.485...], std=[0.229...])
            x:          [B, N_cell, Dim]  -> Cell 输入
            mask:       [B, N_cell]       -> 1.0 (Valid), 0.0 (Pad)
        Returns:
            cls_out:    [B, 1280]         -> 来自 ViT-Huge 的全局特征 (Pooler Output)
            cell_logits:[B, N_cell, 7]    -> 细胞分类结果
            ...
        """
        B, N_cell, _ = x.shape
        device = x.device
        
        # ==========================================
        # 1. Large ViT Branch (HuggingFace ViT-H)
        # ==========================================
        if self.vit_huge is not None:            
            inputs = self.feature_extractor(images=raw_images, return_tensors="pt").to(device)
            
            # ViTModel 前向传播
            with torch.no_grad():
                outputs = self.vit_huge(**inputs)
                cls_out = outputs.last_hidden_state 
            
            # cls_out shape: [B, 1280]
        else:
            cls_out = torch.zeros(B, self.vit_output_dim, device=device)

        # ==========================================
        # 2. Small Transformer Branch (Cell-level)
        # ==========================================
        
        # A. Embedding
        x_embed = self.embedding(x)           # [B, N_cell, hidden_dim]
        
        # B. Permute for Transformer (Seq, Batch, Dim)
        x_perm = x_embed.permute(1, 0, 2).contiguous()     # [N_cell, B, hidden_dim]
        
        # C. CLS Token
        cls_small = self.cls_token.expand(-1, B, -1).contiguous() # [1, B, hidden_dim]
        
        # D. Concat
        x_cat = torch.cat([cls_small, x_perm], dim=0)       # [N_cell+1, B, hidden_dim]
        
        # E. Attention Mask
        cls_mask_valid = torch.ones(B, 1, device=device)
        full_mask_valid = torch.cat([cls_mask_valid, mask], dim=1) # [B, N_cell+1]
        key_padding_mask = (full_mask_valid == 0)            # True where padding
        
        # F. Transformer Encoder
        x_trans = self.transformer(x_cat, src_key_padding_mask=key_padding_mask)
        
        cls_feat_small = x_trans[0]             # [B, hidden_dim]
        cell_feat_seq = x_trans[1:]             # [N_cell, B, hidden_dim]
        
        # G. Cross Attention Preparation
        # 1. Key/Value from CLS
        cls_proj = self.cls_to_kv(cls_feat_small) 
        cls_proj = cls_proj.view(B, self.num_cls_latents, self.adjusted_output_dim)
        cls_kv = cls_proj.permute(1, 0, 2).contiguous() 
        
        # 2. Query from Cells
        cell_feat_batch = cell_feat_seq.permute(1, 0, 2).contiguous() 
        cell_q = self.cell_to_q(cell_feat_batch)
        cell_q = self.pre_norm(cell_q)
        cell_q = cell_q.permute(1, 0, 2).contiguous()
        
        # H. Cross Attention Execution
        attn_out, _ = self.reverse_cross_attention(
            query=cell_q, 
            key=cls_kv, 
            value=cls_kv, 
            need_weights=False 
        )
        
        # I. Residual & Norm
        attn_out = attn_out + cell_q 
        attn_out = self.post_norm(attn_out)
        
        # J. Output Projection
        attn_out = attn_out.permute(1, 0, 2).contiguous()
        x_out = torch.sigmoid(self.final_output(attn_out)) # [B, N_cell, 7]

        return cls_out, x_out, x_out, attn_out