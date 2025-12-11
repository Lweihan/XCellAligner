import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
# import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from collections import defaultdict
import torch.nn.functional as F
# # 设置多进程启动方式为 spawn
# mp.set_start_method('spawn', force=True)

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_heads, num_layers, output_dim=7, max_cells=255):
        super(TransformerEncoder, self).__init__()

        self.max_cells = max_cells
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # 调整 output_dim 以确保它可以被 MultiheadAttention 的 head_dim 整除
        self.adjusted_output_dim = (output_dim // n_heads) * n_heads  # Ensure divisibility by n_heads
        self.embedding = nn.Linear(input_dim, hidden_dim)  # 输入特征映射到 hidden_dim
        self.cls_to_kv = nn.Linear(hidden_dim, self.adjusted_output_dim)
        self.cell_to_q = nn.Linear(hidden_dim, self.adjusted_output_dim)
        self.fc_output = nn.Linear(hidden_dim, output_dim)  # 用于输出特征映射到 output_dim

        # Transformer Encoder 层
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads),
            num_layers=num_layers
        )

        # 用于对比学习的 cls_token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))  # 用于对比学习的cls_token

        #反向 Cross Attention 层
        self.reverse_cross_attention = nn.MultiheadAttention(
            embed_dim=self.adjusted_output_dim,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=False  # 因为我们使用 [seq_len, batch, dim]
        )

        # LayerNorm 和 Dropout
        self.layer_norm = nn.LayerNorm(self.adjusted_output_dim)
        self.dropout = nn.Dropout(0.1)

        # 最终输出映射层
        self.final_output = nn.Linear(self.adjusted_output_dim, output_dim)

    def forward(self, x, mask):
        batch_size, max_cells, input_dim = x.shape

        # 构建输入序列：映射到 hidden_dim
        x = self.embedding(x)  # [batch_size, max_cells, hidden_dim]
        x = x.permute(1, 0, 2)  # 转置为 [max_cells, batch_size, hidden_dim]

        # 在输入中添加 [CLS] token
        cls_tokens = self.cls_token.repeat(1, x.size(1), 1)  # [1, batch_size, hidden_dim]
        x = torch.cat([cls_tokens, x], dim=0)  # [max_cells+1, batch_size, hidden_dim]

        # 构建mask，添加CLS token位置
        cls_mask = torch.ones(batch_size, 1).to(mask.device)  # CLS token位置为有效
        extended_mask = torch.cat([cls_mask, mask], dim=1)  # [batch_size, max_cells+1]
        extended_mask = extended_mask.to(x.device)

        # 经过 Transformer Encoder
        x = self.transformer(x, src_key_padding_mask=extended_mask == 0)  # 使用掩码忽略填充部分

        # 分离CLS token和细胞特征
        cls_token_output = x[0, :, :]  # [batch_size, hidden_dim]
        cell_features = x[1:, :, :]    # [max_cells, batch_size, hidden_dim]

        # 转换回 [batch_size, max_cells, hidden_dim]
        cell_features = cell_features.permute(1, 0, 2)

        # 对每个细胞特征应用线性映射
        cls_token_linear = self.fc_output(cls_token_output)
        # Project [CLS] to adjusted_output_dim for Key/Value
        cls_proj = self.cls_to_kv(cls_token_output)  # [batch_size, adjusted_output_dim]
        cls_kv = cls_proj.unsqueeze(0)  # [1, batch_size, adjusted_output_dim]

        # Project cell_features to adjusted_output_dim for Query
        cell_query = self.cell_to_q(cell_features)  # [batch_size, max_cells, adjusted_output_dim]
        cell_query = cell_query.permute(1, 0, 2)    # [max_cells, batch_size, adjusted_output_dim]

        # Cross-Attention: each cell queries the global [CLS]
        attn_output, _ = self.reverse_cross_attention(
            query=cell_query,
            key=cls_kv,
            value=cls_kv
        )  # output: [max_cells, batch_size, adjusted_output_dim]

        # Reshape back
        attn_output = attn_output.permute(1, 0, 2)  # [batch_size, max_cells, adjusted_output_dim]

        # Optional: residual connection from original cell projection
        # cell_proj_for_res = self.cell_to_q_res(cell_features)  # or reuse cell_query after permute
        # But simpler: just use attn_output as new rep

        # LayerNorm + Dropout
        x = attn_output + cell_query.permute(1, 0, 2)
        x = self.layer_norm(x)

        # Final output
        x = self.final_output(x)  # [batch_size, max_cells, output_dim]

        logits = torch.sigmoid(x)

        return cls_token_linear, x, logits