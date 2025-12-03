import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=3, temperature=0.5, positive_weight=1.0, negative_weight=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin  # 对负样本的 margin
        self.temperature = temperature  # 温度缩放，用于增强相似度差异
        self.positive_weight = positive_weight  # 正样本损失的权重
        self.negative_weight = negative_weight  # 负样本损失的权重

    def forward(self, image1_cls_token, image2_cls_token, label):
        """
        :param image1_cls_token: 第一个图像的 [CLS] token, 形状为 [batch_size, feature_dim]
        :param image2_cls_token: 第二个图像的 [CLS] token, 形状为 [batch_size, feature_dim]
        :param label: 标记正负样本，形状为 [batch_size]
        :return: 对比损失
        """
        # 计算余弦相似度
        cosine_similarity = F.cosine_similarity(image1_cls_token, image2_cls_token, dim=-1)

        cosine_similarity /= self.temperature

        # 计算对比损失
        loss = 0.0
        for i in range(cosine_similarity.size(0)):
            temp_loss = 0.0
            if label[i] == 1:  # 正样本对
                # 对于正样本，期望余弦相似度接近 1
                temp_loss = self.positive_weight * (1 - cosine_similarity[i]) ** 2
            else:  # 负样本对
                # 对于负样本，期望余弦相似度远离 1（即小于一定的 margin）
                temp_loss = torch.max(torch.tensor(0.0).to(image1_cls_token.device), self.margin - cosine_similarity[i]) ** 2
                temp_loss = self.negative_weight * temp_loss
            loss += temp_loss

        return loss.mean()  # 计算平均损失
    
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, feats, labels):
        """
        feats: (N, D)
        labels: (N,)
        """
        device = feats.device
        N = feats.size(0)

        # Normalize
        feats = F.normalize(feats, dim=1)

        # Cosine similarity: (N, N)
        sim_matrix = torch.matmul(feats, feats.t()) / self.temperature

        # Mask: same label = positive
        label_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).to(device)

        # Remove self-comparison
        diag_mask = torch.eye(N, dtype=torch.bool, device=device)
        label_mask = label_mask & (~diag_mask)

        # For each sample i:
        #   numerator = exp(sim(i, pos))
        #   denominator = exp(sim(i, pos)) + exp(sim(i, neg))
        loss_list = []

        for i in range(N):
            pos_idx = label_mask[i]  # (N,)
            neg_idx = ~label_mask[i] & (~diag_mask[i])

            pos_logits = sim_matrix[i][pos_idx]   # may be many
            neg_logits = sim_matrix[i][neg_idx]

            if pos_logits.numel() == 0:
                continue  # no positive example → skip

            numerator = torch.exp(pos_logits)
            denominator = torch.exp(torch.cat([pos_logits, neg_logits]))

            loss_i = -torch.log(numerator.sum() / denominator.sum())
            loss_list.append(loss_i)

        if len(loss_list) == 0:
            return torch.tensor(0.0, device=device)

        return torch.stack(loss_list).mean()
    
def compute_matching_loss(he_features, mif_features):
    """
    使用匈牙利匹配（Hungarian Matching）计算 HE ↔ mIF 的最优一一对应 Loss。

    Args:
        he_features: (B, D)
        mif_features: (B, D)

    Returns:
        matching_loss: 最优匹配后的 loss（scalar）
    """

    # --- Step 1: 计算相似度/距离矩阵 ---
    # 这里使用 cosine distance，越小越相似
    # 若你想用 L2，把下面一行换掉即可
    similarity = F.cosine_similarity(
        he_features.unsqueeze(1),  # (B, 1, D)
        mif_features.unsqueeze(0), # (1, B, D)
        dim=2
    )  # (B, B)

    # 转成 cost（匈牙利算法需要最小化）
    cost_matrix = 1 - similarity.detach().cpu().numpy()

    # --- Step 2: 匈牙利最优匹配 ---
    row_idx, col_idx = linear_sum_assignment(cost_matrix)

    # --- Step 3: 计算匹配后的 loss ---
    # 这里用 InfoNCE 更稳定
    matched_he = he_features[row_idx]        # (B, D)
    matched_mif = mif_features[col_idx]      # (B, D)

    # InfoNCE / negative cosine similarity
    matching_loss = 1 - F.cosine_similarity(matched_he, matched_mif).mean()

    return matching_loss