"""
Hyperbolic projection layer and hyperbolic similarity computation module
Implements Poincaré ball model hyperbolic space projection and similarity computation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HyperbolicProjection(nn.Module):
    """
    双曲投影层：将欧几里得空间的特征投影到Poincaré球模型的双曲空间
    
    参数:
        input_dim: 输入特征维度（CLAP音频嵌入维度，通常是512）
        output_dim: 输出特征维度（双曲空间维度，默认与input_dim相同）
        c: 曲率参数，控制双曲空间的曲率（默认1.0，越小曲率越大）
        clip_r: 裁剪半径，防止数值不稳定（默认0.9）
    """
    def __init__(self, input_dim=512, output_dim=512, c=1.0, clip_r=0.9):
        super(HyperbolicProjection, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.c = c  # 曲率参数
        self.clip_r = clip_r  # 裁剪半径，防止数值不稳定
        
        # 线性投影层
        self.linear = nn.Linear(input_dim, output_dim)
        
        # 初始化权重
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def expmap(self, x):
        """
        指数映射：将欧几里得空间的点映射到Poincaré球模型
        
        参数:
            x: 欧几里得空间的特征 [batch_size, dim]
        
        返回:
            y: Poincaré球模型中的点 [batch_size, dim]
        """
        # 计算x的范数
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        
        # 避免除零和数值不稳定
        x_norm = torch.clamp(x_norm, min=1e-10)
        
        # 计算sqrt(c)
        sqrt_c = torch.sqrt(torch.tensor(self.c, device=x.device, dtype=x.dtype))
        
        # 计算tanh项
        tanh_term = torch.tanh(sqrt_c * x_norm) / (sqrt_c * x_norm)
        
        # 指数映射公式
        y = tanh_term * x
        
        # 裁剪到球内，防止数值不稳定
        y_norm = torch.norm(y, dim=-1, keepdim=True)
        if self.clip_r < 1.0:
            scale = torch.clamp(self.clip_r / (y_norm + 1e-10), max=1.0)
            y = y * scale
        
        # 额外检查：确保范数严格小于1（双曲空间约束）
        y_norm_final = torch.norm(y, dim=-1, keepdim=True)
        if torch.any(y_norm_final >= 1.0):
            # 如果还有超出范围的，强制裁剪到0.99
            scale_final = torch.clamp(0.99 / (y_norm_final + 1e-10), max=1.0)
            y = y * scale_final
        
        # 检查NaN或Inf
        if torch.isnan(y).any() or torch.isinf(y).any():
            # 如果出现NaN或Inf，返回零向量
            y = torch.zeros_like(y)
        
        return y
    
    def forward(self, x):
        """
        前向传播：将输入特征投影到双曲空间
        
        参数:
            x: 输入特征 [batch_size, input_dim]
        
        返回:
            h: 双曲空间中的特征 [batch_size, output_dim]
        """
        # 线性投影
        x_proj = self.linear(x)
        
        # 投影到双曲空间
        h = self.expmap(x_proj)
        
        return h


def poincare_distance(x, y, c=1.0):
    """
    计算Poincaré球模型中两点之间的双曲距离
    
    参数:
        x: 第一个点 [batch_size, dim] 或 [dim]
        y: 第二个点 [batch_size, dim] 或 [dim]
        c: 曲率参数
    
    返回:
        distance: 双曲距离 [batch_size] 或标量
    """
    # 确保x和y在同一设备上
    if x.device != y.device:
        y = y.to(x.device)
    
    # 计算欧几里得范数
    x_norm_sq = torch.sum(x ** 2, dim=-1)
    y_norm_sq = torch.sum(y ** 2, dim=-1)
    
    # 计算差值
    diff = x - y
    diff_norm_sq = torch.sum(diff ** 2, dim=-1)
    
    # 避免数值不稳定
    eps = 1e-10
    x_norm_sq = torch.clamp(x_norm_sq, max=1.0 - eps)
    y_norm_sq = torch.clamp(y_norm_sq, max=1.0 - eps)
    
    # 计算双曲距离
    sqrt_c = torch.sqrt(torch.tensor(c, device=x.device, dtype=x.dtype))
    
    # Poincaré距离公式
    numerator = diff_norm_sq
    denominator = (1 - x_norm_sq) * (1 - y_norm_sq)
    denominator = torch.clamp(denominator, min=eps)
    
    arcosh_arg = 1 + 2 * numerator / denominator
    arcosh_arg = torch.clamp(arcosh_arg, min=1.0 + eps)  # 确保 >= 1
    
    # 使用数值稳定的arccosh计算
    distance = (1.0 / sqrt_c) * torch.acosh(arcosh_arg)
    
    return distance


def hyperbolic_similarity(x, y, c=1.0, temperature=1.0):
    """
    计算双曲相似度（使用双曲距离的负值，并应用温度缩放）
    
    参数:
        x: 第一个点 [batch_size, dim] 或 [dim]
        y: 第二个点 [batch_size, dim] 或 [dim]
        c: 曲率参数
        temperature: 温度参数，用于缩放相似度
    
    返回:
        similarity: 双曲相似度 [batch_size] 或标量
    """
    # 计算双曲距离
    distance = poincare_distance(x, y, c=c)
    
    # 将距离转换为相似度（距离越小，相似度越大）
    similarity = -distance / temperature
    
    return similarity


def hyperbolic_similarity_matrix(audio_embeds, text_embeds, c=1.0, temperature=1.0):
    """
    计算音频嵌入和文本嵌入之间的双曲相似度矩阵
    
    参数:
        audio_embeds: 音频嵌入 [num_audio, dim]
        text_embeds: 文本嵌入 [num_text, dim]
        c: 曲率参数
        temperature: 温度参数
    
    返回:
        similarity_matrix: 相似度矩阵 [num_audio, num_text]
    """
    num_audio = audio_embeds.shape[0]
    num_text = text_embeds.shape[0]
    
    # 初始化相似度矩阵
    similarity_matrix = torch.zeros(num_audio, num_text, device=audio_embeds.device)
    
    # 计算每对音频-文本的相似度
    for i in range(num_audio):
        for j in range(num_text):
            similarity_matrix[i, j] = hyperbolic_similarity(
                audio_embeds[i], text_embeds[j], c=c, temperature=temperature
            )
    
    return similarity_matrix


def hyperbolic_similarity_matrix_batch(audio_embeds, text_embeds, c=1.0, temperature=1.0):
    """
    批量计算音频嵌入和文本嵌入之间的双曲相似度矩阵（优化版本）
    
    参数:
        audio_embeds: 音频嵌入 [num_audio, dim]
        text_embeds: 文本嵌入 [num_text, dim]
        c: 曲率参数
        temperature: 温度参数
    
    返回:
        similarity_matrix: 相似度矩阵 [num_audio, num_text]
    """
    device = audio_embeds.device
    dtype = audio_embeds.dtype
    
    # 扩展维度以便批量计算
    # audio_embeds: [num_audio, 1, dim]
    # text_embeds: [1, num_text, dim]
    audio_expanded = audio_embeds.unsqueeze(1)  # [num_audio, 1, dim]
    text_expanded = text_embeds.unsqueeze(0)     # [1, num_text, dim]
    
    # 计算范数
    audio_norm_sq = torch.sum(audio_embeds ** 2, dim=-1, keepdim=True)  # [num_audio, 1]
    text_norm_sq = torch.sum(text_embeds ** 2, dim=-1, keepdim=True).t()  # [1, num_text]
    
    # 计算差值
    diff = audio_expanded - text_expanded  # [num_audio, num_text, dim]
    diff_norm_sq = torch.sum(diff ** 2, dim=-1)  # [num_audio, num_text]
    
    # 避免数值不稳定
    eps = 1e-10
    audio_norm_sq = torch.clamp(audio_norm_sq, max=1.0 - eps)
    text_norm_sq = torch.clamp(text_norm_sq, max=1.0 - eps)
    
    # 计算双曲距离
    sqrt_c = torch.sqrt(torch.tensor(c, device=device, dtype=dtype))
    
    # Poincaré距离公式
    numerator = diff_norm_sq
    denominator = (1 - audio_norm_sq) * (1 - text_norm_sq)
    denominator = torch.clamp(denominator, min=eps)
    
    arcosh_arg = 1 + 2 * numerator / denominator
    arcosh_arg = torch.clamp(arcosh_arg, min=1.0 + eps)
    
    # 计算双曲距离
    distance = (1.0 / sqrt_c) * torch.acosh(arcosh_arg)
    
    # 转换为相似度
    similarity = -distance / temperature
    
    return similarity


class HyperbolicContrastiveLoss(nn.Module):
    """
    双曲空间中的对比损失函数
    使用双曲相似度替代cosine相似度
    """
    def __init__(self, c=1.0, temperature=1.0):
        super(HyperbolicContrastiveLoss, self).__init__()
        self.c = c
        self.temperature = temperature
    
    def forward(self, audio_features, text_features, labels=None):
        """
        计算双曲对比损失
        
        参数:
            audio_features: 音频特征 [batch_size, dim]
            text_features: 文本特征 [batch_size, dim]
            labels: 标签，如果为None，则假设第i个音频对应第i个文本
        
        返回:
            loss: 对比损失
        """
        batch_size = audio_features.shape[0]
        
        if labels is None:
            # 标准情况：第i个音频对应第i个文本
            labels = torch.arange(batch_size, device=audio_features.device)
        
        # 计算相似度矩阵
        similarity_matrix = hyperbolic_similarity_matrix_batch(
            audio_features, text_features, c=self.c, temperature=self.temperature
        )
        
        # 计算logits（应用温度缩放）
        logits_per_audio = similarity_matrix
        logits_per_text = similarity_matrix.t()
        
        # 计算交叉熵损失
        loss_audio = F.cross_entropy(logits_per_audio, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        
        # 总损失
        total_loss = (loss_audio + loss_text) / 2
        
        return total_loss


def pairwise_poincare_distance(x, y, c=1.0):
    """
    Compute pairwise Poincaré distance between two sets of hyperbolic embeddings.

    Args:
        x: [N, D], hyperbolic embeddings (||x|| < 1)
        y: [M, D], hyperbolic embeddings (||y|| < 1)
        c: curvature

    Returns:
        dist: [N, M]
    """
    device = x.device
    dtype = x.dtype

    x_norm_sq = torch.sum(x ** 2, dim=-1, keepdim=True)          # [N, 1]
    y_norm_sq = torch.sum(y ** 2, dim=-1, keepdim=True).t()     # [1, M]

    diff = x.unsqueeze(1) - y.unsqueeze(0)                      # [N, M, D]
    diff_norm_sq = torch.sum(diff ** 2, dim=-1)                 # [N, M]

    eps = 1e-9
    x_norm_sq = torch.clamp(x_norm_sq, max=1.0 - eps)
    y_norm_sq = torch.clamp(y_norm_sq, max=1.0 - eps)

    denom = (1.0 - x_norm_sq) * (1.0 - y_norm_sq)
    denom = torch.clamp(denom, min=eps)

    z = 1.0 + 2.0 * diff_norm_sq / denom
    z = torch.clamp(z, min=1.0 + eps)

    sqrt_c = torch.sqrt(torch.tensor(c, device=device, dtype=dtype))

    return torch.acosh(z) / sqrt_c


# ============================================================
# Hyperbolic Top-K Triplet Loss (Same-modality hierarchy loss)
# ============================================================

class HyperbolicTopKTripletLoss(nn.Module):
    """
    Hyperbolic triplet loss using top-k nearest neighbors as positives.
    This loss is used to regularize hierarchy / clustering structure
    inside hyperbolic space.
    """
    def __init__(self, c=1.0, margin=0.1, top_k=5):
        super().__init__()
        self.c = c
        self.margin = margin
        self.top_k = top_k

    def forward(self, embeds, neg_samples=20, tau=0.1):
        B = embeds.size(0)
        device = embeds.device

        dist = pairwise_poincare_distance(embeds, embeds, c=self.c)
        dist.fill_diagonal_(float("inf"))

        k = min(self.top_k, B - 1)
        pos_dist, pos_idx = torch.topk(dist, k=k, largest=False, dim=1)

        # positives: log-sum-exp
        pos_term = torch.logsumexp(-pos_dist / tau, dim=1)  # [B]

        # mask out top-k
        mask = torch.ones_like(dist, dtype=torch.bool)
        mask.scatter_(1, pos_idx, False)

        neg_terms = []
        for i in range(B):
            neg_pool = dist[i][mask[i]]
            idx = torch.randint(0, neg_pool.size(0), (neg_samples,), device=device)
            neg_terms.append(
                torch.logsumexp(-neg_pool[idx] / tau, dim=0)
            )

        neg_term = torch.stack(neg_terms)

        loss = -(pos_term - neg_term)
        return loss.mean()


# ============================================================
# Stage-2 Layered Hyperbolic Loss
# ============================================================

class HyperbolicLayeredLoss(nn.Module):
    """
    Stage-2 loss for hyperbolic CLAP:

        L = L_cross_modal
            + lambda_h * (L_text_hierarchy + L_audio_hierarchy)

    - L_cross_modal: standard hyperbolic contrastive loss (same as stage 1)
    - L_*_hierarchy: same-modality hyperbolic top-k triplet loss
    """
    def __init__(
        self,
        c=1.0,
        temperature=1.0,
        top_k=5,
        margin=0.1,
        lambda_h=0.01,
    ):
        super().__init__()

        self.lambda_h = lambda_h

        # Cross-modal alignment loss (do NOT change from stage 1)
        self.cross_modal_loss = HyperbolicContrastiveLoss(
            c=c,
            temperature=temperature
        )

        # Same-modality hierarchy regularization
        self.text_hierarchy_loss = HyperbolicTopKTripletLoss(
            c=c,
            margin=margin,
            top_k=top_k
        )

        self.audio_hierarchy_loss = HyperbolicTopKTripletLoss(
            c=c,
            margin=margin,
            top_k=top_k
        )

    def forward(self, audio_embeds, text_embeds, labels=None):
        """
        Args:
            audio_embeds: [B, D] hyperbolic audio embeddings
            text_embeds:  [B, D] hyperbolic text embeddings
            labels: optional labels for cross-modal loss

        Returns:
            total_loss: scalar
        """
        # 1. Cross-modal alignment (anchor term)
        loss_cross = self.cross_modal_loss(
            audio_embeds, text_embeds, labels
        )

        # 2. Hierarchy regularization (same-modality)
        loss_text_h = self.text_hierarchy_loss(text_embeds)
        loss_audio_h = self.audio_hierarchy_loss(audio_embeds)

        # 3. Total loss
        total_loss = self.lambda_h * (loss_text_h + loss_audio_h)

        return total_loss