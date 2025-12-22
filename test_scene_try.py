import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
import sys
import numpy as np
import warnings

# 引入必要的模块
from models import SUREfcScene
from Clustering import Clustering
from sure_inference import both_infer
from data_loader_noisy_scene import loader_cl_noise
# 设置环境
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = "1"

# --- 参数设置 ---
parser = argparse.ArgumentParser(description='Quality-Aware Scene15 Training')

# 核心训练参数
parser.add_argument('--gpu', default='0', type=str, help='GPU device idx')
parser.add_argument('--epochs', default=200, type=int, help='Total training epochs')
parser.add_argument('--warmup-epochs', default=50, type=int, help='Epochs for Phase 1 (Weighted training without Attention)')
parser.add_argument('--batch-size', default=128, type=int, help='Batch size')
parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
parser.add_argument('--lamda', default=0.1, type=float, help='Weight for Contrastive Loss')

# 固定/默认参数
parser.add_argument('--tau', default=0.5, type=float, help='Temperature parameter')
parser.add_argument('--seed', default=1111, type=int, help='Random seed')
parser.add_argument('--log-interval', default=50, type=int, help='Log interval')
parser.add_argument('--data-name', default='Scene15', type=str, help='Dataset name')

# 兼容性占位参数
parser.add_argument('--neg-prop', default=0, type=int)
parser.add_argument('--aligned-prop', default=1.0, type=float)
parser.add_argument('--complete-prop', default=1.0, type=float)
parser.add_argument('--noisy-training', default=False, type=bool)

args = parser.parse_args()

# 设置 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 1. 跨视图注意力模块 (Phase 2 使用) ---
class CrossViewSemanticEnhancement(nn.Module):
    def __init__(self, feature_dim=512):
        super(CrossViewSemanticEnhancement, self).__init__()
        # 简单的线性变换用于 Attention 的 Q, K, V
        # 假设输入特征已经 normalize 过
        self.scale = feature_dim ** -0.5
        
        # 两个视图的交互层
        self.w_q1 = nn.Linear(feature_dim, feature_dim)
        self.w_k1 = nn.Linear(feature_dim, feature_dim)
        self.w_v1 = nn.Linear(feature_dim, feature_dim)

        self.w_q2 = nn.Linear(feature_dim, feature_dim)
        self.w_k2 = nn.Linear(feature_dim, feature_dim)
        self.w_v2 = nn.Linear(feature_dim, feature_dim)
        
        # 融合门控 (可选，这里简化为残差连接)
        self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(self, h0, h1):
        # h0: (B, D), h1: (B, D)
        
        # View 1 增强: Query=h0, Key=h1, Value=h1 (用 View 2 增强 View 1)
        q1 = self.w_q1(h0)
        k1 = self.w_k1(h1)
        v1 = self.w_v1(h1)
        
        # Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V
        attn_score1 = torch.matmul(q1, k1.transpose(-2, -1)) * self.scale
        attn_prob1 = F.softmax(attn_score1, dim=-1)
        h0_enhanced = torch.matmul(attn_prob1, v1)
        
        # View 2 增强: Query=h1, Key=h0, Value=h0 (用 View 1 增强 View 2)
        q2 = self.w_q2(h1)
        k2 = self.w_k2(h0)
        v2 = self.w_v2(h0)
        
        attn_score2 = torch.matmul(q2, k2.transpose(-2, -1)) * self.scale
        attn_prob2 = F.softmax(attn_score2, dim=-1)
        h1_enhanced = torch.matmul(attn_prob2, v2)
        
        # 残差连接 + Norm: 混合原始特征与增强特征
        # 这样即使 Attention 学不好，也不会破坏原始信息
        z0_final = self.layer_norm(h0 + h0_enhanced)
        z1_final = self.layer_norm(h1 + h1_enhanced)
        
        return z0_final, z1_final

# --- 2. 损失函数定义 ---

# 加权重建损失 (Quality Weighted MSE)
class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()
    
    def forward(self, input, target, quality_weights):
        # quality_weights: (B, ) -> 扩展为 (B, 1) 以广播
        w = quality_weights.view(-1, 1)
        # 计算逐元素平方差
        loss = (input - target) ** 2
        # 加权：质量低(w小)的样本，重建损失权重降低
        loss = loss * w
        return loss.mean()

# 加权对比损失 (Weighted CCL / InfoNCE)
class WeightedCCL(nn.Module):
    def __init__(self, tau=0.5):
        super(WeightedCCL, self).__init__()
        self.tau = tau
        self.device = device
    
    def forward(self, x0, x1, w0=None, w1=None):
        # 归一化
        x0 = F.normalize(x0, dim=1)
        x1 = F.normalize(x1, dim=1)
        
        # 相似度矩阵
        out = torch.mm(x0, x1.t()) / self.tau
        
        # 目标标签
        labels = torch.arange(out.size(0)).to(self.device)
        
        # 计算标准 CrossEntropy (reduction='none' 保留每个样本的 loss)
        loss_func = nn.CrossEntropyLoss(reduction='none')
        l_i2t = loss_func(out, labels)
        l_t2i = loss_func(out.t(), labels)
        
        # 如果提供了权重，则进行加权
        # 逻辑：两个视图的联合质量决定了这对样本在对比学习中的重要性
        # Pair Quality = Q_view1 * Q_view2
        if w0 is not None and w1 is not None:
            # 权重归一化 (可选，防止 loss 数值过小)
            # w_pair = (w0 * w1).detach() 
            w_pair = (w0 * w1) # 保持梯度(虽然weight通常不需要梯度)
            
            l_i2t = l_i2t * w_pair
            l_t2i = l_t2i * w_pair
            
        return l_i2t.mean() + l_t2i.mean()

# --- 3. 训练核心函数 ---
def train_one_epoch(train_loader, models, criterions, optimizer, epoch, args, quality_tensor_v1, quality_tensor_v2):
    backbone, attention_mod = models
    criterion_mse, criterion_ccl = criterions
    
    backbone.train()
    if attention_mod: attention_mod.train()
    
    total_loss = 0
    start_time = time.time()
    
    # 判断当前阶段
    # Phase 1 (Warmup): No Attention, Weighted Contrastive, Weighted Recon
    # Phase 2 (Enhancement): With Attention, Normal Contrastive, Weighted Recon
    use_attention = (epoch >= args.warmup_epochs)
    
    for batch_idx, (x0, x1, labels, _, _, _, indices) in enumerate(train_loader):
        x0, x1 = x0.to(device), x1.to(device)
        indices = indices.to(device)
        
        # 获取当前 Batch 对应的质量权重 (Quality = 1 - NoiseScore)
        # indices 是 loader 返回的样本全局索引
        batch_q1 = 1.0 - quality_tensor_v1[indices]
        batch_q2 = 1.0 - quality_tensor_v2[indices]
        
        # Flatten
        x0_flat = x0.view(x0.size(0), -1)
        x1_flat = x1.view(x1.size(0), -1)
        
        # Backbone Forward
        # model返回: h0, h1 (hidden), z0, z1 (recon)
        h0, h1, z0, z1 = backbone(x0_flat, x1_flat)
        
        # --- Loss 1: Weighted Reconstruction Loss (始终使用加权) ---
        # 质量越低，权重越低，模型不需要强行去拟合噪声
        loss_mse = criterion_mse(x0_flat, z0, batch_q1) + criterion_mse(x1_flat, z1, batch_q2)
        
        # --- Feature Enhancement & Contrastive Loss ---
        if use_attention:
            # Phase 2: 使用 Attention 增强特征
            h0_final, h1_final = attention_mod(h0, h1)
            
            # Phase 2 Contrastive: 使用 "Normal Loss" (权重设为 1.0 或 None)
            # 依据 Prompt: "这个阶段用正常损失"
            loss_contrast = criterion_ccl(h0_final, h1_final, w0=None, w1=None)

            loss = loss_mse + (1.0 * loss_contrast)
            
        else:
            # Phase 1: 直接使用 Backbone 特征
            h0_final, h1_final = h0, h1
            
            # Phase 1 Contrastive: 使用 "Weighted Loss"
            # 依据 Prompt: "首先通过质量感知带权重的对比损失...训练几轮"
            loss_contrast = criterion_ccl(h0_final, h1_final, w0=batch_q1, w1=batch_q2)

            loss = loss_mse + (args.lamda * loss_contrast)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    epoch_time = time.time() - start_time
    phase_str = "Phase 2 (Attn)" if use_attention else "Phase 1 (Warmup)"
    
    if epoch % args.log_interval == 0:
         logging.info(f"Epoch [{epoch}/{args.epochs}] [{phase_str}] Time: {epoch_time:.2f}s | Loss: {total_loss / len(train_loader):.4f}")

# --- 主程序 ---
def main():
    # 1. 设置 Logger
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')
    print(f"==========\nSetting: GPU={args.gpu}, Lamda={args.lamda}, Epochs={args.epochs} (Warmup={args.warmup_epochs})\n==========")

    # 2. 设置随机种子
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # 3. 加载质量矩阵 (Noise Scores)
    if not os.path.exists('quality_weights.npz'):
        raise FileNotFoundError("Run 'generate_quality_weights.py' first!")
    
    q_data = np.load('quality_weights.npz')
    # 转为 Tensor 并放上 GPU，方便训练时查表
    # 注意：这里存的是 noise_score，训练时会转为 1-score
    quality_tensor_v1 = torch.from_numpy(q_data['noise_score_v1']).float().to(device)
    quality_tensor_v2 = torch.from_numpy(q_data['noise_score_v2']).float().to(device)
    print(f"Loaded Quality Weights. Shapes: {quality_tensor_v1.shape}, {quality_tensor_v2.shape}")

    # 4. 加载数据
    train_loader, all_loader, _ = loader_cl_noise(
        args.batch_size, args.data_name, args.seed
    )

    # 5. 初始化模型
    backbone = SUREfcScene().to(device)
    attention_mod = CrossViewSemanticEnhancement(feature_dim=512).to(device)
    
    # 联合优化器
    params = list(backbone.parameters()) + list(attention_mod.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # 损失函数
    criterion_mse = WeightedMSELoss().to(device)
    criterion_ccl = WeightedCCL(tau=args.tau)

    # 6. 训练循环
    print("Start Training (Quality-Aware Framework)...")
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        train_one_epoch(
            train_loader, 
            (backbone, attention_mod), 
            (criterion_mse, criterion_ccl), 
            optimizer, 
            epoch, 
            args,
            quality_tensor_v1, 
            quality_tensor_v2
        )

        scheduler.step()
        
        # 评估
        if epoch == args.epochs - 1 or (epoch + 1) % 10 == 0:
            # 推理时也需要考虑是否使用 Attention
            # 这里我们简单起见，始终使用 Backbone 提取基础特征进行聚类
            # 或者是使用增强后的特征？通常使用增强后的特征效果更好
            
            backbone.eval()
            attention_mod.eval()
            
            use_attention = (epoch >= args.warmup_epochs)
            
            # 使用 both_infer 的逻辑手动提取特征，因为我们需要处理 Attention
            # 复用 sure_inference.both_infer 的逻辑比较麻烦，直接写一个简单的推理循环
            feat_v0_list, feat_v1_list, gt_list = [], [], []
            
            with torch.no_grad():
                for x0, x1, labels, _, _, _, _ in all_loader:
                    x0, x1 = x0.to(device), x1.to(device)
                    x0_flat = x0.view(x0.size(0), -1)
                    x1_flat = x1.view(x1.size(0), -1)
                    
                    h0, h1, _, _ = backbone(x0_flat, x1_flat)
                    
                    if use_attention:
                        h0, h1 = attention_mod(h0, h1)
                        
                    feat_v0_list.append(h0.cpu().numpy())
                    feat_v1_list.append(h1.cpu().numpy())
                    gt_list.append(labels.cpu().numpy())
            
            # 拼接
            data = [np.concatenate(feat_v0_list), np.concatenate(feat_v1_list)]
            gt_label = np.concatenate(gt_list)
            
            # 聚类评估
            ret = Clustering(data, gt_label, random_state=args.seed)
            acc = ret['kmeans']['accuracy']
            nmi = ret['kmeans']['NMI']
            ari = ret['kmeans']['ARI']
            
            if acc > best_acc:
                best_acc = acc
            
            logging.info(f"Epoch {epoch} Result: ACC={acc:.4f}, NMI={nmi:.4f}, ARI={ari:.4f} (Best ACC: {best_acc:.4f})")

    print(f"Training Finished. Final Best ACC: {best_acc:.4f}")

if __name__ == '__main__':
    main()