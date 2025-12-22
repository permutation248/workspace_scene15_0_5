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

# --- 参数设置 (已更新为最佳实践默认值) ---
parser = argparse.ArgumentParser(description='Quality-Aware Scene15 Training')

# 核心训练参数
parser.add_argument('--gpu', default='0', type=str, help='GPU device idx')
# 建议默认使用 400 轮配合 Scheduler
parser.add_argument('--epochs', default=400, type=int, help='Total training epochs')
parser.add_argument('--warmup-epochs', default=50, type=int, help='Epochs for Phase 1')
# 建议保持 Mini-batch 128 以获得更多更新步数
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

# --- 1. 跨视图注意力模块 ---
class CrossViewSemanticEnhancement(nn.Module):
    def __init__(self, feature_dim=512):
        super(CrossViewSemanticEnhancement, self).__init__()
        self.scale = feature_dim ** -0.5
        
        self.w_q1 = nn.Linear(feature_dim, feature_dim)
        self.w_k1 = nn.Linear(feature_dim, feature_dim)
        self.w_v1 = nn.Linear(feature_dim, feature_dim)

        self.w_q2 = nn.Linear(feature_dim, feature_dim)
        self.w_k2 = nn.Linear(feature_dim, feature_dim)
        self.w_v2 = nn.Linear(feature_dim, feature_dim)
        
        self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(self, h0, h1):
        # View 1 增强
        q1 = self.w_q1(h0)
        k1 = self.w_k1(h1)
        v1 = self.w_v1(h1)
        
        attn_score1 = torch.matmul(q1, k1.transpose(-2, -1)) * self.scale
        attn_prob1 = F.softmax(attn_score1, dim=-1)
        h0_enhanced = torch.matmul(attn_prob1, v1)
        
        # View 2 增强
        q2 = self.w_q2(h1)
        k2 = self.w_k2(h0)
        v2 = self.w_v2(h0)
        
        attn_score2 = torch.matmul(q2, k2.transpose(-2, -1)) * self.scale
        attn_prob2 = F.softmax(attn_score2, dim=-1)
        h1_enhanced = torch.matmul(attn_prob2, v2)
        
        z0_final = self.layer_norm(h0 + h0_enhanced)
        z1_final = self.layer_norm(h1 + h1_enhanced)
        
        return z0_final, z1_final

# --- 2. 损失函数定义 ---

# 修改后的重建损失：支持加权或不加权(Full)
class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()
    
    def forward(self, input, target, quality_weights=None):
        # 基础 MSE
        loss = (input - target) ** 2
        
        # 如果提供了权重，则进行加权；否则计算完整平均值
        if quality_weights is not None:
            w = quality_weights.view(-1, 1)
            loss = loss * w
            
        return loss.mean()

# 加权对比损失
class WeightedCCL(nn.Module):
    def __init__(self, tau=0.5):
        super(WeightedCCL, self).__init__()
        self.tau = tau
        self.device = device
    
    def forward(self, x0, x1, w0=None, w1=None):
        x0 = F.normalize(x0, dim=1)
        x1 = F.normalize(x1, dim=1)
        
        out = torch.mm(x0, x1.t()) / self.tau
        labels = torch.arange(out.size(0)).to(self.device)
        
        loss_func = nn.CrossEntropyLoss(reduction='none')
        l_i2t = loss_func(out, labels)
        l_t2i = loss_func(out.t(), labels)
        
        # 如果提供了权重，则加权；否则就是标准 InfoNCE
        if w0 is not None and w1 is not None:
            w_pair = (w0 * w1)
            l_i2t = l_i2t * w_pair
            l_t2i = l_t2i * w_pair
            
        return l_i2t.mean() + l_t2i.mean()

# --- 3. 训练核心函数 (关键逻辑修改) ---
def train_one_epoch(train_loader, models, criterions, optimizer, epoch, args, quality_tensor_v1, quality_tensor_v2):
    backbone, attention_mod = models
    criterion_mse, criterion_ccl = criterions
    
    backbone.train()
    if attention_mod: attention_mod.train()
    
    total_loss = 0
    start_time = time.time()
    
    use_attention = (epoch >= args.warmup_epochs)
    
    for batch_idx, (x0, x1, labels, _, _, _, indices) in enumerate(train_loader):
        x0, x1 = x0.to(device), x1.to(device)
        indices = indices.to(device)
        
        # 获取质量权重
        batch_q1 = 1.0 - quality_tensor_v1[indices]
        batch_q2 = 1.0 - quality_tensor_v2[indices]
        
        x0_flat = x0.view(x0.size(0), -1)
        x1_flat = x1.view(x1.size(0), -1)
        
        # Backbone Forward
        h0, h1, z0, z1 = backbone(x0_flat, x1_flat)
        
        # --- 核心逻辑分支 ---
        
        if not use_attention:
            # === Phase 1: Warmup ===
            # 1. 对比损失: 【加权】 (Weighted Contrastive)
            #    逻辑: 初始特征含噪，需利用质量权重避开噪声样本
            h0_final, h1_final = h0, h1
            loss_contrast = criterion_ccl(h0_final, h1_final, w0=batch_q1, w1=batch_q2)
            
            # 2. 重建损失: 【完整/不加权】 (Full Reconstruction)
            #    逻辑: 初始阶段让 AE 学习完整数据分布，包括噪声
            loss_mse = criterion_mse(x0_flat, z0, quality_weights=None) + \
                       criterion_mse(x1_flat, z1, quality_weights=None)
            
            loss = loss_mse + (0.1 * loss_contrast)
            
        else:
            # === Phase 2: Alignment & Enhancement ===
            # 1. 特征增强
            h0_final, h1_final = attention_mod(h0, h1)
            
            # 2. 对比损失: 【完整/不加权】 (Full Contrastive)
            #    逻辑: 特征已通过 Attention 增强去噪，信任其进行标准对齐
            loss_contrast = criterion_ccl(h0_final, h1_final, w0=None, w1=None)
            
            # 3. 重建损失: 【加权】 (Weighted Reconstruction)
            #    逻辑: 后期微调时，不再强求拟合噪声，只重建可信部分
            loss_mse = criterion_mse(x0_flat, z0, quality_weights=batch_q1) + \
                       criterion_mse(x1_flat, z1, quality_weights=batch_q2)

            loss = loss_mse + (1.0 * loss_contrast)
        
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
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')
    print(f"==========\nSetting: GPU={args.gpu}, Lamda={args.lamda}, Epochs={args.epochs} (Warmup={args.warmup_epochs})\n==========")
    print(f"Strategy: Phase 1 [Weighted CCL + Full MSE] -> Phase 2 [Full CCL + Weighted MSE]")

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # 加载质量矩阵
    if not os.path.exists('quality_weights.npz'):
        raise FileNotFoundError("Run 'generate_quality_weights.py' first!")
    
    q_data = np.load('quality_weights.npz')
    quality_tensor_v1 = torch.from_numpy(q_data['noise_score_v1']).float().to(device)
    quality_tensor_v2 = torch.from_numpy(q_data['noise_score_v2']).float().to(device)
    print(f"Loaded Quality Weights. Shapes: {quality_tensor_v1.shape}, {quality_tensor_v2.shape}")

    # 加载数据
    train_loader, all_loader, _ = loader_cl_noise(
        args.batch_size, args.data_name, args.seed
    )

    # 初始化模型
    backbone = SUREfcScene().to(device)
    attention_mod = CrossViewSemanticEnhancement(feature_dim=512).to(device)
    
    # 优化器与调度器 (包含 Cosine Annealing)
    params = list(backbone.parameters()) + list(attention_mod.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # 损失函数
    criterion_mse = WeightedMSELoss().to(device)
    criterion_ccl = WeightedCCL(tau=args.tau)

    print("Start Training...")
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
        
        # 更新学习率
        scheduler.step()
        
        # 评估
        if epoch == args.epochs - 1 or (epoch + 1) % 10 == 0:
            backbone.eval()
            attention_mod.eval()
            
            use_attention = (epoch >= args.warmup_epochs)
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
            
            data = [np.concatenate(feat_v0_list), np.concatenate(feat_v1_list)]
            gt_label = np.concatenate(gt_list)
            
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