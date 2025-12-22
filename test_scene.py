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
# [修正] 确保从生成渐变噪声的脚本中导入 loader
from data_loader_noisy_scene import loader_cl_noise

# 设置环境
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = "1"

# --- 参数设置 ---
parser = argparse.ArgumentParser(description='Quality-Aware Scene15 Training with Strict Alignment')

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

# 兼容性占位参数 (loader 需要)
parser.add_argument('--neg-prop', default=0, type=int)
parser.add_argument('--aligned-prop', default=1.0, type=float)
parser.add_argument('--complete-prop', default=1.0, type=float)
parser.add_argument('--noisy-training', default=False, type=bool)

args = parser.parse_args()

# 设置 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 1. 质量门控注意力模块 (Quality-Gated Attention) ---
class QualityGatedAttention(nn.Module):
    def __init__(self, feature_dim=512):
        super(QualityGatedAttention, self).__init__()
        self.scale = feature_dim ** -0.5
        
        # 两个视图的交互层
        self.w_q1 = nn.Linear(feature_dim, feature_dim)
        self.w_k1 = nn.Linear(feature_dim, feature_dim)
        self.w_v1 = nn.Linear(feature_dim, feature_dim)

        self.w_q2 = nn.Linear(feature_dim, feature_dim)
        self.w_k2 = nn.Linear(feature_dim, feature_dim)
        self.w_v2 = nn.Linear(feature_dim, feature_dim)
        
        self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(self, h0, h1, q0, q1):
        """
        h0, h1: Feature embeddings (B, D)
        q0, q1: Quality scores (B, 1), range [0, 1]. High means Good.
        """
        # --- View 1 Enhancement (Guided by View 2) ---
        # Query from V1, Key/Value from V2
        query1 = self.w_q1(h0)
        key1   = self.w_k1(h1)
        value1 = self.w_v1(h1)
        
        attn_scores1 = torch.matmul(query1, key1.transpose(-2, -1)) * self.scale
        attn_probs1 = F.softmax(attn_scores1, dim=-1) # (B, B)
        
        context_from_v2 = torch.matmul(attn_probs1, value1) # (B, D)
        
        # [Gating]: 只有当 View 2 质量高时，才允许它的信息流入 View 1
        gated_context_v2 = context_from_v2 * q1 
        
        # --- View 2 Enhancement (Guided by View 1) ---
        query2 = self.w_q2(h1)
        key2   = self.w_k2(h0)
        value2 = self.w_v2(h0)
        
        attn_scores2 = torch.matmul(query2, key2.transpose(-2, -1)) * self.scale
        attn_probs2 = F.softmax(attn_scores2, dim=-1)
        
        context_from_v1 = torch.matmul(attn_probs2, value2)
        
        # [Gating]: 只有当 View 1 质量高时，才允许它的信息流入 View 2
        gated_context_v1 = context_from_v1 * q0

        # Residual Connection & Norm
        z0_final = self.layer_norm(h0 + gated_context_v2)
        z1_final = self.layer_norm(h1 + gated_context_v1)
        
        return z0_final, z1_final

# --- 2. 损失函数定义 ---

class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()
    
    def forward(self, input, target, quality_weights):
        w = quality_weights.view(-1, 1)
        loss = (input - target) ** 2
        loss = loss * w  # 始终加权：不强求重建噪声
        return loss.mean()

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
        
        if w0 is not None and w1 is not None:
            # Joint weight: w0 * w1
            w_pair = (w0 * w1)
            l_i2t = l_i2t * w_pair
            l_t2i = l_t2i * w_pair
            
        return l_i2t.mean() + l_t2i.mean()

# --- 3. 训练核心函数 ---
def train_one_epoch(train_loader, models, criterions, optimizer, epoch, args, quality_tensor_v1, quality_tensor_v2):
    backbone, attention_mod = models
    criterion_mse, criterion_ccl = criterions
    
    backbone.train()
    attention_mod.train()
    
    total_loss = 0
    start_time = time.time()
    
    # Phase Switching
    use_attention = (epoch >= args.warmup_epochs)
    
    for batch_idx, (x0, x1, labels, _, _, _, indices) in enumerate(train_loader):
        x0, x1 = x0.to(device), x1.to(device)
        indices = indices.to(device)
        
        # Retrieve Quality Scores (B, 1) [1.0=Clean, 0.0=Noisy]
        batch_q1 = (1.0 - quality_tensor_v1[indices]).view(-1, 1)
        batch_q2 = (1.0 - quality_tensor_v2[indices]).view(-1, 1)
        
        x0_flat = x0.view(x0.size(0), -1)
        x1_flat = x1.view(x1.size(0), -1)
        
        # Backbone Forward
        h0, h1, z0, z1 = backbone(x0_flat, x1_flat)
        
        # Loss 1: Reconstruction 
        # [策略]: 始终加权。避免网络为了降低 Loss 而去记忆高斯噪声。
        loss_mse = criterion_mse(x0_flat, z0, batch_q1) + criterion_mse(x1_flat, z1, batch_q2)
        
        # Loss 2: Contrastive
        if use_attention:
            # --- Phase 2: Strict Alignment with Gated Attention ---
            # 1. 质量门控 Attention: 干净视图修复脏视图，脏视图不污染干净视图
            h0_final, h1_final = attention_mod(h0, h1, batch_q1, batch_q2)
            
            # 2. [关键策略]: Strict Contrastive Loss
            # 不传入权重 (w0=None, w1=None)。
            # 强制要求修复后的特征必须在隐空间对齐。
            # 这会倒逼 Attention 模块必须起作用，否则 Loss 降不下去。
            loss_contrast = criterion_ccl(h0_final, h1_final, w0=None, w1=None)
            
        else:
            # --- Phase 1: Warmup (Filtered Training) ---
            h0_final, h1_final = h0, h1
            
            # [策略]: 加权对比。
            # 在没有 Attention 修复之前，脏样本无法对齐，必须忽略它们，防止梯度污染。
            loss_contrast = criterion_ccl(h0_final, h1_final, w0=batch_q1.view(-1), w1=batch_q2.view(-1))

        loss = loss_mse + (args.lamda * loss_contrast)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    epoch_time = time.time() - start_time
    phase_str = "Phase 2 (Strict Attn)" if use_attention else "Phase 1 (Warmup)"
    
    if epoch % args.log_interval == 0:
         logging.info(f"Epoch [{epoch}/{args.epochs}] [{phase_str}] Time: {epoch_time:.2f}s | Loss: {total_loss / len(train_loader):.4f}")

# --- 主程序 ---
def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')
    print(f"==========\nSetting: GPU={args.gpu}, Lamda={args.lamda}, Epochs={args.epochs} (Warmup={args.warmup_epochs})\n==========")

    # Seed
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Load Quality Weights
    if not os.path.exists('quality_weights.npz'):
        raise FileNotFoundError("Run 'generate_quality_weights.py' first!")
    
    q_data = np.load('quality_weights.npz')
    quality_tensor_v1 = torch.from_numpy(q_data['noise_score_v1']).float().to(device)
    quality_tensor_v2 = torch.from_numpy(q_data['noise_score_v2']).float().to(device)
    print(f"Loaded Quality Weights. Shapes: {quality_tensor_v1.shape}")

    # Load Data
    train_loader, all_loader, _ = loader_cl_noise(
        args.batch_size, args.data_name, args.seed
    )

    # Models
    backbone = SUREfcScene().to(device)
    attention_mod = QualityGatedAttention(feature_dim=512).to(device)
    
    optimizer = torch.optim.Adam(
        list(backbone.parameters()) + list(attention_mod.parameters()), 
        lr=args.lr
    )
    
    criterion_mse = WeightedMSELoss().to(device)
    criterion_ccl = WeightedCCL(tau=args.tau)

    print("Start Training (Strict Quality-Gated Framework)...")
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
        
        # Eval
        if epoch == args.epochs - 1 or (epoch + 1) % 10 == 0:
            backbone.eval()
            attention_mod.eval()
            use_attention = (epoch >= args.warmup_epochs)
            
            feat_v0_list, feat_v1_list, gt_list = [], [], []
            
            with torch.no_grad():
                for x0, x1, labels, _, _, _, indices in all_loader:
                    x0, x1 = x0.to(device), x1.to(device)
                    indices = indices.to(device)
                    
                    # Inference时也需要质量分数来控制Attention
                    batch_q1 = (1.0 - quality_tensor_v1[indices]).view(-1, 1)
                    batch_q2 = (1.0 - quality_tensor_v2[indices]).view(-1, 1)
                    
                    x0_flat = x0.view(x0.size(0), -1)
                    x1_flat = x1.view(x1.size(0), -1)
                    
                    h0, h1, _, _ = backbone(x0_flat, x1_flat)
                    
                    if use_attention:
                        h0, h1 = attention_mod(h0, h1, batch_q1, batch_q2)
                        
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
            
            logging.info(f"Epoch {epoch} Result: ACC={acc:.4f}, NMI={nmi:.4f} (Best: {best_acc:.4f})")

    print(f"Final Best ACC: {best_acc:.4f}")

if __name__ == '__main__':
    main()