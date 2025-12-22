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

# 引入你的模块
from Clustering import Clustering
from data_loader_noisy_scene import loader_cl_noise
# 注意：你需要把上面的 MomentumSURE 和 CrossViewSemanticEnhancement 保存到 models.py 或者直接放在这里
from models import MomentumSURE, CrossViewSemanticEnhancement 

warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = "1"

# --- 参数设置 ---
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--epochs', default=400, type=int)
parser.add_argument('--warmup-epochs', default=100, type=int) # 建议 100
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--lamda', default=0.1, type=float)
parser.add_argument('--tau', default=0.5, type=float)
parser.add_argument('--momentum', default=0.99, type=float, help='Momentum for target encoder')
parser.add_argument('--seed', default=1111, type=int)
parser.add_argument('--log-interval', default=50, type=int)
parser.add_argument('--data-name', default='Scene15', type=str)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 损失函数 ---
class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()
    def forward(self, input, target, quality_weights=None):
        loss = (input - target) ** 2
        if quality_weights is not None:
            loss = loss * quality_weights.view(-1, 1)
        return loss.mean()

class MomentumContrastiveLoss(nn.Module):
    def __init__(self, tau=0.5):
        super(MomentumContrastiveLoss, self).__init__()
        self.tau = tau
        self.device = device
    
    def forward(self, online_feat, target_feat, w_online=None, w_target=None):
        """
        计算 Online 特征与 Target 特征之间的对比损失
        """
        q = F.normalize(online_feat, dim=1)
        k = F.normalize(target_feat, dim=1) # Target 特征来自 Momentum Encoder
        
        # Sim(q, k)
        logits = torch.mm(q, k.t()) / self.tau
        labels = torch.arange(logits.size(0)).to(self.device)
        
        # 交叉熵
        loss = F.cross_entropy(logits, labels, reduction='none')
        
        # 加权逻辑 (Phase 1 使用)
        if w_online is not None and w_target is not None:
            w_pair = w_online * w_target
            loss = loss * w_pair
            
        return loss.mean()

# --- 训练逻辑 ---
def train_one_epoch(train_loader, models, criterions, optimizer, epoch, args, q_v1, q_v2):
    backbone, attention_mod = models
    criterion_mse, criterion_cl = criterions
    
    backbone.train() # Online Encoder 训练
    if attention_mod: attention_mod.train()
    
    total_loss = 0
    start_time = time.time()
    
    # 策略判断
    use_attention = (epoch >= args.warmup_epochs)
    
    for batch_idx, (x0, x1, labels, _, _, _, indices) in enumerate(train_loader):
        x0, x1 = x0.to(device), x1.to(device)
        indices = indices.to(device)
        batch_q1 = 1.0 - q_v1[indices]
        batch_q2 = 1.0 - q_v2[indices]
        
        x0_flat = x0.view(x0.size(0), -1)
        x1_flat = x1.view(x1.size(0), -1)
        
        # 1. Online Forward (带梯度)
        h0, h1, z0, z1 = backbone.forward_online(x0_flat, x1_flat)
        
        # 2. Target Forward (无梯度，用于对比学习的目标)
        with torch.no_grad():
            # 动量更新
            backbone._momentum_update_target_encoder()
            h0_t, h1_t = backbone.forward_target(x0_flat, x1_flat)

        # --- 渐进式信任策略实现 ---
        if not use_attention:
            # === Phase 1: Warmup ===
            # 重建: Full MSE
            l_rec = criterion_mse(x0_flat, z0, None) + criterion_mse(x1_flat, z1, None)
            
            # 对比: Online <-> Target (Weighted)
            # 逻辑: 让 Online 慢慢靠近稳定的 Target
            l_con = criterion_cl(h0, h1_t, batch_q1, batch_q2) + \
                    criterion_cl(h1, h0_t, batch_q2, batch_q1)
            
        else:
            # === Phase 2: Attention Enhancement ===
            # Attention 介入 (只处理 Online 特征)
            h0_enhanced, h1_enhanced = attention_mod(h0, h1)
            
            # 重建: Weighted MSE (忽略噪声)
            l_rec = criterion_mse(x0_flat, z0, batch_q1) + criterion_mse(x1_flat, z1, batch_q2)
            
            # 对比: Enhanced Online <-> Target (Full)
            # 逻辑: 增强后的特征是可信的，用力拉向稳定的 Target
            # 注意: Target 特征不需要过 Attention，它代表原始数据的平滑分布
            l_con = criterion_cl(h0_enhanced, h1_t, None, None) + \
                    criterion_cl(h1_enhanced, h0_t, None, None)

        loss = l_rec + args.lamda * l_con
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    # Log...
    epoch_time = time.time() - start_time
    if epoch % args.log_interval == 0:
        print(f"Epoch {epoch} Loss: {total_loss/len(train_loader):.4f}")

# --- 主程序 (已修复参数调用) ---
def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')
    print(f"==========\nSetting: GPU={args.gpu}, Lamda={args.lamda}, Epochs={args.epochs}, Momentum={args.momentum}\n==========")

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # 1. 加载质量权重
    if not os.path.exists('quality_weights.npz'):
        raise FileNotFoundError("Run 'generate_quality_weights.py' first!")
    
    q_data = np.load('quality_weights.npz')
    quality_tensor_v1 = torch.from_numpy(q_data['noise_score_v1']).float().to(device)
    quality_tensor_v2 = torch.from_numpy(q_data['noise_score_v2']).float().to(device)
    print(f"Loaded Quality Weights. Shapes: {quality_tensor_v1.shape}, {quality_tensor_v2.shape}")

    # 2. 加载数据
    # 注意：确保 loader 返回的数据维度与 MomentumSURE 中的 dim_v1, dim_v2 一致
    train_loader, all_loader, _ = loader_cl_noise(args.batch_size, args.data_name, args.seed)

    # 3. 实例化模型
    # 假设 View 1 是 GIST (512), View 2 是 PHOG (100) 或其他，请根据实际情况修改 models.py 中的 dim
    # 这里我们使用默认参数初始化
    backbone = MomentumSURE(feature_dim=512, momentum=args.momentum).to(device)
    attention_mod = CrossViewSemanticEnhancement(feature_dim=512).to(device)
    
    # 4. 优化器 (只优化 Online Encoder 和 Attention)
    params = list(backbone.encoder_v0.parameters()) + \
             list(backbone.encoder_v1.parameters()) + \
             list(backbone.decoder_v0.parameters()) + \
             list(backbone.decoder_v1.parameters()) + \
             list(attention_mod.parameters())
             
    optimizer = torch.optim.Adam(params, lr=args.lr)
    # 使用 Cosine Scheduler 消除震荡
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # 5. 损失函数
    criterion_mse = WeightedMSELoss().to(device)
    # 动量对比损失
    criterion_cl = MomentumContrastiveLoss(tau=args.tau).to(device)

    print("Start Momentum Dual-Tower Training...")
    best_acc = 0.0
    
    # 6. 训练循环
    for epoch in range(args.epochs):
        # === [修复点] 这里填入了完整的参数 ===
        train_one_epoch(
            train_loader, 
            (backbone, attention_mod),      # models 元组
            (criterion_mse, criterion_cl),  # criterions 元组
            optimizer, 
            epoch, 
            args, 
            quality_tensor_v1,              # q_v1
            quality_tensor_v2               # q_v2
        )
        
        # 更新学习率
        scheduler.step()
        
        # 7. 评估 (每10轮或最后一轮)
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
                    
                    # 评估时使用 Online Encoder (带梯度更新的那个)
                    h0, h1, _, _ = backbone.forward_online(x0_flat, x1_flat)
                    
                    # 如果进入第二阶段，应用 Attention 增强特征
                    if use_attention:
                        h0, h1 = attention_mod(h0, h1)
                        
                    feat_v0_list.append(h0.cpu().numpy())
                    feat_v1_list.append(h1.cpu().numpy())
                    gt_list.append(labels.cpu().numpy())
            
            # 拼接数据进行聚类
            data = [np.concatenate(feat_v0_list), np.concatenate(feat_v1_list)]
            gt_label = np.concatenate(gt_list)
            
            ret = Clustering(data, gt_label, random_state=args.seed)
            acc = ret['kmeans']['accuracy']
            nmi = ret['kmeans']['NMI']
            ari = ret['kmeans']['ARI']
            
            if acc > best_acc:
                best_acc = acc
            
            logging.info(f"Epoch {epoch} Result: ACC={acc:.4f}, NMI={nmi:.4f} (Best: {best_acc:.4f})")

    print(f"Training Finished. Final Best ACC: {best_acc:.4f}")

if __name__ == '__main__':
    main()