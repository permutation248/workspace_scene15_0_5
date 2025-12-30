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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap

# 引入必要的模块
from models import SUREfcScene
from Clustering import Clustering
from sure_inference import both_infer
from data_loader_noisy_scene import loader_cl_noise

# 设置环境
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = "1"

# --- 参数设置 ---
parser = argparse.ArgumentParser(description='Global Learnable Intra-Sample Attention')
parser.add_argument('--gpu', default='0', type=str, help='GPU device idx')
parser.add_argument('--epochs', default=200, type=int, help='Total training epochs')
parser.add_argument('--warmup-epochs', default=50, type=int, help='Epochs for Phase 1')
parser.add_argument('--batch-size', default=128, type=int, help='Batch size')
parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
parser.add_argument('--lamda', default=0.1, type=float, help='Weight for Contrastive Loss')
parser.add_argument('--tau', default=0.5, type=float, help='Temperature parameter')
parser.add_argument('--seed', default=1111, type=int, help='Random seed')
parser.add_argument('--log-interval', default=50, type=int, help='Log interval')
parser.add_argument('--data-name', default='Scene15', type=str, help='Dataset name')
parser.add_argument('--neg-prop', default=0, type=int)
parser.add_argument('--aligned-prop', default=1.0, type=float)
parser.add_argument('--complete-prop', default=1.0, type=float)
parser.add_argument('--noisy-training', default=False, type=bool)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- [可视化工具] ---
# 注意：由于Attention机制改变，热力图逻辑仅展示View1的一致性变化
def visualize_repair_effect(h0_orig, h1_orig, h0_new, h1_new, quality_v1, quality_v2, epoch, save_dir='vis_results'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    h0_orig = F.normalize(h0_orig, dim=1).detach().cpu().numpy()
    h1_orig = F.normalize(h1_orig, dim=1).detach().cpu().numpy()
    h0_new = F.normalize(h0_new, dim=1).detach().cpu().numpy()
    
    q1 = quality_v1.detach().cpu().numpy()
    drift_v1 = np.linalg.norm(h0_new - h0_orig, axis=1)
    
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=q1, y=drift_v1, alpha=0.6, color='b')
    plt.title(f'Epoch {epoch}: Quality vs. Drift (Intra-Sample Fusion)')
    plt.xlabel('Sample Quality')
    plt.ylabel('Feature Drift')
    plt.savefig(f'{save_dir}/drift_scatter_epoch_{epoch}.png')
    plt.close()

# --- 1. 全局可学习参数 Attention 模块 (样本内跨视图版) ---
class GlobalLearnableAttention(nn.Module):
    def __init__(self, num_samples, feature_dim=512):
        super(GlobalLearnableAttention, self).__init__()
        self.scale = feature_dim ** -0.5
        self.num_samples = num_samples
        self.feature_dim = feature_dim
        
        # --- 全局参数定义 ---
        # Q1, K1: 对应 View 1 的查询和键
        # Q2, K2: 对应 View 2 的查询和键
        # 形状均为 (N, D)
        self.Q1 = nn.Parameter(torch.empty(num_samples, feature_dim))
        self.K1 = nn.Parameter(torch.empty(num_samples, feature_dim))

        self.Q2 = nn.Parameter(torch.empty(num_samples, feature_dim))
        self.K2 = nn.Parameter(torch.empty(num_samples, feature_dim))
        
        self.reset_parameters()

    def reset_parameters(self):
        """
        初始化策略：
        1. Q1, Q2 独立正态分布初始化 -> 它们之间近似正交 (Q1 . Q2 ≈ 0)
        2. K1 = Q1, K2 = Q2 -> 自身点积很大 (Q1 . K1 >> 0)
        
        结果：
        计算 View 1 的 Attention 时：
           Score_Self  = Q1 . K1 = |Q1|^2 (大)
           Score_Other = Q1 . K2 = Q1 . Q2 (接近 0)
        因此初始阶段，每个视图只关注自己。
        """
        # 使用较大的 std 保证 softmax 后的 one-hot 特性
        std_val = 0.5 
        nn.init.normal_(self.Q1, mean=0, std=std_val)
        nn.init.normal_(self.Q2, mean=0, std=std_val)
        
        # 关键：将 K 绑定到对应的 Q 上
        self.K1.data.copy_(self.Q1.data)
        self.K2.data.copy_(self.Q2.data)
        
        print(f">> Global Attention Initialized. Mode: Intra-Sample Cross-View. std={std_val}")

    def forward(self, h0, h1, indices):
        """
        h0: (Batch, D) - View 1 特征
        h1: (Batch, D) - View 2 特征
        indices: (Batch, )
        """
        batch_size = h0.size(0)
        
        # 1. 提取当前 Batch 的全局参数
        q1 = self.Q1[indices] # (B, D)
        k1 = self.K1[indices] # (B, D)
        q2 = self.Q2[indices] # (B, D)
        k2 = self.K2[indices] # (B, D)

        # ==========================================
        #  View 1 更新逻辑 (Z0)
        # ==========================================
        # Query: View 1 的 Q -> 形状 (B, 1, D)
        q_v1 = q1.unsqueeze(1)
        
        # Keys: [View 1 的 K, View 2 的 K] -> 形状 (B, 2, D)
        k_stack = torch.stack([k1, k2], dim=1)
        
        # Values: [View 1 本身, View 2 本身] -> 形状 (B, 2, D)
        v_stack = torch.stack([h0, h1], dim=1)
        
        # Attention Score: (B, 1, D) @ (B, D, 2) -> (B, 1, 2)
        # 结果是一个 Batch 中每个样本对 V1 和 V2 的权重分布
        scores_v1 = torch.bmm(q_v1, k_stack.transpose(1, 2)) * self.scale
        attn_prob_v1 = F.softmax(scores_v1, dim=-1) # (B, 1, 2)
        
        # 加权求和: (B, 1, 2) @ (B, 2, D) -> (B, 1, D)
        z0_enhanced = torch.bmm(attn_prob_v1, v_stack).squeeze(1)
        z0_final = F.normalize(z0_enhanced, dim=1)

        # ==========================================
        #  View 2 更新逻辑 (Z1)
        # ==========================================
        # Query: View 2 的 Q
        q_v2 = q2.unsqueeze(1)
        
        # Keys & Values: 顺序保持一致 [V1, V2] 或者是 [V2, V1] 都可以，这里保持 [V1, V2] 统一索引
        # Index 0: View 1, Index 1: View 2
        
        scores_v2 = torch.bmm(q_v2, k_stack.transpose(1, 2)) * self.scale
        attn_prob_v2 = F.softmax(scores_v2, dim=-1) # (B, 1, 2)
        
        z1_enhanced = torch.bmm(attn_prob_v2, v_stack).squeeze(1)
        z1_final = F.normalize(z1_enhanced, dim=1)
        
        # 打印一下形状以确认 (仅在第一个batch打印)
        if torch.rand(1).item() < 0.01:
            print(f"DEBUG: Attn Prob Shape: {attn_prob_v1.shape} (Should be [B, 1, 2])")

        return z0_final, z1_final

# --- 2. 损失函数 ---
class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()
    
    def forward(self, input, target, quality_weights=None):
        loss = (input - target) ** 2
        if quality_weights is not None:
            w = quality_weights.view(-1, 1)
            loss = loss * w
        return loss.mean()

class WeightedCCL(nn.Module):
    def __init__(self, tau=0.5):
        super(WeightedCCL, self).__init__()
        self.tau = tau
        self.device = device
    
    def forward(self, x0, x1, w0=None, w1=None):
        out = torch.mm(x0, x1.t()) / self.tau
        labels = torch.arange(out.size(0)).to(self.device)
        loss_func = nn.CrossEntropyLoss(reduction='none')
        l_i2t = loss_func(out, labels)
        l_t2i = loss_func(out.t(), labels)
        
        if w0 is not None and w1 is not None:
            w_pair = (w0 * w1)
            l_i2t = l_i2t * w_pair
            l_t2i = l_t2i * w_pair
            
        return l_i2t.mean() + l_t2i.mean()

# --- 3. 训练函数 ---
def train_one_epoch(train_loader, models, criterions, optimizer, epoch, args, quality_tensor_v1, quality_tensor_v2):
    backbone, attention_mod = models
    criterion_mse, criterion_ccl = criterions
    
    backbone.train()
    if attention_mod: attention_mod.train()
    
    total_loss = 0
    start_time = time.time()
    use_attention = (epoch >= args.warmup_epochs)
    
    do_vis = (epoch % args.log_interval == 0) and use_attention
    vis_done = False

    for batch_idx, (x0, x1, labels, _, _, _, indices) in enumerate(train_loader):
        x0, x1 = x0.to(device), x1.to(device)
        indices = indices.to(device)
        
        batch_q1 = (1.0 - quality_tensor_v1[indices]) ** 2 
        batch_q2 = (1.0 - quality_tensor_v2[indices]) ** 2
        
        x0_flat = x0.view(x0.size(0), -1)
        x1_flat = x1.view(x1.size(0), -1)
        
        h0, h1, z0, z1 = backbone(x0_flat, x1_flat)
        
        if use_attention:
            # Phase 2: 使用全局可学习 Attention (Intra-Sample)
            h0_final, h1_final = attention_mod(h0, h1, indices)
            
            if do_vis and not vis_done:
                visualize_repair_effect(h0, h1, h0_final, h1_final, batch_q1, batch_q2, epoch)
                vis_done = True
            
            loss_mse = criterion_mse(x0_flat, z0, batch_q1) + criterion_mse(x1_flat, z1, batch_q2)
            loss_contrast = criterion_ccl(h0_final, h1_final, w0=None, w1=None)
            loss = loss_mse + (args.lamda * loss_contrast)
        else:
            # Phase 1: 预热
            h0_final, h1_final = h0, h1
            loss_mse = criterion_mse(x0_flat, z0, None) + criterion_mse(x1_flat, z1, None)
            loss_contrast = criterion_ccl(h0_final, h1_final, w0=batch_q1, w1=batch_q2)
            loss = loss_mse + (args.lamda * loss_contrast)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    epoch_time = time.time() - start_time
    phase_str = "Phase 2 (GlbAttn)" if use_attention else "Phase 1 (Warmup)"
    
    if epoch % args.log_interval == 0:
         logging.info(f"Epoch [{epoch}/{args.epochs}] [{phase_str}] Time: {epoch_time:.2f}s | Loss: {total_loss / len(train_loader):.4f}")

# --- 主程序 ---
def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')
    print(f"==========\nSetting: GPU={args.gpu}, Lamda={args.lamda}, Epochs={args.epochs}\n==========")

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if not os.path.exists('quality_weights.npz'):
        raise FileNotFoundError("Run 'generate_quality_weights.py' first!")
    
    q_data = np.load('quality_weights.npz')
    quality_tensor_v1 = torch.from_numpy(q_data['noise_score_v1']).float().to(device)
    quality_tensor_v2 = torch.from_numpy(q_data['noise_score_v2']).float().to(device)

    train_loader, all_loader, _ = loader_cl_noise(args.batch_size, args.data_name, args.seed)
    
    total_samples = len(all_loader.dataset)
    print(f"Total samples detected: {total_samples}")

    backbone = SUREfcScene().to(device)
    attention_mod = GlobalLearnableAttention(num_samples=total_samples, feature_dim=512).to(device)
    
    params = list(backbone.parameters()) + list(attention_mod.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    criterion_mse = WeightedMSELoss().to(device)
    criterion_ccl = WeightedCCL(tau=args.tau)

    print("Start Training...")
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        train_one_epoch(
            train_loader, (backbone, attention_mod), (criterion_mse, criterion_ccl), 
            optimizer, epoch, args, quality_tensor_v1, quality_tensor_v2
        )
        scheduler.step()
        
        if epoch == args.epochs - 1 or (epoch + 1) % 10 == 0:
            backbone.eval()
            attention_mod.eval()
            use_attention = (epoch >= args.warmup_epochs)
            feat_v0_list, feat_v1_list, gt_list = [], [], []
            
            with torch.no_grad():
                for x0, x1, labels, _, _, _, indices in all_loader:
                    x0, x1 = x0.to(device), x1.to(device)
                    indices = indices.to(device)
                    x0_flat = x0.view(x0.size(0), -1)
                    x1_flat = x1.view(x1.size(0), -1)
                    
                    h0, h1, _, _ = backbone(x0_flat, x1_flat)
                    
                    if use_attention:
                        h0, h1 = attention_mod(h0, h1, indices)
                        
                    feat_v0_list.append(h0.cpu().numpy())
                    feat_v1_list.append(h1.cpu().numpy())
                    gt_list.append(labels.cpu().numpy())
            
            feat_v0_all = np.concatenate(feat_v0_list)
            feat_v1_all = np.concatenate(feat_v1_list)
            gt_label = np.concatenate(gt_list)
            
            data = [feat_v0_all, feat_v1_all]
            
            ret = Clustering(data, gt_label, random_state=args.seed)
            acc = ret['kmeans']['accuracy']
            nmi = ret['kmeans']['NMI']
            
            if acc > best_acc: best_acc = acc
            logging.info(f"Epoch {epoch} Result: ACC={acc:.4f}, NMI={nmi:.4f} (Best: {best_acc:.4f})")

    print(f"Final Best ACC: {best_acc:.4f}")

if __name__ == '__main__':
    main()