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

# 引入必要的模块 (假设这些文件在你的目录下)
from models import SUREfcScene
from Clustering import Clustering
from sure_inference import both_infer
from data_loader_noisy_scene import loader_cl_noise

# 设置环境
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = "1"

# --- 参数设置 ---
parser = argparse.ArgumentParser(description='Quality-Aware Scene15 Training')
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

# --- [可视化工具] 核心新增函数 ---
def visualize_repair_effect(h0_orig, h1_orig, h0_new, h1_new, quality_v1, quality_v2, epoch, save_dir='vis_results'):
    """
    可视化 Attention 对不同质量样本的修复效果
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 1. 准备数据 (转为 numpy)
    h0_orig = F.normalize(h0_orig, dim=1).detach().cpu().numpy()
    h1_orig = F.normalize(h1_orig, dim=1).detach().cpu().numpy()
    h0_new = F.normalize(h0_new, dim=1).detach().cpu().numpy()
    h1_new = F.normalize(h1_new, dim=1).detach().cpu().numpy()
    q1 = quality_v1.detach().cpu().numpy()
    
    # 2. 计算视图间的一致性 (Cosine Similarity)
    # 原始一致性: diag(h0 @ h1.T)
    sim_orig = (h0_orig * h1_orig).sum(axis=1)
    # 增强后一致性
    sim_new = (h0_new * h1_new).sum(axis=1)
    
    # 3. 计算特征漂移量 (Feature Drift): ||h_new - h_orig||
    # 这代表了 Attention 模块对原始特征改变了多少
    drift_v1 = np.linalg.norm(h0_new - h0_orig, axis=1)
    
    # --- 图表 1: 散点图 (质量 vs. 修复强度) ---
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=q1, y=drift_v1, alpha=0.6, color='b')
    # 添加拟合线
    sns.regplot(x=q1, y=drift_v1, scatter=False, color='r')
    plt.title(f'Epoch {epoch}: Sample Quality vs. Feature Drift (Repair Intensity)')
    plt.xlabel('Sample Quality (1 - NoiseScore)')
    plt.ylabel('Feature Drift (L2 Distance)')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_dir}/drift_scatter_epoch_{epoch}.png')
    plt.close()

    # --- 图表 2: 热力图 (排序后的对比) ---
    # 对样本按质量从低到高排序
    sorted_indices = np.argsort(q1)
    
    # 为了可视化清晰，只取部分样本（例如 batch 中的前 50 个或者均匀采样）
    # 这里我们取质量最低的 20 个和质量最高的 10 个进行对比展示
    low_q_idx = sorted_indices[:20] 
    high_q_idx = sorted_indices[-10:]
    display_idx = np.concatenate([low_q_idx, high_q_idx])
    
    # 准备热力图数据
    heatmap_data = np.stack([
        q1[display_idx],          # Col 0: 质量
        sim_orig[display_idx],    # Col 1: 原始一致性
        sim_new[display_idx],     # Col 2: 增强后一致性
        drift_v1[display_idx]     # Col 3: 修复强度
    ], axis=1)
    
    # 归一化 Drift 以便绘图颜色协调
    heatmap_data[:, 3] = heatmap_data[:, 3] / (heatmap_data[:, 3].max() + 1e-6)

    plt.figure(figsize=(8, 10))
    # 绘制热力图
    ax = sns.heatmap(heatmap_data, cmap="coolwarm", annot=True, fmt=".2f", 
                     xticklabels=['Quality', 'Orig Sim', 'Enhance Sim', 'Drift (Norm)'],
                     yticklabels=[f'Sample {i}' for i in range(len(display_idx))])
    
    # 画一条线分割低质量和高质量区域
    ax.hlines([20], *ax.get_xlim(), colors='white', linestyles='dashed', linewidth=2)
    plt.text(4.2, 10, 'Low Quality Area', verticalalignment='center', rotation=270)
    plt.text(4.2, 25, 'High Quality Area', verticalalignment='center', rotation=270)
    
    plt.title(f'Epoch {epoch}: Repair Effect Heatmap (Sorted by Quality)')
    plt.savefig(f'{save_dir}/repair_heatmap_epoch_{epoch}.png')
    plt.close()
    
    print(f"Visualization saved to {save_dir}/")

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
        # View 1 增强 (利用 View 2)
        q1 = self.w_q1(h0)
        k1 = self.w_k1(h1)
        v1 = self.w_v1(h1)
        
        attn_score1 = torch.matmul(q1, k1.transpose(-2, -1)) * self.scale
        attn_prob1 = F.softmax(attn_score1, dim=-1)
        h0_enhanced = torch.matmul(attn_prob1, v1)
        
        # View 2 增强 (利用 View 1)
        q2 = self.w_q2(h1)
        k2 = self.w_k2(h0)
        v2 = self.w_v2(h0)
        
        attn_score2 = torch.matmul(q2, k2.transpose(-2, -1)) * self.scale
        attn_prob2 = F.softmax(attn_score2, dim=-1)
        h1_enhanced = torch.matmul(attn_prob2, v2)
        
        z0_final = self.layer_norm(h0 + h0_enhanced)
        z1_final = self.layer_norm(h1 + h1_enhanced)
        
        return z0_final, z1_final

# --- 2. 损失函数 (保持不变) ---
class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()
    
    def forward(self, input, target, quality_weights):
        w = quality_weights.view(-1, 1)
        loss = (input - target) ** 2
        loss = loss * w
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
            w_pair = (w0 * w1)
            l_i2t = l_i2t * w_pair
            l_t2i = l_t2i * w_pair
            
        return l_i2t.mean() + l_t2i.mean()

# --- 3. 训练核心函数 (已修改以支持可视化) ---
def train_one_epoch(train_loader, models, criterions, optimizer, epoch, args, quality_tensor_v1, quality_tensor_v2):
    backbone, attention_mod = models
    criterion_mse, criterion_ccl = criterions
    
    backbone.train()
    if attention_mod: attention_mod.train()
    
    total_loss = 0
    start_time = time.time()
    use_attention = (epoch >= args.warmup_epochs)
    
    # 决定是否在这个 Epoch 进行可视化 (例如每50轮，且只在 Phase 2 进行)
    do_vis = (epoch % args.log_interval == 0) and use_attention
    vis_done = False # 确保只画一个 batch

    for batch_idx, (x0, x1, labels, _, _, _, indices) in enumerate(train_loader):
        x0, x1 = x0.to(device), x1.to(device)
        indices = indices.to(device)
        
        batch_q1 = 1.0 - quality_tensor_v1[indices]
        batch_q2 = 1.0 - quality_tensor_v2[indices]
        
        x0_flat = x0.view(x0.size(0), -1)
        x1_flat = x1.view(x1.size(0), -1)
        
        h0, h1, z0, z1 = backbone(x0_flat, x1_flat)
        loss_mse = criterion_mse(x0_flat, z0, batch_q1) + criterion_mse(x1_flat, z1, batch_q2)
        
        if use_attention:
            h0_final, h1_final = attention_mod(h0, h1)
            
            # --- 可视化插入点 ---
            if do_vis and not vis_done:
                # 传入原始特征(h0)和增强后特征(h0_final)进行对比
                visualize_repair_effect(h0, h1, h0_final, h1_final, batch_q1, batch_q2, epoch)
                vis_done = True
            # --------------------

            loss_contrast = criterion_ccl(h0_final, h1_final, w0=None, w1=None)
            loss = loss_mse + (1.0 * loss_contrast)
        else:
            h0_final, h1_final = h0, h1
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

    backbone = SUREfcScene().to(device)
    attention_mod = CrossViewSemanticEnhancement(feature_dim=512).to(device)
    
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
            
            if acc > best_acc: best_acc = acc
            logging.info(f"Epoch {epoch} Result: ACC={acc:.4f}, NMI={nmi:.4f} (Best: {best_acc:.4f})")

    print(f"Final Best ACC: {best_acc:.4f}")

if __name__ == '__main__':
    main()