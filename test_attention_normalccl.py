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
from sklearn.manifold import TSNE  # 引入 TSNE
import umap  # 引入 UMAP

# 引入必要的模块 (假设这些文件在你的目录下)
from models import SUREfcScene
from Clustering import Clustering
from sure_inference import both_infer
from data_loader_noisy_scene import loader_cl_noise

# 设置环境
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = "1"

# --- 参数设置 ---
parser = argparse.ArgumentParser(description='Quality-Aware Scene15 Training with t-SNE')
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

# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def visualize_umap_distribution(features, labels, indices, total_samples, epoch, save_dir='vis_results'):
    """
    使用 UMAP 生成类别分布与噪声等级可视化
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    print(f"Epoch {epoch}: Computing UMAP for visualization...")
    
    # 1. 运行 UMAP
    # n_neighbors: 邻域大小，影响局部 vs 全局结构的平衡
    # min_dist: 允许的嵌入点之间的最小距离
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embeddings = reducer.fit_transform(features)
    
    # 2. 重建噪声等级 (逻辑与 t-SNE 函数保持一致)
    intervals_v1 = [
        (0.0, 0.1, 0.2), (0.1, 0.2, 0.4), (0.2, 0.3, 0.6),
        (0.3, 0.4, 0.8), (0.4, 0.5, 1.0), (0.5, 1.0, 0.0)
    ]
    intervals_v2 = [
        (0.0, 0.4, 0.0), (0.4, 0.5, 0.2), (0.5, 0.6, 0.4),
        (0.6, 0.7, 0.6), (0.7, 0.8, 0.8), (0.8, 0.9, 1.0), (0.9, 1.0, 0.0)
    ]
    
    noise_levels_v1 = np.zeros(len(indices))
    noise_levels_v2 = np.zeros(len(indices))
    
    for i, idx in enumerate(indices):
        for start, end, alpha in intervals_v1:
            s_idx = int(total_samples * start)
            e_idx = int(total_samples * end)
            if end >= 1.0: e_idx = total_samples
            if s_idx <= idx < e_idx:
                noise_levels_v1[i] = alpha
                break
        
        for start, end, alpha in intervals_v2:
            s_idx = int(total_samples * start)
            e_idx = int(total_samples * end)
            if end >= 1.0: e_idx = total_samples
            if s_idx <= idx < e_idx:
                noise_levels_v2[i] = alpha
                break

    # 3. 绘图 (三联画)
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    
    # 子图 1: 类别分布
    sns.scatterplot(x=embeddings[:,0], y=embeddings[:,1], hue=labels, 
                    palette='tab20', s=15, ax=axes[0], legend='full')
    axes[0].set_title(f'Epoch {epoch}: UMAP by Class Label')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    # 子图 2: 视图 1 噪声分布
    sns.scatterplot(x=embeddings[:,0], y=embeddings[:,1], hue=noise_levels_v1, 
                    palette='viridis', s=15, ax=axes[1])
    axes[1].set_title(f'Epoch {epoch}: UMAP by View 1 Noise Level')
    axes[1].legend(title='V1 Noise')

    # 子图 3: 视图 2 噪声分布
    sns.scatterplot(x=embeddings[:,0], y=embeddings[:,1], hue=noise_levels_v2, 
                    palette='viridis', s=15, ax=axes[2])
    axes[2].set_title(f'Epoch {epoch}: UMAP by View 2 Noise Level')
    axes[2].legend(title='V2 Noise')
    
    plt.tight_layout()
    save_path = f'{save_dir}/umap_epoch_{epoch}.png'
    plt.savefig(save_path)
    plt.close()
    print(f"UMAP visualization saved to {save_path}")

# --- [可视化工具 1] 修复效果热力图 (保持原样) ---
def visualize_repair_effect(h0_orig, h1_orig, h0_new, h1_new, quality_v1, quality_v2, epoch, save_dir='vis_results'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    h0_orig = F.normalize(h0_orig, dim=1).detach().cpu().numpy()
    h1_orig = F.normalize(h1_orig, dim=1).detach().cpu().numpy()
    h0_new = F.normalize(h0_new, dim=1).detach().cpu().numpy()
    h1_new = F.normalize(h1_new, dim=1).detach().cpu().numpy()
    q1 = quality_v1.detach().cpu().numpy()
    
    sim_orig = (h0_orig * h1_orig).sum(axis=1)
    sim_new = (h0_new * h1_new).sum(axis=1)
    drift_v1 = np.linalg.norm(h0_new - h0_orig, axis=1)
    
    # 散点图
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=q1, y=drift_v1, alpha=0.6, color='b')
    sns.regplot(x=q1, y=drift_v1, scatter=False, color='r')
    plt.title(f'Epoch {epoch}: Sample Quality vs. Feature Drift')
    plt.xlabel('Sample Quality (1 - NoiseScore)')
    plt.ylabel('Feature Drift (L2 Distance)')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_dir}/drift_scatter_epoch_{epoch}.png')
    plt.close()

    # 热力图
    sorted_indices = np.argsort(q1)
    low_q_idx = sorted_indices[:20] 
    high_q_idx = sorted_indices[-10:]
    display_idx = np.concatenate([low_q_idx, high_q_idx])
    
    heatmap_data = np.stack([
        q1[display_idx],          
        sim_orig[display_idx],    
        sim_new[display_idx],     
        drift_v1[display_idx]     
    ], axis=1)
    
    heatmap_data[:, 3] = heatmap_data[:, 3] / (heatmap_data[:, 3].max() + 1e-6)

    plt.figure(figsize=(8, 10))
    ax = sns.heatmap(heatmap_data, cmap="coolwarm", annot=True, fmt=".2f", 
                     xticklabels=['Quality', 'Orig Sim', 'Enhance Sim', 'Drift (Norm)'],
                     yticklabels=[f'Sample {i}' for i in range(len(display_idx))])
    
    ax.hlines([20], *ax.get_xlim(), colors='white', linestyles='dashed', linewidth=2)
    plt.text(4.2, 10, 'Low Quality Area', verticalalignment='center', rotation=270)
    plt.text(4.2, 25, 'High Quality Area', verticalalignment='center', rotation=270)
    plt.title(f'Epoch {epoch}: Repair Effect Heatmap')
    plt.savefig(f'{save_dir}/repair_heatmap_epoch_{epoch}.png')
    plt.close()

# --- [可视化工具 2] 新增: t-SNE 分布与噪声等级可视化 ---
def visualize_tsne_distribution(features, labels, indices, total_samples, epoch, save_dir='vis_results'):
    """
    Args:
        features: (N, D) 拼接后的多视图特征
        labels: (N,) 真实标签
        indices: (N,) 样本的原始索引 (用于反推噪声等级)
        total_samples: N 数据集总数
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    print(f"Epoch {epoch}: Computing t-SNE for visualization...")
    
    # 1. 运行 t-SNE
    # 为了速度，可以先用 PCA 降维到 50，再用 t-SNE
    tsne = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=42)
    embeddings = tsne.fit_transform(features)
    
    # 2. 重建噪声等级 (根据 data_loader_noisy_scene.py 中的逻辑)
    # 注意：这里的逻辑必须与 loader_cl_noise 中的 intervals 保持一致
    intervals_v1 = [
        (0.0, 0.1, 0.2), (0.1, 0.2, 0.4), (0.2, 0.3, 0.6),
        (0.3, 0.4, 0.8), (0.4, 0.5, 1.0), (0.5, 1.0, 0.0)
    ]
    intervals_v2 = [
        (0.0, 0.4, 0.0), (0.4, 0.5, 0.2), (0.5, 0.6, 0.4),
        (0.6, 0.7, 0.6), (0.7, 0.8, 0.8), (0.8, 0.9, 1.0), (0.9, 1.0, 0.0)
    ]
    
    noise_levels_v1 = np.zeros(len(indices))
    noise_levels_v2 = np.zeros(len(indices))
    
    for i, idx in enumerate(indices):
        # 计算该样本在原始全量数据中的相对位置比例
        # loader_cl_noise 中是先 shuffle 再按顺序注入噪声，所以 index 就对应位置
        # intervals 中的定义是 start_idx = int(N * start_ratio)
        
        # 判断 View 1
        for start, end, alpha in intervals_v1:
            s_idx = int(total_samples * start)
            e_idx = int(total_samples * end)
            if end >= 1.0: e_idx = total_samples
            if s_idx <= idx < e_idx:
                noise_levels_v1[i] = alpha
                break
        
        # 判断 View 2
        for start, end, alpha in intervals_v2:
            s_idx = int(total_samples * start)
            e_idx = int(total_samples * end)
            if end >= 1.0: e_idx = total_samples
            if s_idx <= idx < e_idx:
                noise_levels_v2[i] = alpha
                break

    # 3. 绘图 (三联画)
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    
    # 子图 1: 类别分布 (Class Distribution)
    sns.scatterplot(x=embeddings[:,0], y=embeddings[:,1], hue=labels, 
                    palette='tab20', s=15, ax=axes[0], legend='full')
    axes[0].set_title(f'Epoch {epoch}: t-SNE by Class Label')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    # 子图 2: 视图 1 噪声分布 (View 1 Noise Level)
    # 使用 viridis 颜色映射，颜色越亮/黄代表噪声越大 (1.0)，越暗/紫代表越干净 (0.0)
    scatter2 = sns.scatterplot(x=embeddings[:,0], y=embeddings[:,1], hue=noise_levels_v1, 
                    palette='viridis', s=15, ax=axes[1])
    axes[1].set_title(f'Epoch {epoch}: t-SNE by View 1 Noise Level')
    # 重新设置 Legend 标题
    axes[1].legend(title='V1 Noise')

    # 子图 3: 视图 2 噪声分布 (View 2 Noise Level)
    scatter3 = sns.scatterplot(x=embeddings[:,0], y=embeddings[:,1], hue=noise_levels_v2, 
                    palette='viridis', s=15, ax=axes[2])
    axes[2].set_title(f'Epoch {epoch}: t-SNE by View 2 Noise Level')
    axes[2].legend(title='V2 Noise')
    
    plt.tight_layout()
    save_path = f'{save_dir}/tsne_epoch_{epoch}.png'
    plt.savefig(save_path)
    plt.close()
    print(f"t-SNE visualization saved to {save_path}")

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

# --- 2. 损失函数 ---
# --- 2. 损失函数 (修改版) ---
class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()
    
    def forward(self, input, target, quality_weights=None):
        # 1. 计算基础的平方误差
        loss = (input - target) ** 2
        
        # 2. 判断是否需要加权
        if quality_weights is not None:
            w = quality_weights.view(-1, 1)
            loss = loss * w
        
        # 3. 返回均值
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

# --- 3. 训练函数 ---
def train_one_epoch(train_loader, models, criterions, optimizer, epoch, args, quality_tensor_v1, quality_tensor_v2):
    backbone, attention_mod = models
    criterion_mse, criterion_ccl = criterions
    
    backbone.train()
    if attention_mod: attention_mod.train()
    
    total_loss = 0
    start_time = time.time()
    use_attention = (epoch >= args.warmup_epochs)
    
    # 可视化采样控制
    do_vis = (epoch % args.log_interval == 0) and use_attention
    vis_done = False

    for batch_idx, (x0, x1, labels, _, _, _, indices) in enumerate(train_loader):
        x0, x1 = x0.to(device), x1.to(device)
        indices = indices.to(device)
        
        # 建议将权重映射修改为：
        # q_old 为 1 代表好，0 代表坏
        # 使用平方权重：使低质量样本的权重下降更快
        batch_q1 = (1.0 - quality_tensor_v1[indices]) ** 2 
        batch_q2 = (1.0 - quality_tensor_v2[indices]) ** 2

        # 如果担心梯度消失，可以保留一个极小的 epsilon
        # batch_q1 = batch_q1.clamp(min=1e-4)
        
        x0_flat = x0.view(x0.size(0), -1)
        x1_flat = x1.view(x1.size(0), -1)
        
        h0, h1, z0, z1 = backbone(x0_flat, x1_flat)
        # loss_mse = criterion_mse(x0_flat, z0, batch_q1) + criterion_mse(x1_flat, z1, batch_q2)
        
        if use_attention:
            h0_final, h1_final = attention_mod(h0, h1)
            
            # 训练中的可视化 (修复效果热力图)
            if do_vis and not vis_done:
                visualize_repair_effect(h0, h1, h0_final, h1_final, batch_q1, batch_q2, epoch)
                vis_done = True
            
            loss_mse = criterion_mse(x0_flat, z0, batch_q1) + criterion_mse(x1_flat, z1, batch_q2)
            loss_contrast = criterion_ccl(h0_final, h1_final, w0=None, w1=None)
            loss = loss_mse + (args.lamda * loss_contrast)
        else:
            h0_final, h1_final = h0, h1
            loss_mse = criterion_mse(x0_flat, z0, None) + criterion_mse(x1_flat, z1, None)
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
    
    # 获取数据集总样本数，用于计算噪声比例
    total_samples = len(all_loader.dataset)

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
        
        # --- Evaluation & t-SNE Visualization ---
        if epoch == args.epochs - 1 or (epoch + 1) % 10 == 0:
            backbone.eval()
            attention_mod.eval()
            use_attention = (epoch >= args.warmup_epochs)
            feat_v0_list, feat_v1_list, gt_list, indices_list = [], [], [], []
            
            with torch.no_grad():
                # 注意：这里需要接收 indices
                for x0, x1, labels, _, _, _, indices in all_loader:
                    x0, x1 = x0.to(device), x1.to(device)
                    x0_flat = x0.view(x0.size(0), -1)
                    x1_flat = x1.view(x1.size(0), -1)
                    
                    h0, h1, _, _ = backbone(x0_flat, x1_flat)
                    
                    if use_attention:
                        h0, h1 = attention_mod(h0, h1)
                        
                    feat_v0_list.append(h0.cpu().numpy())
                    feat_v1_list.append(h1.cpu().numpy())
                    gt_list.append(labels.cpu().numpy())
                    indices_list.append(indices.cpu().numpy())
            
            # 准备聚类数据
            feat_v0_all = np.concatenate(feat_v0_list)
            feat_v1_all = np.concatenate(feat_v1_list)
            gt_label = np.concatenate(gt_list)
            indices_all = np.concatenate(indices_list)
            
            data = [feat_v0_all, feat_v1_all]
            
            # 1. 执行聚类评估
            ret = Clustering(data, gt_label, random_state=args.seed)
            acc = ret['kmeans']['accuracy']
            nmi = ret['kmeans']['NMI']
            ari = ret['kmeans']['ARI']
            
            if acc > best_acc: best_acc = acc
            logging.info(f"Epoch {epoch} Result: ACC={acc:.4f}, NMI={nmi:.4f} (Best: {best_acc:.4f})")
            
            # 2. 执行 t-SNE 可视化 (将两视图特征拼接作为样本表示)
            # t-SNE 比较耗时，与聚类频率保持一致
            features_concat = np.concatenate([feat_v0_all, feat_v1_all], axis=1)
            # visualize_tsne_distribution(features_concat, gt_label, indices_all, total_samples, epoch)

            # visualize_umap_distribution(features_concat, gt_label, indices_all, total_samples, epoch)

    print(f"Final Best ACC: {best_acc:.4f}")

if __name__ == '__main__':
    main()