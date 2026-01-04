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
from sklearn.cluster import KMeans

# 引入必要的模块
from models import SUREfcScene
from Clustering import Clustering
from data_loader_noisy_scene import loader_cl_noise

# 设置环境
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = "1"

# --- 参数设置 ---
parser = argparse.ArgumentParser(description='Best-View Selection KMeans & Weighted CCL')
parser.add_argument('--gpu', default='0', type=str, help='GPU device idx')
parser.add_argument('--epochs', default=200, type=int, help='Total training epochs')
parser.add_argument('--warmup-epochs', default=50, type=int, help='Epochs for Phase 1 (Warmup)')
parser.add_argument('--batch-size', default=128, type=int, help='Batch size')
parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
parser.add_argument('--lamda', default=0.1, type=float, help='Weight for Contrastive Loss')
parser.add_argument('--beta', default=0.1, type=float, help='Weight for Pseudo-Label Supervised Loss')
parser.add_argument('--alpha', default=0.1, type=float, help='Weight for Robust Contrastive Loss')
parser.add_argument('--tau', default=0.5, type=float, help='Temperature parameter')
parser.add_argument('--q', default=0.7, type=float, help='Parameter q for Robust Contrastive Loss')
parser.add_argument('--seed', default=1111, type=int, help='Random seed')
parser.add_argument('--log-interval', default=50, type=int, help='Log interval')
parser.add_argument('--data-name', default='Scene15', type=str, help='Dataset name')
parser.add_argument('--num-classes', default=15, type=int, help='Number of clusters/classes')

# 兼容性参数
parser.add_argument('--neg-prop', default=0, type=int)
parser.add_argument('--aligned-prop', default=1.0, type=float)
parser.add_argument('--complete-prop', default=1.0, type=float)
parser.add_argument('--noisy-training', default=False, type=bool)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 2. 伪标签生成 (Best View Selection) ---
def generate_best_view_pseudo_labels(model, data_loader, quality_v1, quality_v2, num_classes, device):
    """
    生成伪标签策略：
    1. 提取所有视图特征。
    2. 对于每个样本，比较 Q1 和 Q2，选择质量最高的视图特征作为代表。
    3. 仅对这些“最佳特征”进行 K-Means 聚类。
    """
    model.eval()
    print("  > Extracting features for pseudo-label generation...")
    
    features_v1_list = []
    features_v2_list = []
    indices_list = []
    
    with torch.no_grad():
        for x0, x1, _, _, _, _, indices in data_loader:
            x0 = x0.to(device).view(x0.size(0), -1)
            x1 = x1.to(device).view(x1.size(0), -1)
            z0, z1, _, _ = model(x0, x1)
            features_v1_list.append(z0.cpu().numpy())
            features_v2_list.append(z1.cpu().numpy())
            indices_list.append(indices.cpu().numpy())
            
    features_v1 = np.concatenate(features_v1_list, axis=0)
    features_v2 = np.concatenate(features_v2_list, axis=0)
    all_indices = np.concatenate(indices_list, axis=0)
    
    N = features_v1.shape[0]
    
    # 获取质量分数 (越高越好)
    q1_all = 1.0 - quality_v1.cpu().numpy()
    q2_all = 1.0 - quality_v2.cpu().numpy()
    
    # 按照 all_indices 对齐质量分数
    q1_sorted = np.zeros(N)
    q2_sorted = np.zeros(N)
    q1_sorted[all_indices] = q1_all[all_indices]
    q2_sorted[all_indices] = q2_all[all_indices]
    
    # --- 关键修改：构建最佳视图特征集 ---
    best_features = np.zeros_like(features_v1)
    
    # 记录选择了哪个视图 (用于调试信息)
    select_v1_count = 0
    
    for i in range(N):
        if q1_sorted[i] >= q2_sorted[i]:
            best_features[i] = features_v1[i]
            select_v1_count += 1
        else:
            best_features[i] = features_v2[i]
            
    print(f"  > Selected Best Views: V1={select_v1_count}, V2={N - select_v1_count}")
    
    # --- K-Means 聚类 ---
    print(f"  > Running K-Means on {N} representative samples (Best Views)...")
    # 这里不需要 sample_weight，因为我们已经通过“择优录取”硬性筛选了高质量特征
    kmeans = KMeans(n_clusters=num_classes, n_init=20, random_state=args.seed)
    kmeans.fit(best_features)
    
    cluster_centers = kmeans.cluster_centers_
    all_labels = kmeans.labels_ # (N, )
    
    # 构建全局标签张量
    final_labels_global = torch.zeros(len(data_loader.dataset), dtype=torch.long).to(device)
    final_labels_global[all_indices] = torch.from_numpy(all_labels).to(device)
    
    return final_labels_global, cluster_centers

# --- 3. 损失函数 ---

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
    """
    加权对比损失 (Outer Weighting):
    正样本和负样本均受质量权重影响。
    loss_ij = weight_ij * CrossEntropy(logits, label)
    """
    def __init__(self, tau=0.5):
        super(WeightedCCL, self).__init__()
        self.tau = tau
        self.device = device

    def forward(self, x0, x1, w0=None, w1=None):
        # 计算 Logits
        out = torch.mm(x0, x1.t()) / self.tau
        
        # 标签
        labels = torch.arange(out.size(0)).to(self.device)
        
        # 计算基础 CrossEntropyLoss (Reduction=None 以便加权)
        loss_func = nn.CrossEntropyLoss(reduction='none')
        l_i2t = loss_func(out, labels)
        l_t2i = loss_func(out.t(), labels)
        
        # 加权逻辑
        if w0 is not None and w1 is not None:
            # w_pair[i] = w0[i] * w1[i] 
            # 代表这对样本 (View1_i, View2_i) 的整体可靠性
            # 将此权重应用于该样本产生的 Loss
            w_pair = (w0 * w1)
            
            l_i2t = l_i2t * w_pair
            l_t2i = l_t2i * w_pair
            
        return l_i2t.mean() + l_t2i.mean()

# --- 4. 训练核心函数 ---
def train_one_epoch(train_loader, model, criterions, optimizer, epoch, args, 
                    quality_tensor_v1, quality_tensor_v2, 
                    pseudo_labels=None, cluster_centers=None):
    
    criterion_mse, criterion_ccl = criterions
    model.train()
    
    total_loss = 0
    start_time = time.time()
    
    use_pseudo = (epoch >= args.warmup_epochs) and (pseudo_labels is not None) and (cluster_centers is not None)
    
    if use_pseudo:
        final_centers = torch.tensor(cluster_centers).float().to(device)
        final_centers = F.normalize(final_centers, dim=1)

    for batch_idx, (x0, x1, _, _, _, _, indices) in enumerate(train_loader):
        x0, x1 = x0.to(device), x1.to(device)
        indices = indices.to(device)
        
        # --- 权重定义 ---
        q1_quality = (1.0 - quality_tensor_v1[indices])
        q2_quality = (1.0 - quality_tensor_v2[indices])
        
        # 1. 重建 & 对比损失权重 (信赖高质量)
        w_quality1 = q1_quality ** 2
        w_quality2 = q2_quality ** 2
        
        # 2. 伪标签损失权重 (反向权重 / Hard Mining)
        # 保持上一版本的逻辑：质量差的样本更需要被拉回中心
        w_pl1 = quality_tensor_v1[indices]
        w_pl2 = quality_tensor_v2[indices]
        
        x0_flat = x0.view(x0.size(0), -1)
        x1_flat = x1.view(x1.size(0), -1)
        
        # Forward
        z0, z1, xr0, xr1 = model(x0_flat, x1_flat)
        
        # --- Loss 1: Reconstruction ---
        loss_mse = criterion_mse(x0_flat, xr0, w_quality1.view(-1, 1)) + \
                   criterion_mse(x1_flat, xr1, w_quality2.view(-1, 1))
        
        if not use_pseudo:
            # === Phase 1: Warmup ===
            # 正负样本均加权 (由 WeightedCCL 内部实现)
            loss_contrast = criterion_ccl(z0, z1, w0=w_quality1, w1=w_quality2)
            loss = loss_mse + (args.lamda * loss_contrast)
        
        else:
            # === Phase 2: Pseudo-Label Supervision ===
            
            # Target (One-Hot)
            batch_y = pseudo_labels[indices]
            p_target = F.one_hot(batch_y, num_classes=args.num_classes).float()

            # Prediction
            zp0 = torch.mm(z0, final_centers.T)
            zp1 = torch.mm(z1, final_centers.T)
            pre0 = F.softmax(zp0, dim=1)
            pre1 = F.softmax(zp1, dim=1)
            
            # Loss Calculation
            log_pre0 = torch.log(pre0 + 1e-8)
            log_pre1 = torch.log(pre1 + 1e-8)
            
            pp0 = (p_target * log_pre0).sum(dim=1)
            pp1 = (p_target * log_pre1).sum(dim=1)
            
            loss_pl0 = - (w_pl1 * pp0).mean()
            loss_pl1 = - (w_pl2 * pp1).mean()
            loss_pl = args.beta * (loss_pl0 + loss_pl1)
            
            # Robust Contrastive Loss (Phase 2)
            cos = z0.mm(z1.t())
            sim_c = (cos / args.tau).exp()
            pos = sim_c.diag()
            q = args.q
            loss_rcl = (((1 - q) * (sim_c.sum(1))**q - pos ** q) / q).sum().mean()
            loss_contrast_robust = args.alpha * loss_rcl
            
            loss = loss_mse + loss_pl + loss_contrast_robust

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    epoch_time = time.time() - start_time
    phase_str = "Phase 2 (Pseudo-Label)" if use_pseudo else "Phase 1 (Warmup)"
    
    if epoch % args.log_interval == 0:
         logging.info(f"Epoch [{epoch}/{args.epochs}] [{phase_str}] Time: {epoch_time:.2f}s | Loss: {total_loss / len(train_loader):.4f}")

# --- 主程序 ---
def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')
    print(f"==========\nSetting: GPU={args.gpu}, Mode=BestViewKMeans, Epochs={args.epochs}\n==========")

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # 1. 加载质量分数
    if not os.path.exists('quality_weights.npz'):
        raise FileNotFoundError("Run 'generate_quality_weights.py' first!")
    
    q_data = np.load('quality_weights.npz')
    quality_tensor_v1 = torch.from_numpy(q_data['noise_score_v1']).float().to(device)
    quality_tensor_v2 = torch.from_numpy(q_data['noise_score_v2']).float().to(device)

    # 2. 加载数据
    train_loader, all_loader, _ = loader_cl_noise(args.batch_size, args.data_name, args.seed)
    
    # 3. 初始化模型
    model = SUREfcScene().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 4. 初始化损失
    criterion_mse = WeightedMSELoss().to(device)
    criterion_ccl = WeightedCCL(tau=args.tau).to(device)
    criterions = (criterion_mse, criterion_ccl)

    print("Start Training...")
    best_acc = 0.0
    
    pseudo_labels = None
    cluster_centers = None
    
    for epoch in range(args.epochs):
        
        # --- Warmup 结束：生成静态伪标签 (Best View Strategy) ---
        if epoch == args.warmup_epochs:
            print(f"\n[Epoch {epoch}] Warmup finished. Generating Pseudo-Labels (Best View Strategy)...")
            pseudo_labels, cluster_centers = generate_best_view_pseudo_labels(
                model, all_loader, quality_tensor_v1, quality_tensor_v2, 
                args.num_classes, device
            )
            print("[Info] Pseudo-Labels and Centers fixed.")

        # --- 训练 ---
        train_one_epoch(
            train_loader, model, criterions, optimizer, epoch, args, 
            quality_tensor_v1, quality_tensor_v2, 
            pseudo_labels=pseudo_labels, cluster_centers=cluster_centers
        )
        
        # --- 评估 (同样使用 Best View Strategy) ---
        if epoch == args.epochs - 1 or (epoch + 1) % 10 == 0:
            model.eval()
            
            feat_v1_list, feat_v2_list, gt_list, indices_list = [], [], [], []
            
            with torch.no_grad():
                for x0, x1, labels, _, _, _, indices in all_loader:
                    x0 = x0.to(device).view(x0.size(0), -1)
                    x1 = x1.to(device).view(x1.size(0), -1)
                    
                    z0, z1, _, _ = model(x0, x1)
                        
                    feat_v1_list.append(z0.cpu().numpy())
                    feat_v2_list.append(z1.cpu().numpy())
                    gt_list.append(labels.cpu().numpy())
                    indices_list.append(indices.cpu().numpy())
            
            # 拼接
            feat_v1 = np.concatenate(feat_v1_list)
            feat_v2 = np.concatenate(feat_v2_list)
            gt_label = np.concatenate(gt_list)
            all_indices = np.concatenate(indices_list)
            
            # 获取质量并对齐
            q1_all = 1.0 - quality_tensor_v1.cpu().numpy()
            q2_all = 1.0 - quality_tensor_v2.cpu().numpy()
            q1_sorted = q1_all[all_indices]
            q2_sorted = q2_all[all_indices]
            
            # --- 关键：评估时也只取最好的那个视图 ---
            N = feat_v1.shape[0]
            best_features_eval = np.zeros_like(feat_v1)
            for i in range(N):
                if q1_sorted[i] >= q2_sorted[i]:
                    best_features_eval[i] = feat_v1[i]
                else:
                    best_features_eval[i] = feat_v2[i]
            
            # 传入 Clustering 的数据格式：虽然通常传 list，但这里我们传处理好的单视图矩阵
            # 为了兼容 Clustering.py，我们将其包装成列表，或者直接调用 sklearn 计算
            # Clustering.py 内部是 concatenate list 然后 kmeans
            # 这里我们传入 [best_features_eval]，效果等同于对 best features 聚类
            
            ret = Clustering([best_features_eval], gt_label, random_state=args.seed)
            acc = ret['kmeans']['accuracy']
            nmi = ret['kmeans']['NMI']
            
            if acc > best_acc: best_acc = acc
            logging.info(f"Epoch {epoch} Result (Best View): ACC={acc:.4f}, NMI={nmi:.4f} (Best: {best_acc:.4f})")

    print(f"Final Best ACC: {best_acc:.4f}")

if __name__ == '__main__':
    main()