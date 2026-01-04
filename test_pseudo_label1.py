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
parser = argparse.ArgumentParser(description='Quality-Aware Pseudo-Label Learning for Scene15 (ROLL Style)')
parser.add_argument('--gpu', default='0', type=str, help='GPU device idx')
parser.add_argument('--epochs', default=200, type=int, help='Total training epochs')
parser.add_argument('--warmup-epochs', default=50, type=int, help='Epochs for Phase 1 (Warmup)')
parser.add_argument('--batch-size', default=128, type=int, help='Batch size')
parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
parser.add_argument('--lamda', default=0.1, type=float, help='Weight for Contrastive Loss')
parser.add_argument('--beta', default=0.1, type=float, help='Weight for Pseudo-Label Supervised Loss')
parser.add_argument('--tau', default=0.5, type=float, help='Temperature parameter')
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

# --- 2. 核心逻辑: 加权 K-Means 与 伪标签生成 ---
def generate_weighted_pseudo_labels(model, data_loader, quality_v1, quality_v2, num_classes, device):
    """
    执行一次性伪标签生成：
    1. 提取特征。
    2. 加权 K-Means 获取语义中心。
    3. 在拼接特征上统一预测，然后切分。
    4. 冲突消解：以高质量视图为准。
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
            
            # 使用模型提取特征 (z0, z1 是归一化后的)
            z0, z1, _, _ = model(x0, x1)
            
            features_v1_list.append(z0.cpu().numpy())
            features_v2_list.append(z1.cpu().numpy())
            indices_list.append(indices.cpu().numpy())
            
    # 拼接全量数据
    features_v1 = np.concatenate(features_v1_list, axis=0)
    features_v2 = np.concatenate(features_v2_list, axis=0)
    all_indices = np.concatenate(indices_list, axis=0)
    
    N = features_v1.shape[0]
    
    # 准备质量权重 (对齐到 all_indices 顺序)
    # 质量分数: 0=Clean, 1=Noisy -> 转换权重: 1 - score
    q1_all = 1.0 - quality_v1.cpu().numpy()
    q2_all = 1.0 - quality_v2.cpu().numpy()
    
    # 按提取时的 index 顺序重排质量分数，确保对应正确
    q1_sorted = np.zeros(N)
    q2_sorted = np.zeros(N)
    q1_sorted[all_indices] = q1_all[all_indices]
    q2_sorted[all_indices] = q2_all[all_indices]
    
    # --- 加权 K-Means 聚类 ---
    # 拼接所有视图特征 (2N, D)
    pool_features = np.concatenate([features_v1, features_v2], axis=0)
    pool_weights = np.concatenate([q1_sorted, q2_sorted], axis=0)
    
    print(f"  > Running Weighted K-Means on {2*N} samples...")
    # 使用 sample_weight 进行加权聚类
    kmeans = KMeans(n_clusters=num_classes, n_init=20, random_state=args.seed)
    kmeans.fit(pool_features, sample_weight=pool_weights)
    
    cluster_centers = kmeans.cluster_centers_ # (K, D)
    
    # --- 生成每个视图的初步标签 (修改点) ---
    # 按照您的要求：不单独预测，而是用拼接好的 pool feature 进行预测
    print("  > Predicting labels using pooled features...")
    all_labels = kmeans.predict(pool_features)
    
    # 切分回 View1 和 View2
    labels_v1 = all_labels[:N]
    labels_v2 = all_labels[N:]
    
    # --- 冲突消解 (Conflict Resolution) ---
    # 如果 v1 和 v2 预测不一致，采信质量高的那个
    final_labels = np.zeros(N, dtype=np.int64)
    conflict_count = 0
    
    for i in range(N):
        if labels_v1[i] == labels_v2[i]:
            final_labels[i] = labels_v1[i]
        else:
            conflict_count += 1
            if q1_sorted[i] >= q2_sorted[i]:
                final_labels[i] = labels_v1[i]
            else:
                final_labels[i] = labels_v2[i]
                
    print(f"  > Pseudo-labels generated. Conflicts resolved: {conflict_count}/{N} ({conflict_count/N:.2%})")
    
    # 将 final_labels 按照 original indices 的顺序放回去 (因为 all_indices 可能不是 0,1,2...)
    # 创建一个全局张量存储
    final_labels_global = torch.zeros(len(data_loader.dataset), dtype=torch.long).to(device)
    final_labels_global[all_indices] = torch.from_numpy(final_labels).to(device)
    
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

# --- 4. 训练核心函数 ---
def train_one_epoch(train_loader, model, criterions, optimizer, epoch, args, 
                    quality_tensor_v1, quality_tensor_v2, 
                    pseudo_labels=None, cluster_centers=None):
    
    criterion_mse, criterion_ccl = criterions
    model.train()
    
    total_loss = 0
    start_time = time.time()
    
    # 判断是否处于 Phase 2 (使用伪标签)
    use_pseudo = (epoch >= args.warmup_epochs) and (pseudo_labels is not None) and (cluster_centers is not None)
    
    # 将 cluster_centers 转为 Tensor
    if use_pseudo:
        final_centers = torch.tensor(cluster_centers).float().to(device)
        # 确保 centers 是归一化的 (因为 features 也是归一化的)
        final_centers = F.normalize(final_centers, dim=1)

    for batch_idx, (x0, x1, _, _, _, _, indices) in enumerate(train_loader):
        x0, x1 = x0.to(device), x1.to(device)
        indices = indices.to(device)
        
        # 质量权重
        q1 = (1.0 - quality_tensor_v1[indices])
        q2 = (1.0 - quality_tensor_v2[indices])
        w1 = q1 ** 2
        w2 = q2 ** 2
        
        x0_flat = x0.view(x0.size(0), -1)
        x1_flat = x1.view(x1.size(0), -1)
        
        # Forward
        z0, z1, xr0, xr1 = model(x0_flat, x1_flat)
        
        # --- Loss 1 & 2: Reconstruction & Contrastive (始终存在) ---
        loss_mse = criterion_mse(x0_flat, xr0, w1.view(-1, 1)) + \
                   criterion_mse(x1_flat, xr1, w2.view(-1, 1))
        
        loss_contrast = criterion_ccl(z0, z1, w0=w1, w1=w2)
        
        loss = loss_mse + (args.lamda * loss_contrast)
        
        # --- Loss 3: Pseudo-Label Supervision (仅 Phase 2) ---
        if use_pseudo:
            # 1. 获取当前 batch 的伪标签
            batch_y = pseudo_labels[indices] # (B, )
            
            # 转换为 One-Hot (B, K)
            p_target = F.one_hot(batch_y, num_classes=args.num_classes).float()
            
            # 2. 计算预测 Logits: Feature * Centers.T
            # (B, D) @ (D, K) -> (B, K)
            zp0 = torch.mm(z0, final_centers.T)
            zp1 = torch.mm(z1, final_centers.T)
            
            # 3. Softmax
            pre0 = F.softmax(zp0, dim=1)
            pre1 = F.softmax(zp1, dim=1)
            
            # 4. 计算一致性权重 w
            sim = pre0.mm(pre1.t())
            diag = sim.diag()
            # 归一化权重
            w_consistency = diag / (diag.sum().detach() + 1e-8)
            
            # 5. 计算 Loss
            log_pre0 = torch.log(pre0 + 1e-8)
            log_pre1 = torch.log(pre1 + 1e-8)
            
            pp0 = (p_target * log_pre0).sum(dim=1)
            pp1 = (p_target * log_pre1).sum(dim=1)
            
            # 6. 加权求和: 结合 一致性权重 和 样本质量权重
            batch_weight0 = w_consistency * w1
            batch_weight1 = w_consistency * w2
            
            loss_pl0 = - (batch_weight0 * pp0).mean()
            loss_pl1 = - (batch_weight1 * pp1).mean()
            
            loss = loss + args.beta * (loss_pl0 + loss_pl1)

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
    print(f"==========\nSetting: GPU={args.gpu}, Lamda={args.lamda}, Beta={args.beta}, Epochs={args.epochs}\n==========")

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
    
    # 3. 初始化模型 (无分类头)
    model = SUREfcScene().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 4. 初始化损失
    criterion_mse = WeightedMSELoss().to(device)
    criterion_ccl = WeightedCCL(tau=args.tau).to(device)
    criterions = (criterion_mse, criterion_ccl)

    print("Start Training...")
    best_acc = 0.0
    
    # 全局变量
    pseudo_labels = None
    cluster_centers = None
    
    for epoch in range(args.epochs):
        
        # --- 关键修改：只在 Warmup 结束时生成一次伪标签 ---
        if epoch == args.warmup_epochs:
            print(f"\n[Epoch {epoch}] Warmup finished. Generating Static Weighted Pseudo-Labels...")
            
            # 生成伪标签和中心
            pseudo_labels, cluster_centers = generate_weighted_pseudo_labels(
                model, all_loader, quality_tensor_v1, quality_tensor_v2, 
                args.num_classes, device
            )
            print("[Info] Pseudo-Labels and Centers fixed for Phase 2 training.")

        # --- 训练 ---
        train_one_epoch(
            train_loader, model, criterions, optimizer, epoch, args, 
            quality_tensor_v1, quality_tensor_v2, 
            pseudo_labels=pseudo_labels, cluster_centers=cluster_centers
        )
        
        # --- 评估 (每10个epoch或最后) ---
        if epoch == args.epochs - 1 or (epoch + 1) % 10 == 0:
            model.eval()
            feat_v0_list, feat_v1_list, gt_list = [], [], []
            
            with torch.no_grad():
                for x0, x1, labels, _, _, _, indices in all_loader:
                    x0 = x0.to(device).view(x0.size(0), -1)
                    x1 = x1.to(device).view(x1.size(0), -1)
                    
                    z0, z1, _, _ = model(x0, x1)
                        
                    feat_v0_list.append(z0.cpu().numpy())
                    feat_v1_list.append(z1.cpu().numpy())
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