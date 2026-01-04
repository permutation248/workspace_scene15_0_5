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
parser = argparse.ArgumentParser(description='Quality-Aware Pseudo-Label Learning (ROLL Style)')
parser.add_argument('--gpu', default='0', type=str, help='GPU device idx')
parser.add_argument('--epochs', default=200, type=int, help='Total training epochs')
parser.add_argument('--warmup-epochs', default=50, type=int, help='Epochs for Phase 1 (Warmup)')
parser.add_argument('--batch-size', default=128, type=int, help='Batch size')
parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
parser.add_argument('--lamda', default=0.1, type=float, help='Weight for Contrastive Loss')
parser.add_argument('--beta', default=0.1, type=float, help='Weight for Pseudo-Label Supervised Loss')
parser.add_argument('--alpha', default=0.1, type=float, help='Weight for Robust Contrastive Loss')
parser.add_argument('--tau', default=0.5, type=float, help='Temperature parameter')
parser.add_argument('--q', default=0.7, type=float, help='Parameter q for Robust Contrastive Loss') # 新增 q 参数
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

# --- 工具函数 ---
def ind2vec(ind, N=None, C=None):
    ind = np.asarray(ind)
    if ind.ndim == 1:
        ind = ind[:, np.newaxis]
    if N is None: N = ind.shape[0]
    if C is None: C = args.num_classes
    enc = np.zeros((N, C), dtype=np.float32)
    enc[np.arange(N), ind.flatten()] = 1.0
    return enc

# --- 2. 伪标签生成 (保持之前的逻辑，使用拼接特征) ---
def generate_weighted_pseudo_labels(model, data_loader, quality_v1, quality_v2, num_classes, device):
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
    
    # 质量分数处理
    q1_all = 1.0 - quality_v1.cpu().numpy()
    q2_all = 1.0 - quality_v2.cpu().numpy()
    q1_sorted = np.zeros(N)
    q2_sorted = np.zeros(N)
    q1_sorted[all_indices] = q1_all[all_indices]
    q2_sorted[all_indices] = q2_all[all_indices]
    
    # 拼接并加权聚类
    pool_features = np.concatenate([features_v1, features_v2], axis=0)
    pool_weights = np.concatenate([q1_sorted, q2_sorted], axis=0)
    
    print(f"  > Running Weighted K-Means on {2*N} samples...")
    kmeans = KMeans(n_clusters=num_classes, n_init=20, random_state=args.seed)
    kmeans.fit(pool_features, sample_weight=pool_weights)
    
    cluster_centers = kmeans.cluster_centers_
    
    # 统一预测
    print("  > Predicting labels using pooled features...")
    all_labels = kmeans.predict(pool_features)
    labels_v1 = all_labels[:N]
    labels_v2 = all_labels[N:]
    
    # 冲突消解
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

# --- 4. 训练核心函数 (Refined with ROLL logic) ---
def train_one_epoch(train_loader, model, criterions, optimizer, epoch, args, 
                    quality_tensor_v1, quality_tensor_v2, 
                    pseudo_labels=None, cluster_centers=None):
    
    criterion_mse, criterion_ccl = criterions
    model.train()
    
    total_loss = 0
    start_time = time.time()
    
    # 判断是否处于 Phase 2
    use_pseudo = (epoch >= args.warmup_epochs) and (pseudo_labels is not None) and (cluster_centers is not None)
    
    if use_pseudo:
        final_centers = torch.tensor(cluster_centers).float().to(device)
        # 归一化中心，保证内积计算一致性
        final_centers = F.normalize(final_centers, dim=1)

    for batch_idx, (x0, x1, _, _, _, _, indices) in enumerate(train_loader):
        x0, x1 = x0.to(device), x1.to(device)
        indices = indices.to(device)
        
        # 1. 质量权重 (仅用于 MSE 重建损失)
        # 我们希望噪声大的样本不要主导重建
        q1 = (1.0 - quality_tensor_v1[indices])
        q2 = (1.0 - quality_tensor_v2[indices])
        w1 = q1 ** 2
        w2 = q2 ** 2
        
        x0_flat = x0.view(x0.size(0), -1)
        x1_flat = x1.view(x1.size(0), -1)
        
        # Forward
        # z0, z1 是归一化的特征
        z0, z1, xr0, xr1 = model(x0_flat, x1_flat)
        
        # --- Loss 1: Reconstruction (Weighted by Quality) ---
        loss_mse = criterion_mse(x0_flat, xr0, w1.view(-1, 1)) + \
                   criterion_mse(x1_flat, xr1, w2.view(-1, 1))
        
        if not use_pseudo:
            # === Phase 1: Warmup ===
            # 使用普通的加权对比损失
            loss_contrast = criterion_ccl(z0, z1, w0=w1, w1=w2)
            loss = loss_mse + (args.lamda * loss_contrast)
        
        else:
            # === Phase 2: Pseudo-Label Supervision (ROLL Logic) ===
            
            # A. 准备伪标签 target
            batch_y = pseudo_labels[indices]
            p_target = F.one_hot(batch_y, num_classes=args.num_classes).float() # (B, K)

            # B. 计算 Logits 和 Probability
            # z0: (B, D), centers: (K, D) -> zp0: (B, K)
            zp0 = torch.mm(z0, final_centers.T)
            zp1 = torch.mm(z1, final_centers.T)
            
            # 使用 Softmax 归一化 (dim=1)
            pre0 = F.softmax(zp0, dim=1)
            pre1 = F.softmax(zp1, dim=1)
            
            # C. 计算一致性权重 w (参照 ROLL Snippet)
            # 计算样本间的一致性相似度
            sim = pre0.mm(pre1.t()) # (B, B)
            diag = sim.diag()       # (B, ) 对角线即为该样本 View1 和 View2 预测的一致性
            
            # 归一化 w，使其作为本 Batch 内的相对权重
            # 注意：这里 detach，不让梯度通过权重反传
            w_consistency = diag / (diag.sum().detach() + 1e-8)
            
            # D. 计算伪标签损失 (Pseudo-Label Loss)
            # 重点：此处只使用 w_consistency，**不再**乘 w1/w2 (quality)
            # 这样坏样本也能接收到基于一致性的监督信号进行修正
            
            log_pre0 = torch.log(pre0 + 1e-8)
            log_pre1 = torch.log(pre1 + 1e-8)
            
            # p_target * log_pre -> 选出目标类的 log 概率
            # sum(dim=1) 得到每个样本的 loss 值 (负数)
            pp0 = (p_target * log_pre0).sum(dim=1)
            pp1 = (p_target * log_pre1).sum(dim=1)
            
            # ROLL snippet: lossa = - (args.beta * w * pp0).mean()
            loss_pl0 = - (w_consistency * pp0).mean()
            loss_pl1 = - (w_consistency * pp1).mean()
            
            loss_pl = args.beta * (loss_pl0 + loss_pl1)
            
            # E. Robust Contrastive Loss (RCL) (参照 ROLL Snippet)
            # 替代普通的 Contrastive Loss
            
            # 计算特征相似度
            cos = z0.mm(z1.t())
            sim_c = (cos / args.tau).exp()
            pos = sim_c.diag() # 正样本对的相似度
            q = args.q
            
            # RCL 公式: sum( ((1-q)*sum(sim)^q - pos^q) / q )
            # sim_c.sum(1) 是对比损失分母中的项
            loss_rcl = (((1 - q) * (sim_c.sum(1))**q - pos ** q) / q).sum().mean()
            
            loss_contrast_robust = args.alpha * loss_rcl
            
            # F. 总损失
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
    print(f"==========\nSetting: GPU={args.gpu}, Lamda={args.lamda}, Beta={args.beta}, Alpha={args.alpha}, Q={args.q}, Epochs={args.epochs}\n==========")

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
    
    # 4. 初始化损失 (Phase 1 用)
    criterion_mse = WeightedMSELoss().to(device)
    criterion_ccl = WeightedCCL(tau=args.tau).to(device)
    criterions = (criterion_mse, criterion_ccl)

    print("Start Training...")
    best_acc = 0.0
    
    pseudo_labels = None
    cluster_centers = None
    
    for epoch in range(args.epochs):
        
        # --- Warmup 结束时生成一次伪标签 ---
        if epoch == args.warmup_epochs:
            print(f"\n[Epoch {epoch}] Warmup finished. Generating Static Weighted Pseudo-Labels...")
            
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
        
        # --- 评估 ---
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