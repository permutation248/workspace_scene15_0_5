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
from data_loader_noisy_scene import loader_cl_noise_0_5

# 设置环境
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = "1"

# --- 参数设置 ---
parser = argparse.ArgumentParser(description='MMD-based HSACC Implementation')
parser.add_argument('--gpu', default='0', type=str, help='GPU device idx')
parser.add_argument('--epochs', default=200, type=int, help='Total training epochs')
parser.add_argument('--warmup-epochs', default=50, type=int, help='Epochs for Phase 1')
parser.add_argument('--batch-size', default=128, type=int, help='Batch size')
parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
parser.add_argument('--lamda', default=0.1, type=float, help='Weight for Contrastive Loss')
parser.add_argument('--gamma', default=0.1, type=float, help='Weight for MMD Loss') # 新增 MMD 权重参数
parser.add_argument('--tau', default=0.5, type=float, help='Temperature parameter')
parser.add_argument('--seed', default=1111, type=int, help='Random seed')
parser.add_argument('--log-interval', default=50, type=int, help='Log interval')
parser.add_argument('--data-name', default='Scene15', type=str, help='Dataset name')
parser.add_argument('--noisy-training', default=False, type=bool)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- MMD 相关函数实现 ---

def guassian_kernel_mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    计算高斯核矩阵
    """
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    
    # 1. 计算 pairwise distance (L2范数平方)
    # total: (N, D), total0: (N, 1, D), total1: (1, N, D)
    total0 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1)**2).sum(2) 
    
    # 2. 确定带宽 sigma
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
    
    # 3. 多核高斯混合
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    
    return sum(kernel_val) # 返回叠加后的核矩阵

def MMD(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    计算最大均值差异（MMD）损失
    source: (B, D)
    target: (B, D)
    """
    n = int(source.size()[0])
    m = int(target.size()[0])

    # 获取高斯核矩阵 (n+m, n+m)
    # 这里的 fix_sigma=None 表示自适应计算带宽，或者您可以按照 HSACC 论文设为特定值
    kernels = guassian_kernel_mmd(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    # 获取kernel矩阵的各部分
    XX = kernels[:n, :n]  # Source <-> Source
    YY = kernels[n:, n:]  # Target <-> Target
    XY = kernels[:n, n:]  # Source <-> Target
    YX = kernels[n:, :n]  # Target <-> Source

    # 对每个部分进行归一化并求和
    loss = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)
    
    return loss

# --- [可视化工具] 融合效果可视化 ---
def visualize_fusion_distribution(h0, h1, h_fused, epoch, save_dir='vis_results'):
    """
    可视化 H0, H1 和 H_fused 的分布关系
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 取前200个样本进行可视化，避免过密
    idx = np.arange(min(200, h0.shape[0]))
    
    h0 = h0[idx].detach().cpu().numpy()
    h1 = h1[idx].detach().cpu().numpy()
    hf = h_fused[idx].detach().cpu().numpy()
    
    # 为了在二维平面展示，使用 PCA 降维
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    
    # 拼接所有数据统一降维
    data = np.concatenate([h0, h1, hf], axis=0)
    data_2d = pca.fit_transform(data)
    
    n = len(idx)
    v1_2d = data_2d[:n]
    v2_2d = data_2d[n:2*n]
    f_2d = data_2d[2*n:]
    
    plt.figure(figsize=(8, 8))
    plt.scatter(v1_2d[:, 0], v1_2d[:, 1], c='r', alpha=0.3, label='View 1', marker='x')
    plt.scatter(v2_2d[:, 0], v2_2d[:, 1], c='b', alpha=0.3, label='View 2', marker='x')
    plt.scatter(f_2d[:, 0], f_2d[:, 1], c='g', alpha=0.8, label='Fused H', marker='o')
    
    # 画出连接线，展示融合位置
    for i in range(min(20, n)): # 只画前20个样本的连线
        plt.plot([v1_2d[i,0], f_2d[i,0]], [v1_2d[i,1], f_2d[i,1]], 'k-', alpha=0.1)
        plt.plot([v2_2d[i,0], f_2d[i,0]], [v2_2d[i,1], f_2d[i,1]], 'k-', alpha=0.1)
        
    plt.legend()
    plt.title(f'Epoch {epoch}: Distribution of Views and Fused Representation')
    plt.savefig(f'{save_dir}/fusion_pca_epoch_{epoch}.png')
    plt.close()

# --- 2. 损失函数 ---
class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()
    
    def forward(self, input, target, noise_score_0_5=None):
        loss = (input - target) ** 2
        if noise_score_0_5 is not None:
            w = noise_score_0_5.view(-1, 1)
            loss = loss * w
        return loss.mean()

class WeightedCCL(nn.Module):
    def __init__(self, tau=0.5):
        super(WeightedCCL, self).__init__()
        self.tau = tau
        self.device = device
    
    def forward(self, x0, x1, w0=None, w1=None):
        # 输入应归一化
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
def train_one_epoch(train_loader, backbone, criterions, optimizer, epoch, args, quality_tensor_v1, quality_tensor_v2):
    criterion_mse, criterion_ccl = criterions
    backbone.train()
    
    total_loss = 0
    mmd_loss_val = 0
    start_time = time.time()
    
    # 是否进入第二阶段 (Fusion + MMD)
    use_fusion_mmd = (epoch >= args.warmup_epochs)
    
    do_vis = (epoch % args.log_interval == 0) and use_fusion_mmd
    vis_done = False

    for batch_idx, (x0, x1, labels, _, _, _, indices) in enumerate(train_loader):
        x0, x1 = x0.to(device), x1.to(device)
        indices = indices.to(device)
        
        # 1. 获取当前 Batch 的质量分数 (0~1, 1代表无噪声)
        # quality_tensor 存储的是 noise_score_0_5 (0干净, 1纯噪声)，所以要 1 - noise
        q1_raw = (1.0 - quality_tensor_v1[indices])
        q2_raw = (1.0 - quality_tensor_v2[indices])
        
        # 2. 计算加权 MSE 损失的权重 (平方惩罚低质量)
        mse_w1 = q1_raw ** 2
        mse_w2 = q2_raw ** 2
        
        # Flatten Input
        x0_flat = x0.view(x0.size(0), -1)
        x1_flat = x1.view(x1.size(0), -1)
        
        # Backbone Forward
        # h0, h1: 归一化后的隐层特征 (z in paper)
        # z0, z1: 解码器重构输出 (x_hat in paper)
        h0, h1, z0, z1 = backbone(x0_flat, x1_flat)
        
        # --- 基础损失计算 ---
        if use_fusion_mmd:
             # Phase 2: 重建损失加权
            loss_mse = criterion_mse(x0_flat, z0, mse_w1) + criterion_mse(x1_flat, z1, mse_w2)
        else:
             # Phase 1: 重建损失不加权 (预热稳定性)
            loss_mse = criterion_mse(x0_flat, z0, None) + criterion_mse(x1_flat, z1, None)
            
        # 对比损失始终加权 (Guided by quality)
        loss_contrast = criterion_ccl(h0, h1, w0=mse_w1, w1=mse_w2)
        
        loss = loss_mse + (args.lamda * loss_contrast)
        
        # --- [Phase 2 核心逻辑: 质量加权融合 + MMD] ---
        if use_fusion_mmd:
            # =========================================================================
            # 1. 计算融合权重 (Fusion Weights)
            #    基于质量分数计算相对权重，使得 H 更偏向高质量视图
            #    w1 = q1 / (q1 + q2), w2 = q2 / (q1 + q2)
            #    加上 epsilon 防止除零 (虽然 q 应该 >=0)
            # =========================================================================
            epsilon = 1e-6
            sum_q = q1_raw + q2_raw + epsilon
            w_v1 = (q1_raw / sum_q).view(-1, 1)  # (B, 1)
            w_v2 = (q2_raw / sum_q).view(-1, 1)  # (B, 1)
            
            # =========================================================================
            # 2. 生成高级融合表征 H (High-level Representation)
            #    H = w1 * z1 + w2 * z2
            # =========================================================================
            H_fused = w_v1 * h0 + w_v2 * h1
            
            # 融合后再次归一化，保持在超球面上，与 h0, h1 分布一致
            H_fused = F.normalize(H_fused, dim=1)
            
            # 可视化 (仅在特定Epoch的一个Batch做)
            if do_vis and not vis_done:
                #visualize_fusion_distribution(h0, h1, H_fused, epoch)
                vis_done = True

            # =========================================================================
            # 3. 计算 MMD 损失 (Distribution Alignment)
            #    目标：拉近 视图1分布(h0) 与 融合分布(H) 的距离
            #          拉近 视图2分布(h1) 与 融合分布(H) 的距离
            #    这迫使单视图表征向高质量的融合中心靠拢，实现隐式修复
            # =========================================================================
            # HSACC 论文通常使用多核 MMD
            loss_mmd_v1 = MMD(h0, H_fused, kernel_mul=2.0, kernel_num=5)
            loss_mmd_v2 = MMD(h1, H_fused, kernel_mul=2.0, kernel_num=5)
            
            loss_mmd_total = loss_mmd_v1 + loss_mmd_v2
            
            # 将 MMD 加入总 Loss
            loss = loss + (args.gamma * loss_mmd_total)
            mmd_loss_val += loss_mmd_total.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    epoch_time = time.time() - start_time
    phase_str = "Phase 2 (MMD)" if use_fusion_mmd else "Phase 1 (Warmup)"
    
    if epoch % args.log_interval == 0:
        log_msg = f"Epoch [{epoch}/{args.epochs}] [{phase_str}] Time: {epoch_time:.2f}s | Loss: {total_loss / len(train_loader):.4f}"
        if use_fusion_mmd:
            log_msg += f" | MMD Loss: {mmd_loss_val / len(train_loader):.4f}"
        logging.info(log_msg)

# --- 主程序 ---
def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')
    print(f"==========\nSetting: GPU={args.gpu}, Lamda={args.lamda}, Gamma={args.gamma}, Epochs={args.epochs}\n==========")

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if not os.path.exists('noise_score_0_5.npz'):
        raise FileNotFoundError("Run 'generate_noise_score_0_5.py' first!")
    
    # 载入质量分数
    q_data = np.load('noise_score_0_5.npz')
    quality_tensor_v1 = torch.from_numpy(q_data['noise_score_0_5_v1']).float().to(device)
    quality_tensor_v2 = torch.from_numpy(q_data['noise_score_0_5_v2']).float().to(device)

    # 载入数据
    train_loader, all_loader, _ = loader_cl_noise_0_5(args.batch_size, args.data_name, args.seed)
    
    total_samples = len(all_loader.dataset)
    print(f"Total samples detected: {total_samples}")

    backbone = SUREfcScene().to(device)
    
    # 注意：不再需要 Attention 模块
    
    optimizer = torch.optim.Adam(backbone.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    criterion_mse = WeightedMSELoss().to(device)
    criterion_ccl = WeightedCCL(tau=args.tau)

    print("Start Training...")
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        # 训练一个 Epoch
        train_one_epoch(
            train_loader, backbone, (criterion_mse, criterion_ccl), 
            optimizer, epoch, args, quality_tensor_v1, quality_tensor_v2
        )
        scheduler.step()
        
        # --- 评估 ---
        if epoch == args.epochs - 1 or (epoch + 1) % 10 == 0:
            backbone.eval()
            feat_v0_list, feat_v1_list, gt_list = [], [], []
            
            with torch.no_grad():
                for x0, x1, labels, _, _, _, indices in all_loader:
                    x0, x1 = x0.to(device), x1.to(device)
                    x0_flat = x0.view(x0.size(0), -1)
                    x1_flat = x1.view(x1.size(0), -1)
                    
                    # 只需要 Backbone 提取特征
                    h0, h1, _, _ = backbone(x0_flat, x1_flat)
                    
                    feat_v0_list.append(h0.cpu().numpy())
                    feat_v1_list.append(h1.cpu().numpy())
                    gt_list.append(labels.cpu().numpy())
            
            feat_v0_all = np.concatenate(feat_v0_list)
            feat_v1_all = np.concatenate(feat_v1_list)
            gt_label = np.concatenate(gt_list)
            
            # 使用简单的拼接特征进行聚类评估
            data = [feat_v0_all, feat_v1_all]
            
            ret = Clustering(data, gt_label, random_state=args.seed)
            acc = ret['kmeans']['accuracy']
            nmi = ret['kmeans']['NMI']
            
            if acc > best_acc: best_acc = acc
            logging.info(f"Epoch {epoch} Result: ACC={acc:.4f}, NMI={nmi:.4f} (Best: {best_acc:.4f})")

    print(f"Final Best ACC: {best_acc:.4f}")

if __name__ == '__main__':
    main()