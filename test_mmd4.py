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
parser = argparse.ArgumentParser(description='MMD-based HSACC with Anchor-Based Weighting')
parser.add_argument('--gpu', default='0', type=str, help='GPU device idx')
parser.add_argument('--epochs', default=200, type=int, help='Total training epochs')
parser.add_argument('--warmup-epochs', default=50, type=int, help='Epochs for Phase 1')
parser.add_argument('--batch-size', default=128, type=int, help='Batch size')
parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
parser.add_argument('--lamda', default=0.1, type=float, help='Weight for Contrastive Loss')
parser.add_argument('--gamma', default=0.1, type=float, help='Weight for MMD Loss')
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
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1)**2).sum(2) 
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)

def MMD(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n = int(source.size()[0])
    m = int(target.size()[0])
    kernels = guassian_kernel_mmd(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:n, :n]
    YY = kernels[n:, n:]
    XY = kernels[:n, n:]
    YX = kernels[n:, :n]
    loss = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)
    return loss

# --- 可视化 ---
def visualize_fusion_distribution(h0, h1, h_fused, epoch, save_dir='vis_results'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    idx = np.arange(min(200, h0.shape[0]))
    h0 = h0[idx].detach().cpu().numpy()
    h1 = h1[idx].detach().cpu().numpy()
    hf = h_fused[idx].detach().cpu().numpy()
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
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
    for i in range(min(20, n)):
        plt.plot([v1_2d[i,0], f_2d[i,0]], [v1_2d[i,1], f_2d[i,1]], 'k-', alpha=0.1)
        plt.plot([v2_2d[i,0], f_2d[i,0]], [v2_2d[i,1], f_2d[i,1]], 'k-', alpha=0.1)
    plt.legend()
    plt.title(f'Epoch {epoch}: Distribution of Views and Fused Representation')
    plt.savefig(f'{save_dir}/fusion_pca_epoch_{epoch}.png')
    plt.close()

# --- 2. 损失函数 (核心修改在 WeightedCCL) ---

class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()
    
    def forward(self, input, target, quality_weights=None):
        loss = (input - target) ** 2
        if quality_weights is not None:
            w = quality_weights.view(-1, 1)
            loss = loss * w
        return loss.mean()

class AnchorWeightedCCL(nn.Module):
    """
    Anchor-Based Weighted Contrastive Loss
    谁是 Anchor，就用谁的质量权重。
    """
    def __init__(self, tau=0.5):
        super(AnchorWeightedCCL, self).__init__()
        self.tau = tau
        self.device = device
    
    def forward(self, x0, x1, w0=None, w1=None):
        # x0, x1: (B, D)
        # w0, w1: (B, 1) or (B,) - 质量权重
        
        # 1. 计算相似度矩阵和 Logits
        out = torch.mm(x0, x1.t()) / self.tau
        labels = torch.arange(out.size(0)).to(self.device)
        loss_func = nn.CrossEntropyLoss(reduction='none')
        
        # 2. 计算基础损失 (不加权)
        # l_i2t: View 1 as Anchor -> View 2 (batch_size,)
        l_i2t = loss_func(out, labels)
        # l_t2i: View 2 as Anchor -> View 1 (batch_size,)
        l_t2i = loss_func(out.t(), labels)
        
        # 3. 施加锚点权重 (Anchor-based Weighting)
        if w0 is not None and w1 is not None:
            # 确保权重是 1D (B,) 以便与 loss (B,) 逐元素相乘
            w0_flat = w0.view(-1)
            w1_flat = w1.view(-1)
            
            # [核心修改]
            # l_i2t 是以 View 1 为 Anchor 的损失，其可靠性由 View 1 样本质量 (w0) 决定
            l_i2t = l_i2t * w0_flat
            
            # l_t2i 是以 View 2 为 Anchor 的损失，其可靠性由 View 2 样本质量 (w1) 决定
            l_t2i = l_t2i * w1_flat
            
        return l_i2t.mean() + l_t2i.mean()

# --- 3. 训练函数 ---
def train_one_epoch(train_loader, backbone, criterions, optimizer, epoch, args, quality_tensor_v1, quality_tensor_v2):
    criterion_mse, criterion_ccl = criterions
    backbone.train()
    
    total_loss = 0
    mmd_loss_val = 0
    start_time = time.time()
    
    use_fusion_mmd = (epoch >= args.warmup_epochs)
    
    do_vis = (epoch % args.log_interval == 0) and use_fusion_mmd
    vis_done = False

    for batch_idx, (x0, x1, labels, _, _, _, indices) in enumerate(train_loader):
        x0, x1 = x0.to(device), x1.to(device)
        indices = indices.to(device)
        
        # 1. 获取质量分数 (保留平方逻辑，因为这对噪声抑制至关重要)
        # 0=Clean, 1=Noisy -> Quality = 1-Noise
        q1_raw = (1.0 - quality_tensor_v1[indices])
        q2_raw = (1.0 - quality_tensor_v2[indices])
        
        # 施加平方，作为"软阈值"过滤掉严重噪声
        mse_w1 = q1_raw ** 2
        mse_w2 = q2_raw ** 2
        
        x0_flat = x0.view(x0.size(0), -1)
        x1_flat = x1.view(x1.size(0), -1)
        
        h0, h1, z0, z1 = backbone(x0_flat, x1_flat)
        
        # --- 重建损失 ---
        if use_fusion_mmd:
            # Phase 2: 加权
            # 注意: mse_w1 需要 reshape 成 (B, 1) 给 MSELoss 用
            loss_mse = criterion_mse(x0_flat, z0, mse_w1.view(-1, 1)) + \
                       criterion_mse(x1_flat, z1, mse_w2.view(-1, 1))
        else:
            loss_mse = criterion_mse(x0_flat, z0, None) + criterion_mse(x1_flat, z1, None)
            
        # --- 对比损失 (锚点加权) ---
        # 传入的 w 可以是 (B,) 或 (B, 1)，Criterion 内部会处理
        loss_contrast = criterion_ccl(h0, h1, w0=mse_w1, w1=mse_w2)
        
        loss = loss_mse + (args.lamda * loss_contrast)
        
        # --- MMD 融合 ---
        if use_fusion_mmd:
            epsilon = 1e-6
            sum_q = q1_raw + q2_raw + epsilon
            w_v1 = (q1_raw / sum_q).view(-1, 1)
            w_v2 = (q2_raw / sum_q).view(-1, 1)
            
            H_fused = w_v1 * h0 + w_v2 * h1
            H_fused = F.normalize(H_fused, dim=1)
            
            if do_vis and not vis_done:
                visualize_fusion_distribution(h0, h1, H_fused, epoch)
                vis_done = True

            loss_mmd_v1 = MMD(h0, H_fused, kernel_mul=2.0, kernel_num=5)
            loss_mmd_v2 = MMD(h1, H_fused, kernel_mul=2.0, kernel_num=5)
            loss = loss + (args.gamma * (loss_mmd_v1 + loss_mmd_v2))
            mmd_loss_val += (loss_mmd_v1.item() + loss_mmd_v2.item())
        
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

    if not os.path.exists('quality_weights.npz'):
        raise FileNotFoundError("Run 'generate_quality_weights.py' first!")
    
    q_data = np.load('quality_weights.npz')
    quality_tensor_v1 = torch.from_numpy(q_data['noise_score_v1']).float().to(device)
    quality_tensor_v2 = torch.from_numpy(q_data['noise_score_v2']).float().to(device)

    train_loader, all_loader, _ = loader_cl_noise(args.batch_size, args.data_name, args.seed)
    
    total_samples = len(all_loader.dataset)
    print(f"Total samples detected: {total_samples}")

    backbone = SUREfcScene().to(device)
    optimizer = torch.optim.Adam(backbone.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    criterion_mse = WeightedMSELoss().to(device)
    # 使用修改后的 Anchor-Based CCL
    criterion_ccl = AnchorWeightedCCL(tau=args.tau).to(device)

    print("Start Training...")
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        train_one_epoch(
            train_loader, backbone, (criterion_mse, criterion_ccl), 
            optimizer, epoch, args, quality_tensor_v1, quality_tensor_v2
        )
        scheduler.step()
        
        if epoch == args.epochs - 1 or (epoch + 1) % 10 == 0:
            backbone.eval()
            feat_v0_list, feat_v1_list, gt_list = [], [], []
            with torch.no_grad():
                for x0, x1, labels, _, _, _, indices in all_loader:
                    x0, x1 = x0.to(device), x1.to(device)
                    x0_flat = x0.view(x0.size(0), -1)
                    x1_flat = x1.view(x1.size(0), -1)
                    h0, h1, _, _ = backbone(x0_flat, x1_flat)
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