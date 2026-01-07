import argparse
import time
import torch
import torch.nn as nn
import logging
import os
import sys
import numpy as np
import warnings

# 引入必要的模块 (请确保这些文件在同一目录下)
from models import SUREfcScene
from Clustering import Clustering
from sure_inference import both_infer
from data_loader_noisy_scene import loader_cl_noise


# 设置环境
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = "2"

# --- 参数设置 ---
parser = argparse.ArgumentParser(description='Scene15 Clean Training (Recon + Contrastive)')

# 核心训练参数
parser.add_argument('--gpu', default='0', type=str, help='GPU device idx')
parser.add_argument('--epochs', default=200, type=int, help='Training epochs')
parser.add_argument('--batch-size', default=128, type=int, help='Batch size')
parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate (Default for Scene15)')
parser.add_argument('--lamda', default=0.1, type=float, help='Weight for Contrastive Loss')

# 固定/默认参数 (保持与源代码逻辑一致)
parser.add_argument('--tau', default=0.5, type=float, help='Temperature parameter')
parser.add_argument('--seed', default=1111, type=int, help='Random seed')
parser.add_argument('--log-interval', default=50, type=int, help='Log interval')
parser.add_argument('--data-name', default='Scene15', type=str, help='Dataset name')

args = parser.parse_args()

# 设置 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 损失函数定义 (CCL) ---
class CCL(nn.Module):
    def __init__(self, tau=0.5):
        super(CCL, self).__init__()
        self.tau = tau
        self.device = device
    
    def forward(self, x0, x1):
        # 归一化特征
        x0 = torch.nn.functional.normalize(x0, dim=1)
        x1 = torch.nn.functional.normalize(x1, dim=1)
        
        # 计算相似度矩阵
        out = torch.mm(x0, x1.t()) / self.tau
        
        # 定义 InfoNCE 损失 (简化版，仅针对干净数据)
        # 对角线为正样本，其余为负样本
        loss_func = nn.CrossEntropyLoss()
        labels = torch.arange(out.size(0)).to(self.device)
        
        loss_i2t = loss_func(out, labels)
        loss_t2i = loss_func(out.t(), labels)
        
        return loss_i2t + loss_t2i

# --- 训练核心函数 ---
def train_one_epoch(train_loader, model, criterions, optimizer, epoch, args):
    model.train()
    criterion_mse, criterion_ccl = criterions
    total_loss = 0
    
    start_time = time.time()
    
    for batch_idx, (x0, x1, labels, _, _, _, _) in enumerate(train_loader):
        x0, x1 = x0.to(device), x1.to(device)
        
        # Flatten input for Linear Model (Scene15)
        x0 = x0.view(x0.size()[0], -1)
        x1 = x1.view(x1.size()[0], -1)
        
        # Forward pass
        # model 返回: z0, z1 (hidden features), xr0, xr1 (reconstructed)
        z0, z1, xr0, xr1 = model(x0, x1)
        
        # 1. Reconstruction Loss (MSE)
        loss_mse = criterion_mse(x0, xr0) + criterion_mse(x1, xr1)
        
        # 2. Contrastive Loss (CCL)
        # 原始代码逻辑: loss_cl=lambda x1,x2,M: -(M * (x1.mm(x2.t())/args.tau).softmax(1).log()).sum(1).mean()
        # 这里使用等价的简化 CCL 实现，或者手动实现原版逻辑：
        
        # ---------------- 原版逻辑复现 ----------------
        loss_cl = lambda x_i, x_j, M: -(M * (x_i.mm(x_j.t()) / args.tau).softmax(1).log()).sum(1).mean()
        I = torch.eye(x0.size(0)).to(device)
        loss_contrast = loss_cl(z0, z1, I)
        # -------------------------------------------

        # Total Loss
        loss = loss_mse + (args.lamda * loss_contrast)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    epoch_time = time.time() - start_time
    if epoch % args.log_interval == 0:
         logging.info(f"Epoch [{epoch}/{args.epochs}] Time: {epoch_time:.2f}s | Loss: {total_loss / len(train_loader):.4f}")

# --- 主程序 ---
def main():
    # 1. 设置 Logger
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')
    print(f"==========\nSetting: GPU={args.gpu}, Lamda={args.lamda}, Epochs={args.epochs}, Batch={args.batch_size}\n==========")

    # 2. 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # 3. 加载数据 (使用重构后的 clean loader)
    # 注意：loader_cl 内部会忽略 neg_prop 等噪声参数
    train_loader, all_loader, _ = loader_cl_noise(
        args.batch_size, args.data_name, args.seed
    )

    # 4. 初始化模型 (针对 Scene15)
    model = SUREfcScene().to(device)
    
    # 5. 优化器与损失
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion_mse = nn.MSELoss().to(device)
    criterion_ccl = CCL(tau=args.tau) # 虽然用作类占位，实际计算用了上面的 lambda 表达式

    # 6. 训练循环
    print("Start Training (Reconstruction + Contrastive)...")
    
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        train_one_epoch(train_loader, model, [criterion_mse, criterion_ccl], optimizer, epoch, args)
        
        # 每次 epoch 结束都进行测试 (或者你可以设置 if epoch == args.epochs - 1 只测最后一次)
        if epoch == args.epochs - 1 or (epoch + 1) % 10 == 0:
            # 推理
            v0, v1, gt_label = both_infer(model, device, all_loader, setting=2, cri=None, return_data=False)
            data = [v0, v1]
            
            # 聚类评估
            ret = Clustering(data, gt_label, random_state=args.seed)
            acc = ret['kmeans']['accuracy']
            nmi = ret['kmeans']['NMI']
            ari = ret['kmeans']['ARI']
            
            if acc > best_acc:
                best_acc = acc
            
            logging.info(f"Epoch {epoch} Result: ACC={acc:.4f}, NMI={nmi:.4f}, ARI={ari:.4f} (Best ACC: {best_acc:.4f})")

    print(f"Training Finished. Final Best ACC: {best_acc:.4f}")

if __name__ == '__main__':
    main()