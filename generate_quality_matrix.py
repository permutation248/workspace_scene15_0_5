import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import logging
from data_loader_noisy_scene import loader_cl_noise

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

# --- 1. 严格的随机种子设置 ---
def setup_seed(seed):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    logger.info(f"Random Seed set to: {seed}")

# --- 2. Information Bottleneck AutoEncoder ---
class SimpleAutoEncoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
        super(SimpleAutoEncoder, self).__init__()
        # Encoder: 压缩信息，过滤噪声
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.Tanh(), # Tanh 在处理归一化数据时通常表现较好
            nn.Linear(input_dim * 2, bottleneck_dim), 
            nn.Tanh()  # 限制 bottleneck 范围，增强过滤能力
        )
        # Decoder: 重建
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, input_dim * 2),
            nn.Tanh(),
            nn.Linear(input_dim * 2, input_dim),
            # 最后一层不加激活，因为输入特征可能是 normalize 后的任意分布（虽通常在-1~1或0~1）
            # 但为了通用性，线性输出最稳妥
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

# --- 3. 训练与损失计算函数 ---
def train_and_evaluate(view_idx, train_loader, all_loader, input_dim, bottleneck_dim, device, args):
    """
    针对特定视图训练 AE 并计算重建损失
    """
    logger.info(f"\nTraining AutoEncoder for View {view_idx + 1} (Input: {input_dim} -> Bottleneck: {bottleneck_dim})...")
    
    model = SimpleAutoEncoder(input_dim, bottleneck_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # --- Training Phase ---
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for batch_idx, (x0, x1, _, _, _, _, _) in enumerate(train_loader):
            # 选择对应的视图数据
            x = x0 if view_idx == 0 else x1
            x = x.to(device)
            
            # Scene15 特征可能是 (Batch, 1, Dim) 或 (Batch, Dim)，需展平
            if len(x.shape) > 2:
                x = x.view(x.size(0), -1)

            optimizer.zero_grad()
            x_recon = model(x)
            loss = criterion(x_recon, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 50 == 0:
            logger.info(f"  Epoch [{epoch+1}/{args.epochs}] Loss: {total_loss / len(train_loader):.6f}")

    # --- Evaluation Phase (Calculate Quality/Recon Loss) ---
    logger.info(f"Calculating reconstruction errors for View {view_idx + 1}...")
    model.eval()
    all_recon_losses = []
    
    with torch.no_grad():
        # 使用 all_loader (无 shuffle) 来保证顺序对应
        for x0, x1, _, _, _, _, _ in all_loader:
            x = x0 if view_idx == 0 else x1
            x = x.to(device)
            if len(x.shape) > 2:
                x = x.view(x.size(0), -1)
            
            x_recon = model(x)
            
            # 计算逐样本的 MSE Loss (不求平均，保留 (Batch, ) 维度)
            # loss = (x - x_recon)^2
            loss_batch = torch.mean(torch.pow(x - x_recon, 2), dim=1)
            all_recon_losses.append(loss_batch.cpu().numpy())
            
    return np.concatenate(all_recon_losses, axis=0)

# --- 4. 验证分析与绘图 ---
def analyze_correlation(recon_losses, intervals, view_name, total_samples):
    """
    分析重建损失与噪声比例的关系
    """
    # 构造 Ground Truth 噪声标签
    # 注意：这里必须与 data_loader_noisy_scene.py 的逻辑完全一致
    gt_noise_levels = np.zeros(total_samples)
    
    # intervals: [(start, end, noise_alpha), ...]
    for start_ratio, end_ratio, alpha in intervals:
        start_idx = int(total_samples * start_ratio)
        end_idx = int(total_samples * end_ratio)
        if end_ratio >= 1.0: end_idx = total_samples
        
        gt_noise_levels[start_idx:end_idx] = alpha

    # 统计每个噪声等级下的平均重建损失
    unique_levels = sorted(list(set([i[2] for i in intervals])))
    avg_losses = []
    stds = []
    
    logger.info(f"\n--- Analysis for {view_name} ---")
    logger.info(f"{'Noise Level':<15} | {'Avg Recon Loss':<15} | {'Std Dev':<15}")
    logger.info("-" * 50)
    
    for level in unique_levels:
        # 找到属于该噪声等级的样本索引
        indices = np.where(np.abs(gt_noise_levels - level) < 1e-5)[0]
        if len(indices) == 0: continue
        
        losses = recon_losses[indices]
        avg = np.mean(losses)
        std = np.std(losses)
        
        avg_losses.append(avg)
        stds.append(std)
        logger.info(f"{level*100:<5.0f}%          | {avg:<15.6f} | {std:<15.6f}")
        
    return unique_levels, avg_losses, stds

# --- 主程序 ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--seed', default=1111, type=int)
    parser.add_argument('--epochs', default=200, type=int, help="AE training epochs")
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--batch-size', default=128, type=int)
    args = parser.parse_args()

    # 1. 设置环境
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    setup_seed(args.seed)

    # 2. 加载数据 (使用 Noisy Loader)
    # 注意：参数保持与 test_roll2.py 一致，确保加载相同的数据分布
    train_loader, all_loader, _ = loader_cl_noise(
        train_bs=args.batch_size, 
        dataset_name='Scene15', NetSeed=args.seed
    )
    
    # 获取数据维度
    # Scene15: View1 (20), View2 (59)
    # 通过 loader 获取一个 batch 来确认维度
    sample_x0, sample_x1, _, _, _, _, _ = next(iter(all_loader))
    dim_v1 = sample_x0.shape[-1] if len(sample_x0.shape) == 2 else sample_x0.shape[-1] * sample_x0.shape[-2]
    dim_v2 = sample_x1.shape[-1] if len(sample_x1.shape) == 2 else sample_x1.shape[-1] * sample_x1.shape[-2]
    # 如果是 (Batch, 1, Dim)，需要 squeeze
    dim_v1 = sample_x0.view(sample_x0.size(0), -1).shape[1]
    dim_v2 = sample_x1.view(sample_x1.size(0), -1).shape[1]

    logger.info(f"Data Loaded. Dim V1: {dim_v1}, Dim V2: {dim_v2}")
    total_samples = len(all_loader.dataset)

    # 3. 训练自编码器并获取质量评分 (重建损失)
    # 设定 Bottleneck: 强制压缩以过滤噪声
    # V1 (20) -> 10, V2 (59) -> 20
    recon_loss_v1 = train_and_evaluate(0, train_loader, all_loader, dim_v1, 10, device, args)
    recon_loss_v2 = train_and_evaluate(1, train_loader, all_loader, dim_v2, 20, device, args)

    # 4. 验证分析
    # 硬编码噪声区间 (需与 data_loader_noisy_scene.py 保持完全一致)
    intervals_v1 = [
        (0.0, 0.1, 0.2), (0.1, 0.2, 0.4), (0.2, 0.3, 0.6), 
        (0.3, 0.4, 0.8), (0.4, 0.5, 1.0), (0.5, 1.0, 0.0)
    ]
    intervals_v2 = [
        (0.0, 0.4, 0.0), (0.4, 0.5, 0.2), (0.5, 0.6, 0.4), 
        (0.6, 0.7, 0.6), (0.7, 0.8, 0.8), (0.8, 0.9, 1.0), (0.9, 1.0, 0.0)
    ]

    levels_v1, losses_v1, _ = analyze_correlation(recon_loss_v1, intervals_v1, "View 1", total_samples)
    levels_v2, losses_v2, _ = analyze_correlation(recon_loss_v2, intervals_v2, "View 2", total_samples)

    # 5. 可视化
    plt.figure(figsize=(12, 5))
    
    # Plot View 1
    plt.subplot(1, 2, 1)
    plt.plot(levels_v1, losses_v1, 'o-', color='b', linewidth=2, markersize=8)
    plt.title(f'View 1: Noise Level vs Recon Loss\n(Bottleneck: {dim_v1}->10)')
    plt.xlabel('Noise Level (Alpha)')
    plt.ylabel('MSE Reconstruction Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot View 2
    plt.subplot(1, 2, 2)
    plt.plot(levels_v2, losses_v2, 's-', color='r', linewidth=2, markersize=8)
    plt.title(f'View 2: Noise Level vs Recon Loss\n(Bottleneck: {dim_v2}->20)')
    plt.xlabel('Noise Level (Alpha)')
    plt.ylabel('MSE Reconstruction Loss')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plot_path = 'noise_recon_analysis.png'
    plt.savefig(plot_path)
    logger.info(f"\nAnalysis plot saved to {plot_path}")

    # 6. 保存质量矩阵 (用于后续训练脚本)
    # 保存为 (N, 2) 的矩阵，或者分开保存
    # 这里我们保存为字典形式，方便读取
    save_path = 'corruption_rate.npz'
    np.savez(save_path, 
             recon_loss_v1=recon_loss_v1, 
             recon_loss_v2=recon_loss_v2,
             sample_indices=np.arange(total_samples)) # 辅助信息
    
    logger.info(f"Quality matrix saved to {save_path}")
    logger.info("Verification Complete: Higher noise should correlate with higher recon loss.")

if __name__ == '__main__':
    main()