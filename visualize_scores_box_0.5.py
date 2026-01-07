import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import argparse

# 设置绘图风格
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 兼容性字体

def reconstruct_ground_truth(total_samples):
    """
    根据提供的 data_loader_noisy_scene.py 逻辑重建 GT 噪声标签
    注意：这里假设 .npz 中的数据顺序与 data_loader 输出的顺序一致
    """
    gt_v1 = np.zeros(total_samples)
    gt_v2 = np.zeros(total_samples)

    # === View 1 噪声区间配置 (来自你的代码) ===
    # (Start Ratio, End Ratio, Alpha)
    intervals_v1_0_5 = [
        (0.0, 0.1, 0.2), 
        (0.1, 0.2, 0.4), 
        (0.2, 0.3, 0.6), 
        (0.3, 0.4, 0.8), 
        (0.4, 0.5, 1.0), 
        (0.5, 1.0, 0.0)  
    ]
    
    # === View 2 噪声区间配置 (来自你的代码) ===
    intervals_v2_0_5 = [
        (0.0, 0.4, 0.0), 
        (0.4, 0.5, 0.2), 
        (0.5, 0.6, 0.4), 
        (0.6, 0.7, 0.6), 
        (0.7, 0.8, 0.8), 
        (0.8, 0.9, 1.0), 
        (0.9, 1.0, 0.0)  
    ]

    # 生成 View 1 标签
    for start, end, alpha in intervals_v1_0_5:
        s_idx = int(total_samples * start)
        e_idx = int(total_samples * end)
        if end >= 1.0: e_idx = total_samples
        if s_idx < e_idx:
            gt_v1[s_idx:e_idx] = alpha
            
    # 生成 View 2 标签
    for start, end, alpha in intervals_v2_0_5:
        s_idx = int(total_samples * start)
        e_idx = int(total_samples * end)
        if end >= 1.0: e_idx = total_samples
        if s_idx < e_idx:
            gt_v2[s_idx:e_idx] = alpha

    return gt_v1, gt_v2

def plot_metric_vs_noise(metrics_v1, gt_v1, metrics_v2, gt_v2, metric_name, save_name):
    """
    绘制箱线图：Metric (Y轴) vs True Noise (X轴)
    """
    # 构造 DataFrame
    df1 = pd.DataFrame({metric_name: metrics_v1, 'True Noise Ratio': np.round(gt_v1, 1), 'View': 'View 1'})
    df2 = pd.DataFrame({metric_name: metrics_v2, 'True Noise Ratio': np.round(gt_v2, 1), 'View': 'View 2'})
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # View 1 Plot
    sns.boxplot(x='True Noise Ratio', y=metric_name, data=df1, ax=axes[0], palette="Blues")
    axes[0].set_title(f'View 1: {metric_name} vs Noise Level')
    axes[0].set_xlabel('True Noise Ratio (Alpha)')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # View 2 Plot
    sns.boxplot(x='True Noise Ratio', y=metric_name, data=df2, ax=axes[1], palette="Reds")
    axes[1].set_title(f'View 2: {metric_name} vs Noise Level')
    axes[1].set_xlabel('True Noise Ratio (Alpha)')
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_name)
    print(f"Plot saved to: {save_name}")
    plt.close()

def load_npz_data(filepath, key_prefix):
    """
    灵活加载 npz 数据，尝试自动推断 key
    """
    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} not found.")
        return None, None
        
    data = np.load(filepath)
    keys = list(data.keys())
    # print(f"Loaded {filepath}, keys: {keys}")
    
    # 尝试常见的 key 命名模式
    v1, v2 = None, None
    
    # 模式 1: noise_score_0_5_v1, noise_score_0_5_v2
    if f'{key_prefix}_v1' in keys:
        v1 = data[f'{key_prefix}_v1']
        v2 = data[f'{key_prefix}_v2']
    # 模式 2: recon_loss_v1, recon_loss_v2
    elif 'recon_loss_v1' in keys and 'loss' in key_prefix:
        v1 = data['recon_loss_v1']
        v2 = data['recon_loss_v2']
    # 模式 3: 暴力匹配第一个和第二个 key
    else:
        # 排除 sample_indices 等辅助 key
        valid_keys = [k for k in keys if 'v1' in k or 'v2' in k or 'view' in k]
        valid_keys.sort()
        if len(valid_keys) >= 2:
            print(f"Auto-detecting keys for {filepath}: {valid_keys[0]}, {valid_keys[1]}")
            v1 = data[valid_keys[0]]
            v2 = data[valid_keys[1]]
            
    return v1, v2

def main():
    # 文件路径配置
    recon_file = 'recon_error_0_5.npz'
    score_file = 'noise_score_0_5.npz'
    
    # 1. 加载 Recon Error
    recon_v1, recon_v2 = load_npz_data(recon_file, 'recon_loss')
    # 2. 加载 Noise Score
    score_v1, score_v2 = load_npz_data(score_file, 'noise_score_0_5')
    
    if recon_v1 is None and score_v1 is None:
        print("No data found. Please check file names.")
        return

    # 获取样本总数 (以读取到的数据为准)
    total_samples = len(recon_v1) if recon_v1 is not None else len(score_v1)
    print(f"Total samples detected: {total_samples}")

    # 3. 重建 Ground Truth 标签
    gt_v1, gt_v2 = reconstruct_ground_truth(total_samples)

    # 4. 绘制 Recon Error 箱线图
    if recon_v1 is not None:
        plot_metric_vs_noise(recon_v1, gt_v1, recon_v2, gt_v2, 
                             metric_name="Recon Error (MSE)", 
                             save_name="vis_box_recon_error_0_5.png")

    # 5. 绘制 Noise Score 箱线图
    if score_v1 is not None:
        plot_metric_vs_noise(score_v1, gt_v1, score_v2, gt_v2, 
                             metric_name="Noise Score (Normalized)", 
                             save_name="vis_box_noise_score_0_5.png")

if __name__ == "__main__":
    main()
    