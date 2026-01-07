import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
from data_loader_noisy_scene import INTERVALS_v1, INTERVALS_v2

# 设置风格
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] # 兼容性字体

def reconstruct_ground_truth(total_samples, seed):
    """
    完全复刻 data_loader 的逻辑来恢复 Ground Truth 噪声标签
    """
    # 1. 模拟打乱索引 (必须与 data_loader 一致)
    rng = np.random.RandomState(seed)
    perm_indices = rng.permutation(total_samples)

    # 3. 生成标签
    gt_v1 = np.zeros(total_samples)
    gt_v2 = np.zeros(total_samples)

    # 这里的逻辑是：data_loader 是先把数据打乱，然后按顺序切分加噪
    # 所以我们先生成“打乱后的顺序”的噪声标签，然后再通过逆排序映射回去（如果需要对应ID）
    # 但由于 npz 里存的已经是按 loader 顺序出来的 loss，我们只需要生成 loader 顺序的 GT 即可
    
    # View 1 GT
    for start, end, alpha in INTERVALS_v1:
        s_idx = int(total_samples * start)
        e_idx = int(total_samples * end)
        if end >= 1.0: e_idx = total_samples
        gt_v1[s_idx:e_idx] = alpha
        
    # View 2 GT
    for start, end, alpha in INTERVALS_v2:
        s_idx = int(total_samples * start)
        e_idx = int(total_samples * end)
        if end >= 1.0: e_idx = total_samples
        gt_v2[s_idx:e_idx] = alpha

    return gt_v1, gt_v2

def normalize_scores(raw_scores, method='minmax'):
    """
    映射函数实验
    """
    if method == 'minmax':
        # 线性归一化
        return (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
    elif method == 'sqrt_minmax':
        # 针对下凹曲线：先开根号拉直，再归一化 (惩罚大误差，提升小误差的区分度)
        # 这通常能把 0.2 的值提上来
        scores = np.sqrt(raw_scores)
        return (scores - scores.min()) / (scores.max() - scores.min())
    elif method == 'log_minmax':
        # 对数变换
        scores = np.log1p(raw_scores)
        return (scores - scores.min()) / (scores.max() - scores.min())
    return raw_scores

def plot_distribution(df, view_name, ax_scatter, ax_line):
    """
    绘制分布散点图和映射曲线图
    """
    # 1. 散点分布图 (Violin + Strip)
    # 这能展示样本的整体分布和离群点
    sns.violinplot(x="True Noise", y="Recon Loss", data=df, ax=ax_scatter, 
                   inner=None, color=".9", linewidth=0)
    sns.stripplot(x="True Noise", y="Recon Loss", data=df, ax=ax_scatter, 
                  size=2, alpha=0.3, jitter=True, palette="viridis")
    
    ax_scatter.set_title(f"{view_name}: Raw Recon Loss Distribution")
    ax_scatter.set_xlabel("True Noise Ratio")
    ax_scatter.set_ylabel("Raw MSE Loss")

    # 2. 映射效果验证图
    # 计算每个噪声等级的平均 loss
    means = df.groupby("True Noise")["Recon Loss"].mean().values
    noise_levels = df.groupby("True Noise")["Recon Loss"].mean().index.values
    
    # 尝试不同的映射
    norm_linear = normalize_scores(means, 'minmax')
    norm_sqrt = normalize_scores(means, 'sqrt_minmax') # 重点关注这个
    
    ax_line.plot(noise_levels, noise_levels, 'k--', label="Ideal Target (y=x)", alpha=0.5)
    ax_line.plot(noise_levels, norm_linear, 'o-', label="Linear Min-Max", color='blue')
    ax_line.plot(noise_levels, norm_sqrt, 's-', label="Sqrt + Min-Max", color='red')
    
    ax_line.set_title(f"{view_name}: Mapping Function Fitting")
    ax_line.set_xlabel("True Noise Ratio")
    ax_line.set_ylabel("Normalized Quality Score (0-1)")
    ax_line.legend()
    ax_line.grid(True, linestyle='--')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1111, type=int)
    args = parser.parse_args()

    # 1. 加载数据
    try:
        data = np.load('recon_errorty.npz')
        loss_v1 = data['recon_loss_v1']
        loss_v2 = data['recon_loss_v2']
        total_samples = len(loss_v1)
        print(f"Loaded recon_errorty.npz with {total_samples} samples.")
    except FileNotFoundError:
        print("Error: recon_errorty.npz not found. Please run generate_quality_matrix.py first.")
        return

    # 2. 恢复 GT 噪声标签 (用于分析，实际无监督训练时不使用)
    gt_v1, gt_v2 = reconstruct_ground_truth(total_samples, args.seed)

    # 3. 构造 DataFrame 方便绘图
    df1 = pd.DataFrame({'Recon Loss': loss_v1, 'True Noise': np.round(gt_v1, 1)})
    df2 = pd.DataFrame({'Recon Loss': loss_v2, 'True Noise': np.round(gt_v2, 1)})

    # 4. 可视化
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # View 1
    plot_distribution(df1, "View 1", axes[0, 0], axes[0, 1])
    # View 2
    plot_distribution(df2, "View 2", axes[1, 0], axes[1, 1])

    plt.tight_layout()
    save_path = 'quality_distribution_analysis.png'
    plt.savefig(save_path)
    print(f"Analysis plot saved to {save_path}")
    
    # 5. 输出简单的统计建议
    print("\n=== Analysis Report ===")
    
    # 简单计算不同映射下，50% 噪声点的得分
    # 获取 View 1 中 0.0 和 1.0 的均值作为 min/max
    v1_min = df1[df1['True Noise'] == 0.0]['Recon Loss'].mean()
    v1_max = df1[df1['True Noise'] == 1.0]['Recon Loss'].mean()
    v1_mid = df1[df1['True Noise'] == 0.6]['Recon Loss'].mean() # 比如看60%的噪声
    
    def get_score(val, min_v, max_v, method):
        if method == 'linear':
            return (val - min_v) / (max_v - min_v)
        if method == 'sqrt':
            return (np.sqrt(val) - np.sqrt(min_v)) / (np.sqrt(max_v) - np.sqrt(min_v))

    print(f"View 1 Check (Target: 0.6 noise -> ~0.6 score):")
    print(f"  - Linear Score: {get_score(v1_mid, v1_min, v1_max, 'linear'):.4f}")
    print(f"  - Sqrt Score:   {get_score(v1_mid, v1_min, v1_max, 'sqrt'):.4f}")
    
    print("\n建议：如果 'Sqrt Score' 更接近 Target，请在后续训练中使用 sqrt(loss) 进行归一化。")

if __name__ == '__main__':
    main()