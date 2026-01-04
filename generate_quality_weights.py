import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import argparse

# 设置绘图风格
sns.set_theme(style="whitegrid")

def reconstruct_ground_truth(total_samples, seed=1111):
    """
    重构真实的噪声标签以进行验证
    (必须与 data_loader_noisy_scene.py 的逻辑保持完全一致)
    """
    # 1. 模拟打乱索引 (虽然这里主要是按区间生成，但保持逻辑完整性)
    # 注意：npz里的数据是按loader顺序存的，所以我们直接按区间生成即可
    
    gt_v1 = np.zeros(total_samples)
    gt_v2 = np.zeros(total_samples)

    # View 1 区间配置
    intervals_v1 = [
        (0.0, 0.1, 0.2), (0.1, 0.2, 0.4), (0.2, 0.3, 0.6), 
        (0.3, 0.4, 0.8), (0.4, 0.5, 1.0), (0.5, 1.0, 0.0)
    ]
    # View 2 区间配置
    intervals_v2 = [
        (0.0, 0.4, 0.0), (0.4, 0.5, 0.2), (0.5, 0.6, 0.4), 
        (0.6, 0.7, 0.6), (0.7, 0.8, 0.8), (0.8, 0.9, 1.0), (0.9, 1.0, 0.0)
    ]

    for start, end, alpha in intervals_v1:
        s = int(total_samples * start)
        e = int(total_samples * end)
        if end >= 1.0: e = total_samples
        gt_v1[s:e] = alpha
        
    for start, end, alpha in intervals_v2:
        s = int(total_samples * start)
        e = int(total_samples * end)
        if end >= 1.0: e = total_samples
        gt_v2[s:e] = alpha

    return gt_v1, gt_v2

def visualize_mapping(score_v1, gt_v1, score_v2, gt_v2, save_path):
    """
    可视化：归一化后的评分 vs 真实噪声比例
    """
    print(f"\nGenerating visualization plot to {save_path}...")
    
    df1 = pd.DataFrame({'Score': score_v1, 'True Noise': np.round(gt_v1, 1), 'View': 'View 1'})
    df2 = pd.DataFrame({'Score': score_v2, 'True Noise': np.round(gt_v2, 1), 'View': 'View 2'})
    df = pd.concat([df1, df2])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 绘制两个视图的子图
    for i, view_name in enumerate(['View 1', 'View 2']):
        ax = axes[i]
        curr_df = df[df['View'] == view_name]
        
        # 1. 绘制分布 (Violin plot)
        sns.violinplot(x="True Noise", y="Score", data=curr_df, ax=ax, 
                       inner=None, color=".9", linewidth=0)
        
        # 2. 绘制散点 (Strip plot) - 降采样以避免过密
        # 如果样本量太大，只画一部分
        if len(curr_df) > 2000:
            plot_df = curr_df.sample(2000, random_state=42)
        else:
            plot_df = curr_df
        sns.stripplot(x="True Noise", y="Score", data=plot_df, ax=ax, 
                      size=2, alpha=0.3, jitter=True, palette="viridis")

        # 3. 绘制均值连线 (验证线性度)
        means = curr_df.groupby("True Noise")["Score"].mean()
        ax.plot(range(len(means)), means.values, 'r-o', linewidth=2, label='Mean Score')
        
        # 绘制理想对角线 (辅助线)
        # 注意：x轴是类别索引，需要转换
        # 理想情况下 0.0 -> 0.0, 0.2 -> 0.2
        # 获取x轴标签对应的数值
        x_labels = sorted(curr_df["True Noise"].unique())
        # 在类别坐标轴上画 y=x 是困难的，我们简单画一条连接 (0,0) 到 (1,1) 的虚线作为参考
        # 这里直接画均值线与 X 轴标签值的对比更直观
        
        ax.set_title(f"{view_name}: Normalized Noise Score vs True Noise")
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Calculated Noise Score (0=Clean, 1=Noisy)")
        ax.set_xlabel("True Noise Ratio (Alpha)")
        ax.legend()
        ax.grid(True, linestyle='--')

    plt.tight_layout()
    plt.savefig(save_path)
    print("Visualization saved.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1111, type=int, help="Seed used for data generation")
    args = parser.parse_args()

    # --- 配置路径 ---
    input_path = 'corruption_rate.npz'
    output_path = 'quality_weights.npz'
    plot_path = 'quality_score_verification.png'

    print(f"[{sys.argv[0]}] Starting generation of Quality/Noise Weights...")

    # 1. 加载原始重建误差
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        return

    data = np.load(input_path)
    recon_error_v1 = data['recon_loss_v1']
    recon_error_v2 = data['recon_loss_v2']
    total_samples = len(recon_error_v1)
    
    print(f"Loaded raw reconstruction errors for {total_samples} samples.")

    # 2. 定义映射函数 (Sqrt + MinMax)
    def map_error_to_score(errors, view_name):
        sqrt_errors = np.sqrt(errors)
        min_val = np.min(sqrt_errors)
        max_val = np.max(sqrt_errors)
        
        if max_val - min_val < 1e-9:
            print(f"Warning: {view_name} constant error.")
            return np.zeros_like(errors)
            
        normalized_score = (sqrt_errors - min_val) / (max_val - min_val)
        
        print(f"\n--- {view_name} Mapping Statistics ---")
        print(f"  Range: [{np.min(normalized_score):.4f}, {np.max(normalized_score):.4f}]")
        return normalized_score

    # 3. 执行映射
    noise_score_v1 = map_error_to_score(recon_error_v1, "View 1")
    noise_score_v2 = map_error_to_score(recon_error_v2, "View 2")

    # 4. 生成 Ground Truth 并可视化
    gt_v1, gt_v2 = reconstruct_ground_truth(total_samples, args.seed)
    visualize_mapping(noise_score_v1, gt_v1, noise_score_v2, gt_v2, plot_path)

    # 5. 保存最终权重
    np.savez(output_path, 
             noise_score_v1=noise_score_v1, 
             noise_score_v2=noise_score_v2)
    
    print(f"\n[Success] Final weights saved to: {output_path}")

if __name__ == '__main__':
    main()