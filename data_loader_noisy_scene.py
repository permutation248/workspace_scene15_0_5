import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import normalize
import torch
import os
from scipy import sparse

# INTERVALS_v1= [
#     (0.0, 0.1, 0.2), 
#     (0.1, 0.2, 0.4), 
#     (0.2, 0.3, 0.6), 
#     (0.3, 0.4, 0.8), 
#     (0.4, 0.5, 1.0), 
#     (0.5, 1.0, 0.0)  
# ]
# INTERVALS_v2 = [
#     (0.0, 0.4, 0.0), 
#     (0.4, 0.5, 0.2), 
#     (0.5, 0.6, 0.4), 
#     (0.6, 0.7, 0.6), 
#     (0.7, 0.8, 0.8), 
#     (0.8, 0.9, 1.0), 
#     (0.9, 1.0, 0.0)  
# ]

INTERVALS_v1 = [
        (0.0, 0.3, 0.0),   # 前段干净数据
        (0.3, 0.4, 0.2), 
        (0.4, 0.5, 0.4), 
        (0.5, 0.6, 0.6), 
        (0.6, 0.7, 0.8), 
        (0.7, 0.8, 1.0),   # 最后一等分 (纯噪声)
        (0.8, 1.0, 0.0)    # 后段干净数据
    ]

INTERVALS_v2 = [
        # --- 回绕部分 (0.0 - 0.2) ---
        (0.0, 0.1, 0.8),   # 第4段 (原 1.0-1.1)
        (0.1, 0.2, 1.0),   # 第5段 (原 1.1-1.2，纯噪声)
        
        # --- 中间干净部分 ---
        (0.2, 0.7, 0.0), 
        
        # --- 正常顺序部分 (0.7 - 1.0) ---
        (0.7, 0.8, 0.2),   # 第1段 (与 View 1 的 1.0 噪声段重叠)
        (0.8, 0.9, 0.4),   # 第2段
        (0.9, 1.0, 0.6)    # 第3段
    ]

def load_raw_mat(dataset_name, root_path='/home/wupeihan/Multi_View/data/'):
    """
    读取原始 .mat 文件并解析为 view1, view2, label
    """
    path = os.path.join(root_path, dataset_name + '.mat')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    
    mat = sio.loadmat(path)
    data = []
    label = None

    if dataset_name == 'Scene15':
        data.append(mat['X1'])
        data.append(mat['X2'])
        label = np.squeeze(mat['Y'])
    else:
        raise NotImplementedError("目前只支持 Scene15 数据集的渐变噪声加载。")

    # 转换为 float32 和 int64
    view1 = data[0].astype(np.float32)
    view2 = data[1].astype(np.float32)
    label = label.astype(np.int64)

    return view1, view2, label

def inject_gradual_noise_mixed(view_data, intervals, seed):
    """
    [修改版] 根据混合比例注入噪声
    公式: X_new = (1 - alpha) * X_raw + alpha * Noise
    :param view_data: (N, D)
    :param intervals: List of tuples [(ratio_start, ratio_end, alpha), ...]
    :param seed: 随机种子
    :return: noisy_view_data
    """
    rng = np.random.RandomState(seed)
    N, D = view_data.shape
    noisy_data = view_data.copy()
    
    # 统计原始数据的分布特性，以便生成同量级的噪声
    data_mean = np.mean(view_data)
    data_std = np.std(view_data)
    
    print(f"  - Injecting Mixed Noise (Data Mean: {data_mean:.4f}, Std: {data_std:.4f})...")

    for start_ratio, end_ratio, alpha in intervals:
        # alpha 是噪声比例 (contamination level)
        # alpha = 0.0 -> 纯原始数据
        # alpha = 1.0 -> 纯高斯噪声
        
        if alpha <= 0:
            continue
            
        start_idx = int(N * start_ratio)
        end_idx = int(N * end_ratio)
        
        # 防止索引越界
        if end_ratio >= 1.0:
            end_idx = N
            
        if start_idx >= end_idx:
            continue
        
        # 1. 取出原始数据片段
        X_raw_segment = view_data[start_idx:end_idx]
        
        # 2. 生成同形状的纯高斯噪声
        # 使用原始数据的统计特性，确保噪声在数值范围上不是异类
        Noise_segment = rng.normal(loc=data_mean, scale=data_std, size=X_raw_segment.shape)
        
        # 3. 执行线性混合
        # 50%污染 => 0.5 * Raw + 0.5 * Noise
        X_mixed = (1 - alpha) * X_raw_segment + alpha * Noise_segment
        
        noisy_data[start_idx:end_idx] = X_mixed.astype(np.float32)
        
        print(f"    - Range [{start_ratio:.1f}-{end_ratio:.1f}]: Mixed Raw with {alpha*100:.0f}% Gaussian Noise.")

    return noisy_data

class SceneDataset(Dataset):
    def __init__(self, view1, view2, labels):
        # 转置数据以匹配原始代码习惯: (Features, Samples)
        self.view1 = view1.T 
        self.view2 = view2.T
        self.labels = labels
        
        self.mask = np.ones((len(labels), 2), dtype=np.int64)

    def __getitem__(self, index):
        fea0 = torch.from_numpy(self.view1[:, index]).type(torch.FloatTensor)
        fea1 = torch.from_numpy(self.view2[:, index]).type(torch.FloatTensor)
        
        fea0 = fea0.unsqueeze(0)
        fea1 = fea1.unsqueeze(0)

        label = torch.tensor(self.labels[index]).long()
        
        class_labels0 = label
        class_labels1 = label
        mask_val = torch.tensor(self.mask[index]).long()
        
        return fea0, fea1, label, class_labels0, class_labels1, mask_val, index

    def __len__(self):
        return len(self.labels)

def loader_cl_noise(train_bs, dataset_name, NetSeed):
    """
    加载 Scene15 并根据设定添加渐变混合噪声
    """
    print(f"========== Loading Scene15 with MIXED GRADUAL NOISE (Seed: {NetSeed}) ==========")
    
    # 1. 加载原始数据
    view1, view2, labels = load_raw_mat('Scene15')
    N = labels.shape[0]
    
    # 2. 打乱数据 (确保噪声分布随机，不偏向特定类别)
    rng = np.random.RandomState(NetSeed)
    perm_indices = rng.permutation(N)
    
    view1 = view1[perm_indices]
    view2 = view2[perm_indices]
    labels = labels[perm_indices]

    # 4. 注入混合噪声
    print("Processing View 1...")
    view1_noisy = inject_gradual_noise_mixed(view1, INTERVALS_v1, seed=NetSeed + 1)
    
    print("Processing View 2...")
    view2_noisy = inject_gradual_noise_mixed(view2, INTERVALS_v2, seed=NetSeed + 2)

    # 5. 归一化 (混合后再进行归一化，符合特征预处理流程)
    view1_noisy = normalize(view1_noisy)
    view2_noisy = normalize(view2_noisy)
    
    # 6. 构建 Dataset
    train_dataset = SceneDataset(view1_noisy, view2_noisy, labels)
    eval_dataset = SceneDataset(view1_noisy, view2_noisy, labels)

    # 7. 构建 Loader
    train_pair_loader = DataLoader(
        train_dataset,
        batch_size=train_bs,
        shuffle=True, # 训练时打乱
        drop_last=False,
        num_workers=0
    )

    all_loader = DataLoader(
        eval_dataset,
        batch_size=128,
        shuffle=False, # 推理时保持顺序
        num_workers=0
    )

    return train_pair_loader, all_loader, NetSeed

# 兼容性接口
def get_train_loader(x0, x1, c0, c1, train_bs):
    dataset = SceneDataset(x0.T, x1.T, c0)
    loader = DataLoader(
        dataset,
        batch_size=train_bs,
        shuffle=True,
        drop_last=False
    )
    return loader