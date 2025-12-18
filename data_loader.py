import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import normalize
import torch
import os
from scipy import sparse

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

    # --- 1. Scene15 ---
    if dataset_name == 'Scene15':
        data.append(mat['X1'])
        data.append(mat['X2'])
        label = np.squeeze(mat['Y'])

    # --- 2. LandUse_21 ---
    elif dataset_name == 'LandUse_21':
        label = np.squeeze(mat['Y'])
        # 处理稀疏矩阵逻辑
        try:
            # 尝试按照 cell array 读取 (根据您之前的代码逻辑)
            data.append(sparse.csr_matrix(mat['X'][0, 1]).A) # View 1
            data.append(sparse.csr_matrix(mat['X'][0, 2]).A) # View 2
        except:
            # 备用方案：直接读取 X1, X2
            if 'X1' in mat and 'X2' in mat:
                data.append(mat['X1'])
                data.append(mat['X2'])
            else:
                raise ValueError("LandUse_21 format not recognized in .mat file")

    # --- 3. NUS-WIDE & XMedia (Deep Features) ---
    elif 'nuswide' in dataset_name or 'xmedia' in dataset_name:
        # 支持 nuswide_deep_2_view, xmedia_deep_2_view
        data.append(mat['Img'])
        data.append(mat['Txt'])
        label = np.squeeze(mat['label'])

    # --- 4. Reuters ---
    elif 'Reuters' in dataset_name:
        # Reuters_dim10
        # 原始代码进行了归一化和拼接
        data.append(normalize(np.vstack((mat['x_train'][0], mat['x_test'][0]))))
        data.append(normalize(np.vstack((mat['x_train'][1], mat['x_test'][1]))))
        label = np.squeeze(np.hstack((mat['y_train'], mat['y_test'])))

    # --- 5. CUB ---
    elif 'CUB' in dataset_name:
        data.append(mat['X1'])
        data.append(mat['X2'])
        label = np.squeeze(mat['Y'])
    
    else:
        raise NotImplementedError(f"Dataset {dataset_name} loading logic not implemented.")

    # 统一转换为 float32 和 int64
    view1 = data[0].astype(np.float32)
    view2 = data[1].astype(np.float32)
    label = label.astype(np.int64)

    return view1, view2, label

class CleanDataset(Dataset):
    def __init__(self, view1, view2, labels):
        """
        :param view1: shape (N_samples, N_features1)
        :param view2: shape (N_samples, N_features2)
        :param labels: shape (N_samples, )
        
        注意：原始代码的 Dataset 实现中，数据存储是转置的 (Features, Samples)，
        这里我们保持这个约定以兼容您的模型输入层。
        """
        # 转置数据以匹配原始代码习惯: (Features, Samples)
        self.view1 = view1.T 
        self.view2 = view2.T
        self.labels = labels
        
        # 创建全1的 mask，表示所有数据都是完整的（没有 missing view）
        self.mask = np.ones((len(labels), 2), dtype=np.int64)

    def __getitem__(self, index):
        # 获取数据 (Features, )
        fea0 = torch.from_numpy(self.view1[:, index]).type(torch.FloatTensor)
        fea1 = torch.from_numpy(self.view2[:, index]).type(torch.FloatTensor)
        
        # 兼容性处理：模型 forward 可能需要维度 (1, Features) 或 (Features)
        # 原始代码使用了 unsqueeze(0)，这里保持一致
        # fea0 = fea0.unsqueeze(0) 
        # fea1 = fea1.unsqueeze(0)
        # *修正*：如果您的模型是一维输入 (Linear)，通常不需要 unsqueeze(0) 除非代码里特意要了。
        # 原始代码 data_loader.py 第 162 行加了 unsqueeze。保留它。
        fea0 = fea0.unsqueeze(0)
        fea1 = fea1.unsqueeze(0)

        label = torch.tensor(self.labels[index]).long()
        
        # --- 兼容性返回 ---
        # 原始训练循环需要解包：(x0, x1, labels, real_labels_X, real_labels_Y, mask, index)
        # 在干净数据集中：
        # labels (用于聚类) = real_labels_X = real_labels_Y = Ground Truth Label
        # mask = 全1
        
        class_labels0 = label
        class_labels1 = label
        mask_val = torch.tensor(self.mask[index]).long()
        
        return fea0, fea1, label, class_labels0, class_labels1, mask_val, index

    def __len__(self):
        return len(self.labels)

def loader_cl(train_bs, neg_prop, aligned_prop, complete_prop, is_noise, dataset_name, NetSeed):
    """
    重构后的加载器接口。
    虽然参数列表为了兼容保持不变，但内部忽略了 neg_prop, is_noise 等参数。
    只加载干净数据。
    """
    print(f"Loading CLEAN dataset: {dataset_name} ...")
    
    # 1. 加载原始数据
    view1, view2, labels = load_raw_mat(dataset_name)
    
    print(f"Data shape: View1: {view1.shape}, View2: {view2.shape}, Labels: {labels.shape}")

    # 2. 构建 Dataset
    clean_dataset = CleanDataset(view1, view2, labels)

    # 3. 构建 DataLoader
    # 对于无监督聚类，通常训练集就是全集
    # 为了兼容代码中的 train_pair_loader 和 all_loader，我们返回两个相同的 loader
    
    # train_pair_loader: 用于对比学习训练 (打乱)
    train_pair_loader = DataLoader(
        clean_dataset,
        batch_size=train_bs,
        shuffle=True,
        drop_last=False, # 建议 False，以免丢掉最后的数据影响聚类评估
        num_workers=0    # 避免多线程数据复制问题，可根据需要改为 4
    )

    # all_loader: 用于推理和评估 (通常不打乱，但在聚类中打乱也没关系，主要是 validation 用)
    all_loader = DataLoader(
        clean_dataset,
        batch_size=128, # 推理时 batch size 可以大一点
        shuffle=False,  # 评估时通常不需要 shuffle，方便对应索引
        num_workers=0
    )

    # 返回 dummy seed 保持接口一致
    return train_pair_loader, all_loader, NetSeed

def get_train_loader(x0, x1, c0, c1, train_bs):
    """
    保留此函数以防 Clustering_generate 等模块调用
    用于在微调阶段重新封装数据
    """
    # 这里 x0, x1 已经是转置过的 (Features, Samples) 或者需要根据上下文判断
    # 假设传入的是 numpy array
    
    # 由于 CleanDataset 内部会执行 .T，如果传入的已经是 (Features, Samples)，
    # 我们需要转回 (Samples, Features) 传入，或者修改 Dataset 逻辑。
    # 为了简单，这里假设传入的是标准 (Samples, Features)
    
    dataset = CleanDataset(x0.T, x1.T, c0) # 注意这里的转置处理，视传入数据形状而定
    
    loader = DataLoader(
        dataset,
        batch_size=train_bs,
        shuffle=True,
        drop_last=False
    )
    return loader