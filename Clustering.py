import numpy as np
import sklearn.metrics as metrics
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import logging

def Clustering(x_list, y, random_state=None):
    """
    执行 KMeans 聚类并计算 ACC, NMI, ARI
    Args:
        x_list: list of numpy arrays (不同视图的特征)
        y: 真实标签 (Ground Truth)
        random_state: 随机种子 (int)
    """
    # 1. 确定聚类数
    n_clusters = np.size(np.unique(y))
    
    # 2. 拼接多视图特征 (Concatenate views)
    # 假设 x_list 是 [view1_features, view2_features]
    # 维度拼接: (N, D1) + (N, D2) -> (N, D1+D2)
    x_final_concat = np.concatenate(x_list, axis=1)

    # 3. 执行 KMeans (显式传入 random_state 以保证复现性)
    # n_init='auto' 或 固定值 (如 20) 可提高稳定性
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
    kmeans.fit(x_final_concat)
    y_pred_kmeans = kmeans.labels_

    # 4. 计算指标
    # 注意: 有些数据集标签是从 1 开始的，如果需要可以调整，但 metrics通常不敏感
    scores, _ = clustering_metric(y, y_pred_kmeans, n_clusters)

    ret = {}
    ret['kmeans'] = scores
    return ret

def best_map(y_true, y_pred):
    """
    使用匈牙利算法 (Hungarian Algorithm) 解决聚类标签与真实标签不匹配的问题，
    从而计算 Accuracy (ACC)。
    替代了原来的 Munkres 库。
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    
    # 构建混淆矩阵 (Confusion Matrix)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    
    # 使用 scipy 的 linear_sum_assignment (最大权匹配)
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    
    # 计算准确率
    return w[row_ind, col_ind].sum() / y_pred.size

def clustering_metric(y_true, y_pred, n_clusters, verbose=False, decimals=4):
    """
    计算聚类指标: ACC, NMI, ARI
    """
    # 1. ACC (Accuracy)
    # 直接使用 best_map 计算最佳匹配下的准确率
    acc = best_map(y_true, y_pred)
    acc = np.round(acc, decimals)

    # 2. NMI (Normalized Mutual Information)
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
    nmi = np.round(nmi, decimals)

    # 3. ARI (Adjusted Rand Index)
    ari = metrics.adjusted_rand_score(y_true, y_pred)
    ari = np.round(ari, decimals)

    if verbose:
        logging.info('ACC: {}, NMI: {}, ARI: {}'.format(acc, nmi, ari))
        
    return {'accuracy': acc, 'NMI': nmi, 'ARI': ari}, None