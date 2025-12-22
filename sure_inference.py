import torch
import numpy as np

def both_infer(model, device, all_loader, setting=2, cri=None, return_data=False):
    """
    Clean version of both_infer for aligned and complete data.
    Ignores 'setting', 'cri' (for alignment logic), and complex recovery logic.
    Just extracts features.
    """
    model.eval()
    
    # 存储结果列表
    view0_features = []
    view1_features = []
    all_labels = []

    # 如果需要返回原始数据 (return_data=True)
    raw_x0 = []
    raw_x1 = []
    
    with torch.no_grad():
        # 适配 refactored data_loader 的返回值: 
        # (x0, x1, labels, class_labels0, class_labels1, mask, index)
        for batch_idx, (x0, x1, labels, _, _, _, _) in enumerate(all_loader):
            
            # 数据迁移到 GPU
            x0 = x0.to(device)
            x1 = x1.to(device)
            
            # Flatten (针对 Scene15 的 Linear 模型输入需求)
            # 如果您的模型输入不需要 flatten，请根据 models.py 调整，
            # 但通常 SUREfcScene 需要 (Batch, FeatureDim)
            if len(x0.shape) > 2:
                x0 = x0.view(x0.size(0), -1)
                x1 = x1.view(x1.size(0), -1)
            
            # 前向传播 (获取隐层特征 z0, z1)
            # model 返回: z0, z1, xr0, xr1
            z0, z1, _, _ = model(x0, x1)
            
            # 收集特征 (转回 CPU numpy)
            view0_features.append(z0.cpu().numpy())
            view1_features.append(z1.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            if return_data:
                raw_x0.append(x0.cpu().numpy())
                raw_x1.append(x1.cpu().numpy())

    # 拼接所有 Batch 的结果
    view0_features = np.concatenate(view0_features, axis=0)
    view1_features = np.concatenate(view1_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # 打印提示，保持和原代码行为一致的日志风格
    if cri is None:
        print('Inference on Clean Data (No Re-alignment needed)')
    else:
        print('Inference on Clean Data (Alignment skipped)')

    # 根据 return_data 返回对应数量的变量
    if return_data:
        raw_x0 = np.concatenate(raw_x0, axis=0)
        raw_x1 = np.concatenate(raw_x1, axis=0)
        # 为了兼容原接口返回 7 个值: 
        # out0, out1, labels, x0, x1, labels0, labels1
        return view0_features, view1_features, all_labels, raw_x0, raw_x1, all_labels, all_labels
    else:
        # 为了兼容 test_scene_clean.py 调用: v0, v1, gt_label = both_infer(...)
        return view0_features, view1_features, all_labels