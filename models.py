import torch.nn as nn
import torch
import torch.nn.functional as F
import copy

# 基础全连接层块
class FCN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[512, 512]):
        super(FCN, self).__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[i+1]))
                layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)

# 动量双塔模型
class MomentumSURE(nn.Module):
    def __init__(self, feature_dim=512, momentum=0.996):
        super(MomentumSURE, self).__init__()
        self.m = momentum
        
        # --- Online Network ---
        # 针对 Scene15 数据集: View1=20, View2=59 (基于之前的日志推断，或根据实际数据维度修改)
        # 这里假设 View 1 dim=512 (GIST?), View 2 dim=... 需确认 loader 输出
        # 假设 loader_cl_noise 输出已经是提取好的特征，这里用 MLP 映射
        # 为了通用性，假设输入已经展平，具体维度在运行时自动适配或硬编码
        # 假设 View 1 (Gist) = 512, View 2 (HOG/LBP等) 
        # *注意*: 请确保这里的输入维度与你的 Scene15 loader 一致
        self.dim_v1 = 20   # 示例维度，请根据 data_loader.py 确认
        self.dim_v2 = 59   # 示例维度，请根据 data_loader.py 确认
        
        # Encoder (Online)
        self.encoder_v0 = FCN(self.dim_v1, feature_dim)
        self.encoder_v1 = FCN(self.dim_v2, feature_dim)
        
        # Decoder (Online) - 用于重建
        self.decoder_v0 = FCN(feature_dim, self.dim_v1)
        self.decoder_v1 = FCN(feature_dim, self.dim_v2)
        
        # --- Target Network (动量塔) ---
        self.target_encoder_v0 = copy.deepcopy(self.encoder_v0)
        self.target_encoder_v1 = copy.deepcopy(self.encoder_v1)
        
        # 冻结 Target 网络参数
        for p in self.target_encoder_v0.parameters(): p.requires_grad = False
        for p in self.target_encoder_v1.parameters(): p.requires_grad = False

    @torch.no_grad()
    def _momentum_update_target_encoder(self):
        """
        动量更新： theta_target = m * theta_target + (1 - m) * theta_online
        """
        for param_o, param_t in zip(self.encoder_v0.parameters(), self.target_encoder_v0.parameters()):
            param_t.data = param_t.data * self.m + param_o.data * (1. - self.m)
        
        for param_o, param_t in zip(self.encoder_v1.parameters(), self.target_encoder_v1.parameters()):
            param_t.data = param_t.data * self.m + param_o.data * (1. - self.m)

    def forward_online(self, x0, x1):
        """在线网络前向传播 (带梯度)"""
        h0 = self.encoder_v0(x0)
        h1 = self.encoder_v1(x1)
        
        # 重建
        z0 = self.decoder_v0(h0)
        z1 = self.decoder_v1(h1)
        
        return h0, h1, z0, z1

    @torch.no_grad()
    def forward_target(self, x0, x1):
        """目标网络前向传播 (无梯度)"""
        h0_t = self.target_encoder_v0(x0)
        h1_t = self.target_encoder_v1(x1)
        return h0_t, h1_t
    
class CrossViewSemanticEnhancement(nn.Module):
    def __init__(self, feature_dim=512):
        super(CrossViewSemanticEnhancement, self).__init__()
        self.scale = feature_dim ** -0.5
        
        self.w_q1 = nn.Linear(feature_dim, feature_dim)
        self.w_k1 = nn.Linear(feature_dim, feature_dim)
        self.w_v1 = nn.Linear(feature_dim, feature_dim)

        self.w_q2 = nn.Linear(feature_dim, feature_dim)
        self.w_k2 = nn.Linear(feature_dim, feature_dim)
        self.w_v2 = nn.Linear(feature_dim, feature_dim)
        
        self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(self, h0, h1):
        # 保持之前的逻辑不变
        q1 = self.w_q1(h0); k1 = self.w_k1(h1); v1 = self.w_v1(h1)
        attn_score1 = torch.matmul(q1, k1.transpose(-2, -1)) * self.scale
        h0_enhanced = torch.matmul(F.softmax(attn_score1, dim=-1), v1)
        
        q2 = self.w_q2(h1); k2 = self.w_k2(h0); v2 = self.w_v2(h0)
        attn_score2 = torch.matmul(q2, k2.transpose(-2, -1)) * self.scale
        h1_enhanced = torch.matmul(F.softmax(attn_score2, dim=-1), v2)
        
        return self.layer_norm(h0 + h0_enhanced), self.layer_norm(h1 + h1_enhanced)
    
class SUREfcScene(nn.Module):  # 20, 59
    def __init__(self):
        super(SUREfcScene, self).__init__()
        num_fea = 512

        self.encoder0 = nn.Sequential(
            nn.Linear(20, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.1),
            
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.1),

            nn.Linear(1024, num_fea),
            nn.BatchNorm1d(num_fea),
            nn.ReLU(True)
        )

        self.encoder1 = nn.Sequential(
            nn.Linear(59, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.1),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.1),

            nn.Linear(1024, num_fea),
            nn.BatchNorm1d(num_fea),
            nn.ReLU(True)
        )

        self.decoder0 = nn.Sequential(nn.Linear(num_fea, 1024), nn.ReLU(), nn.Dropout(0.1), 
                                      nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.1),
                                      nn.Linear(1024, 20))
        self.decoder1 = nn.Sequential(nn.Linear(num_fea, 1024), nn.ReLU(), nn.Dropout(0.1), 
                                      nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.1),
                                      nn.Linear(1024, 59))

    def forward(self, x0, x1):
        z0 = self.encoder0(x0)
        z1 = self.encoder1(x1)
        z0, z1 = F.normalize(z0, dim=1), F.normalize(z1, dim=1)
        xr0 = self.decoder0(z0)
        xr1 = self.decoder1(z1)
        return z0, z1, xr0, xr1