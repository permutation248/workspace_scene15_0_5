import torch.nn as nn
import torch
import torch.nn.functional as F
import copy
    
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