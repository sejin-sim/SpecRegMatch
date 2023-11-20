import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone.ResNet import ResNet, BasicBlock

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.args = args
        self.backbone = ResNet(args, BasicBlock, [2, 2, 2, 2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.regressor = nn.Sequential(nn.Linear(512, 512),nn.BatchNorm1d(512),nn.Linear(512, 11))
        self.projector = nn.Sequential(nn.Linear(512, 512),nn.BatchNorm1d(512),nn.Linear(512, 11))
        self.bn = nn.BatchNorm1d(512, affine=False)
        
    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m, f"Not a square tensor, dimensions found: {n} and {m}"
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
        
    # Barlow twins
    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1) 
        
        pred = self.regressor(x)
        
        bt_loss = 0.0
        if x.size(0) > self.args.batch_size : # unlabeled 있는 경우 = Training
            
            _, f_w, f_s, f_m = x.split(x.size(0) // 4) # (batch * 4, 512)
            
            # barlo twins
            c = self.bn(f_w).T @ self.bn(f_s) / (x.size(0) // 4)
            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum() # Diagonal COmponenet 동일 정보 임베딩
            off_diag = self.off_diagonal(c).pow_(2).sum() # 임베딩 벡터의 다른 요소는 독립적 정보 인코딩
            bt_loss = on_diag + self.args.lambd_bt * off_diag         
        
        return pred, bt_loss
