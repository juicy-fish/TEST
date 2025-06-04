import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossMapperNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_factor=4, dropout=0.2):
        super(CrossMapperNet, self).__init__()
        h1 = input_dim * 2    # 128
        h2 = input_dim * hidden_factor # 256

        self.fc1 = nn.Linear(input_dim, h1)  # 64 128
        self.fc2 = nn.Linear(h1, h2)  # 128 256
        self.norm = nn.LayerNorm(h2)  # 256
        # 临时线性映射（不带参数注册）
        self.shortcut = nn.Linear(h1, h2)  # h1=128, h2=256
        self.bottleneck = nn.Linear(h2, h1) # 256 128
        self.output = nn.Linear(h1, output_dim)  # 128 32

        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        x1 = self.act(self.fc1(x))
        x1 = self.dropout(x1)

        x2 = self.fc2(x1)
        x2 = self.norm(x2)# 添加维度检查和调整
        if x1.shape[1] != x2.shape[1]:
            # 如果维度不匹配，使用1x1卷积调整x1的维度
            x1 = self.shortcut(x1)
        x2 = F.relu(x2 + x1)  # residual

        x3 = self.bottleneck(x2)
        x3 = self.dropout(F.relu(x3))

        out = self.output(x3)
        return torch.sigmoid(out)  # or softmax, if needed


class CrossMapperPair(nn.Module):
    def __init__(self, dim_r, dim_f):
        super(CrossMapperPair, self).__init__()
        self.r_to_f = CrossMapperNet(input_dim=dim_r, output_dim=dim_f)
        self.f_to_r = CrossMapperNet(input_dim=dim_f, output_dim=dim_r)

    def forward(self, w_r, w_f):
        pred_f = self.r_to_f(w_r)  # F_{r→f}(W_r)
        pred_r = self.f_to_r(w_f)  # F_{f→r}(W_f)
        print("second save")
        return pred_f, pred_r


