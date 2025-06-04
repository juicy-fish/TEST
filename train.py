import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataload import *
from Network import *

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class WatermarkDataset(Dataset):
    def __init__(self, size=10000, dim_r=64, dim_f=32):
        self.data = [generate_sample(dim_r, dim_f) for _ in range(size)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# 训练配置
config = {
    'batch_size': 64,  # 批量大小
    'num_epochs': 10,  # 训练轮数
    'learning_rate': 1e-3,  # 学习率
    'dataset_size': 10000,  # 数据集大小
    'dim_r': 64,  # 原始水印维度
    'dim_f': 32  # 特征水印维度
}

# 创建数据集和数据加载器
dataset = WatermarkDataset(size=config['dataset_size'],
                           dim_r=config['dim_r'],
                           dim_f=config['dim_f'])

loader = DataLoader(dataset,
                    batch_size=config['batch_size'],
                    shuffle=True,
                    pin_memory=True)  # 锁页内存，加速GPU数据传输

# 初始化模型和优化器
model = CrossMapperPair(dim_r=config['dim_r'], dim_f=config['dim_f']).to(device)

# 如果有多个GPU，使用数据并行
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

# 学习率调度器
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2)

# 训练循环
for epoch in range(config['num_epochs']):
    model.train()  # 设置为训练模式
    total_loss = 0
    batch_count = 0

    for W_r, W_f in loader:
        # 将数据移至GPU
        W_r = W_r.to(device, non_blocking=True)
        W_f = W_f.to(device, non_blocking=True)

        # 前向传播
        pred_f, pred_r = model(W_r, W_f)

        # 计算损失
        loss_r2f = F.mse_loss(pred_f, W_f)
        loss_f2r = F.mse_loss(pred_r, W_r)
        loss = loss_r2f + loss_f2r

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1

    # 计算平均损失
    avg_loss = total_loss / batch_count

    # 更新学习率
    scheduler.step(avg_loss)

    # 打印训练进度
    print(f"[Epoch {epoch + 1}/{config['num_epochs']}] Loss: {avg_loss:.4f}")

