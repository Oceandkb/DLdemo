# chao
# 时间：2023/11/23 11:23

import torch
from torch import nn
from d2l import torch as d2l
from torch.utils import data

# 创建数据集
true_w = torch.tensor([2, -3.4])
ture_b = 4.2
features, labels = d2l.synthetic_data(true_w, ture_b, 1000)

# 读取数据集：tensordataset和dataloader
def batch_data(datasets, batch_size, is_Train):
    data_array = data.TensorDataset(*datasets) # 忘记加*号，导致一直报错
    dataset = data.DataLoader(data_array, batch_size, shuffle=is_Train)
    return dataset

data_iter = batch_data((features, labels), batch_size=10, is_Train=True)
print(next(iter(data_iter)))

# 定义网络
net = nn.Sequential(nn.Linear(2, 1))

# 参数初始化
net[0].weight.data.normal_(0, 0.01) # 在初始化的代码里，忘记了加data.
net[0].bias.data.fill_(0)

# 定义损失函数
loss = nn.MSELoss()

# 定义优化方法
trainer = torch.optim.SGD(net.parameters(), lr=0.03) # 这个地方没记住，通过看官方文档的实例写的

# 定义超参数
epochs = 3

# 开始训练
for epoch in range(epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    print(loss(net(features), labels))








