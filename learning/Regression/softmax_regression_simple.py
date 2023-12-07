# chao
# 时间：2023/11/28 15:16
# softmax回归的简单实现

import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 定义网络模型，在线性层之前添加另一个展平层：Flatten，用于调整网络输入的形状
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 10)
)

# 定义初始化权重方法
def init_weights(m):
    '''

    :param m: m代表一个模块
    :return:
    '''
    if type(m) == nn.Linear: # 如果模块m属于Linear类型，则初始化m的weight
        nn.init.normal_(m.weight, mean=0, std=0.01) # nn中初始化权重的方法，传入需要初始化的张量、均值（默认为0）和标准差（默认1.0）

net.apply(init_weights)

# 定义损失交叉熵函数
loss = nn.CrossEntropyLoss(reduction='none') # reduction是指定损失的计算方式，none返回的是每个样本的交叉熵损失值

# 定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

# 训练
num_epochs = 3
# 保存了定义的方法，但是调用时找不到
# d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
