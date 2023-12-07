# chao
# 时间：2023/11/20 20:23
# 使用已经封装好的方法，简介的实现线性回归

import torch
import numpy as np
from torch.utils import data
from d2l import torch as d2l
from torch import nn
# from linear_network import *
# 构造数据集
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

# 读取数据集
def load_array(data_arrays, batch_size, is_Train):
    '''
    构造一个pytorch数据迭代器
    :param data_arrays: 数据数组
    :param batch_size: 批量大小
    :param is_Train: 表示是否希望数据迭代器对象在每个迭代周期内打乱数据
    :return:
    '''
    dataset = data.TensorDataset(*data_arrays) # 将多个张量封装为一个张量，配合DataLoader一起使用
    # TensorDataset的传参张量时，前边加*号是为了解包，让参数数量匹配
    return data.DataLoader(dataset, batch_size, shuffle=is_Train) # 使用DataLoader加载数据，按照batch_size返回数据

batch_size = 10
data_iter = load_array((features, labels), batch_size, is_Train=True)

# di = iter(data_iter) # iter()是一个内置的迭代器，传入一个迭代对象，创建一个可以迭代访问集合元素的对象
# print(next(di)) # 使用next方法访问元素
# print(next(di)) # 访问下一个

# 定义模型
net = nn.Sequential(nn.Linear(2, 1)) # Linear模型中，输入样本特征为2个，输出的标签是一个标量，所以是1

# 初始化模型参数
# 通过net[0]访问第一层网络
net[0].weight.data.normal_(0, 0.01) # weight.data 读取权重，使用normal_方法重写参数值，两个传参，第一个是均值，第二个标准差
net[0].bias.data.fill_(0) # bias.data 读取偏置，使用fill_方法重写参数值

# 定义损失函数
loss = nn.MSELoss() # MSELoss是nn库中的一个类，用来计算均方误差

# 定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr = 0.03)
print(next(net.parameters())) # parameters()方法返回的是一个可迭代访问的参数对象
# optim库中给了梯度下降的各种变种，只需要调用SGD类，并且传入需要优化的参数，以及超参数；这里通过net.parameters来访问模型中的参数

# 训练
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        # 下边四行代码是和官方给出的torch.optim.SGD的实例是一样的
        l = loss(net(X), y)
        trainer.zero_grad() # 在每次迭代更新，或者计算梯度之前，把参数的梯度归零，类似于grad.zero_()
        l.backward() # 没有先求和再反向传播，因为已经MSELoss返回的是已经求和之后的结果
        trainer.step() # SGD中的一个方法，进行单独的更新优化，用于更新模型参数值
        # 我理解的是，对loss计算完反向传播求梯度后，调用step方法，执行SGD算法
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

