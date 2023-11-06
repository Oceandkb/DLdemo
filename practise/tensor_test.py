# chao
# 时间：2023/11/2 09:52

#创建tensor
#计算tensor中元素的数量
#计算tensor的大小
#改变tensor的形状
#初始化tensor


import torch

x = torch.arange(12)
print(x)
print(x.numel())
print(x.size())

X = x.reshape(3, -1)
print(X)
print(X.size())

x0 = torch.zeros(3,4)
x1 = torch.ones(3,4)
print(x0)
print(x1)