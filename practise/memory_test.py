# chao
# 时间：2023/11/3 18:04


#节省内存的赋值方法

import torch

x = torch.arange(4).reshape(2,-1)
before = id(x)
print(x)
y = torch.ones(2,2)
print(y)

x[:] = x + y
print(before == id(x))

x += y
