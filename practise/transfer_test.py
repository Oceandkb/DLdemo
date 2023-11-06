# chao
# 时间：2023/11/6 09:40

#ndarry和tensor之间的转换， tensor转换为python标量

import torch

x = torch.tensor([1])
print('x is:', x)


x_n = x.numpy()
print(type(x_n))

x_t = torch.tensor(x_n)
print(type(x_t))

x_b = x.item()
print(type(x_b))
x_f = float(x)
print(type(x_f))
x_i = int(x)
print(type(x_i))