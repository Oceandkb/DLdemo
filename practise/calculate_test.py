# chao
# 时间：2023/11/3 11:02

#执行tensor之间的运算符运算和逻辑运算符运算，分别按照行和列的方式连结两个相同形状的tensor

import torch

x = torch.tensor([[1,2,3],[4,5,6]])
y = torch.arange(6).reshape(2,-1)
print('x is:', x)
print('y is:', y)

print('x+y=', x + y)
print('x-y=',x - y)
print('x*y=', x * y)
print('x**y=', x ** y)
print('exp(x)=', torch.exp(x))
print('x > y is:', x > y)

#连结xy

xy = torch.cat([x, y], dim=0)
yx = torch.cat([x, y], dim=1)
print('xy is:', xy)
print('yx is:', yx)

#对tensor的元素求和

print('the sum of x is:', torch.sum(x), 'the size of sum x is:', torch.sum(x).size())