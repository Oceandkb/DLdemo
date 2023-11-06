# chao
# 时间：2023/11/1 21:09

import torch

x = torch.tensor([1.0,2,3,4])
y = torch.tensor([2,3,4,5])

#tensor之间也可以运用标准运算符去计算，计算方式是按元素，两个或者多个tensor必须是相同size
print(x+y)
print(x-y)
print(x*y)
print(x/y)
print(x**y) #幂运算
print(torch.exp(x))

#按照线性代数运算
X = torch.arange(12, dtype=torch.float32).reshape(3,-1) #dtype=torch.float32是设置tensor的数值类型
Y = torch.tensor([[2.1,1,4,3],[1,2,3,4],[4,3,2,1]])
#将两个tensor连结
XY = torch.cat((X, Y), dim=0) #按照行进行连结
print(XY)
YX = torch.cat((X, Y), dim=1) #按照列进行连结
print(YX)

#通过逻辑运算，构建tensor
Z = X == Y #直接使用逻辑运算符计算两个tensor，输出即可
print(Z)
W = X != Y
print(W)

#对tensor求和，得到一个单元素的tensor，也就是一个标量
O = X.sum()
print(O, O.numel(), O.size()) #标量的纬度为空


#tensor之间的运算符运算、逻辑运算符运算、分别按照行和列的方式连结两个相同形状的tensor