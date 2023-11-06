# chao
# 时间：2023/11/3 11:18


#通过广播机制（broadcasting mechanism）来复制元素，使不同形状的tensor可以进行运算
import torch

a = torch.tensor([[1,3], [1,4], [3,2]])
b = torch.ones(1, 2)
print('a = ', a)
print('b = ', b)

print('a+b=', a+b)