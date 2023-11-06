# chao
# 时间：2023/11/2 10:45

#通过广播机制（broadcasting mechanism）来复制元素，使不同形状的tensor可以进行运算
import torch

a = torch.arange(3).reshape(3,1)
b = torch.arange(2).reshape(-1,2)
print(a, b) #此时的a和b形状不同，无法进行运算符计算
# 但是pytorch的广播机制，可以自动的将两个tensor拓展成3*2的tensor去进行计算
c = a + b
print(c)

x = torch.tensor([[1,2,3]]) #1*3
y = torch.arange(3).reshape(3,-1)   #3*1
print(x)
print(y)

z = x + y
print(z)