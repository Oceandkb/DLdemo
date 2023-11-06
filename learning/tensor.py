# chao
# 时间：2023/11/1 18:19

import torch

a = torch.tensor(1) #创建一个标量
b = torch.tensor([1]) #创建一个张量

x = torch.arange(12) #创建一个tensor，包含0开始的前12个整数
X = x.reshape(3,4)   #reshape方法改变tensor的形状，reshape参数为改变的宽和高
X1 = x.reshape(3,-1) #reshape方法给定一个宽或者高，可以使用-1代替另一个，自动计算
print(x)
print(x.numel()) #计算tensor的数量
print(X.shape)   #输出tensor的形状
print(X.size())  #计算tensor的纬度
print(X,X1)

#初始化矩阵，元素为0，1
y0 = torch.zeros((2,3,4)) #创建一个元素都为0的tensor，其中第一个参数2代表数量，第二和第三个参数分别代表宽高（行数列数）
y1 = torch.ones((2,3,4))  #创建一个元素都为1的tensor
print(y0,y1)

#初始化矩阵,从特征的概率分布中随机采样
#从标准高斯分布中随机采样
z0 = torch.randn(3,4)
print(z0)
#从列表中取值
z1 = torch.tensor([[2,1,3,4],[1,2,3,4],[4,3,2,1]])
#当取列表中的值创建tensor时，只能取一个列表中的值，如果要创建大形状的tensor，需要使用嵌套列表的形式，否则会报错
print(z1)
print(z1.shape)
z2 = torch.tensor([2,1,34,4])
print(z2)


#创建tensor:torch.tensor() torch.arange()
#计算tensor中元素的数量:x.numel()
#计算tensor的形状：x.shape
#改变tensor的形状：x.reshape(a,b) == x.reshape(a,-1) == x.reshape(-1,b)
#初始化tensor：torch.zeros() torch.ones() torch.tensor([[a,b,c,d]])