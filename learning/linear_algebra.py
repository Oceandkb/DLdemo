# chao
# 时间：2023/11/6 19:56

#the basic knowledge of linear algebra

import torch

#测试是否可以用gpu跑
torch.__version__
print(f"{torch.backends.mps.is_available(),torch.backends.mps.is_built()}")
aa = torch.rand(5).to("mps")
print(aa)


#实例化两个标量x，y
x = torch.tensor(1.0).to('mps')
y = torch.tensor(2.0)

print('x+y=', x+y, 'x*y=', x*y, 'x-y=', x-y, 'x/y=', x/y, 'x**y=', x**y)

#向量
a = torch.arange(6).reshape(2,3)
print('a is', a)
print('the second element of a is', a[1])
print(a[0:5:2])
print('the length of a is', len(a))
print('the shape of a is', a.shape)

#矩阵
A = torch.arange(20, dtype=torch.float32).reshape(5,4)
print('A is:\n', A)
print('the transpose of A is\n', A.T)

B = torch.tensor([[1,2,3],[2,0,4],[3,4,5]])
print('B is\n', B)
print(B == B.T)  #对称矩阵的转置等于他本身


X = torch.arange(24).reshape(2, 3, 4)
print('X is\n', X)

a = torch.arange(20, dtype=torch.float32).reshape(5, 4)
b = a.clone()
print('a is:\n', a, '\n a+b is:\n', a+b, '\n a*b=\n', a*b)

c = 2
print('X+c=\n', X + c, '\n X*c=\n', X * c) #张量乘或者加上一个标量，不改变形状，是对张量的每个元素进行运算

#通过求和对张量降维
print('A is\n', A)
print('the sum of A is\n', A.sum())

#也可以通过指定张量沿着哪一个轴去降维
print('the sum pf A which axis == 0 is\n', A.sum(axis = 0))
print('the sum pf A which axis == 1 is\n', A.sum(axis = 1))

#求张量的平均值
print('the mean of A is\n', A.mean())

#通过计算平均值也可以降低维度，也可以按照轴去求
print('axis == 0, the mean of A is\n', A.mean(axis = 0))
print('axis == 1, the mean of A is\n', A.mean(axis = 1))

#通过添加参数，来保持求和或者均值时维度不变
print('keep dimension, the sum of A is\n', A.sum(axis = 0, keepdims=True))
print('keep dimension, the sum of A is\n', A.sum(axis = 1, keepdims=True))


# 使用cumsum方法，按照某个轴累加，不改变张量的形状和维度
print('the cumsum of a is \n', a.cumsum(axis = 0))

# 点积 dot product
w = torch.arange(6)
z = torch.tensor([1,4,5,5,4,5])
print('<w, z> is', torch.dot(w, z))
#也可以通过对乘积求和表示点积
print('<w, z> is', torch.sum(w * z))

#矩阵-向量积，矩阵的维度必须要和向量的长度相同
d = torch.arange(4, dtype=torch.float32)
print('A和d的向量积为：', torch.mv(A, d))

#矩阵和矩阵的乘法， 要求第一个矩阵的列数等于第二个矩阵的行数 C=AB
B = torch.ones(4, 3)
print('A和B的积为\n', torch.mm(A, B))

#范数
u = torch.tensor([3.0, -4.0])
print('u的L2范数为：', torch.norm(u))
print('u的L1范数为：', torch.abs(u).sum())
v = torch.ones(3, 4)
print('矩阵v的L2范数为：', torch.norm(v))


# • 标量、向量、矩阵和张量是线性代数中的基本数学对象。
# • 向量泛化自标量，矩阵泛化自向量。
# • 标量、向量、矩阵和张量分别具有零、一、二和任意数量的轴。
# • 一个张量可以通过sum和mean沿指定的轴降低维度。
# • 两个矩阵的按元素乘法被称为他们的Hadamard积。它与矩阵乘法不同。
# • 在深度学习中，我们经常使用范数，如L1范数、L2范数和Frobenius范数。
# • 我们可以对标量、向量、矩阵和张量执行各种操作。
