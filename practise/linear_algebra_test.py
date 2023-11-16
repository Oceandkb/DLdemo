# chao
# 时间：2023/11/8 09:53

import torch

A = torch.arange(6).reshape(2, 3)
print('the length A is', len(A))
B = torch.tensor([[1, 2, 3], [4, 5, 6]])
print('A is\n', A, '\nB is\n', B)

# 1. 证明一个矩阵A的转置的转置是A，即(A⊤)⊤ = A。
print((A.T).T == A)

# 2. 给出两个矩阵A和B，证明“它们转置的和”等于“它们和的转置”，即A⊤ + B⊤ = (A + B)⊤。
AB = A + B
TAB = A.T + B.T
print(AB.T == TAB)

# 3. 给定任意方阵A，A + A⊤ 总是对称的吗?为什么?
a = torch.ones(3, 3)
print('a is\n', a)
print('the length a is', len(a))
print(a + a.T == (a + a.T).T)

# 4. 本节中定义了形状(2,3,4)的张量X。len(X)的输出结果是什么?
x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
print('the length of x is:', len(x))

# 5. 对于任意形状的张量X,len(X)是否总是对应于X特定轴的长度?这个轴是什么?
print('根据输出来看，如果是一个向量，长度就是向量的长度，如果是一个二维矩阵，那么长度是0轴的长度,'
      '三维矩阵的话，长度计算得失z轴的长度，推算下来， 计算的应该是定义时，第一个轴的长度')

# 6. 运行A/A.sum(axis=1)，看看会发生什么。请分析一下原因?
# print(A/A.sum(axis=1)) # 会报错，提示前后矩阵的维度不一致，因为求和后维度发生了改变
print(A/A.sum(axis=1, keepdims=True)) # 加上keepdims的参数后，求和不降维，可以正常运算

# 7. 考虑一个具有形状(2,3,4)的张量，在轴0、1、2上的求和输出是什么形状?
print('x 在0轴上求和的形状：', x.sum(axis=0).shape, x.sum(axis=0))
print('x 在1轴上求和的形状：', x.sum(axis=1).shape)
print('x 在2轴上求和的形状：', x.sum(axis=2).shape)

# 8. 为linalg.norm函数提供3个或更多轴的张量，并观察其输出。对于任意形状的张量这个函数计算得到 什么?
print('the norm of x is', torch.linalg.norm(x))

# 9. 点积、向量积、矩阵积
q = torch.tensor([1, 3, 4, 5], dtype=torch.float32)
w = torch.ones(4)
e = torch.tensor([[1, 4, 6, 7], [2, 4, 5, 7]], dtype=torch.float32)
r = torch.arange(8, dtype=torch.float32).reshape(4, 2)
print('点积', q, w, torch.dot(q, w))
print('向量积', torch.mv(e, q) )
print('矩阵积', torch.mm(e, r))

# 10. 范数
print('L1', torch.abs(q).sum())
print('L2', torch.norm(e))