# chao
# 时间：2023/11/8 14:29

#自动微分

import torch

x = torch.arange(4.0, requires_grad=True)
print(x.grad) #默认值是none
y = 2 * torch.dot(x, x)
print(y)
y.backward() #通过反向传播算法计算y每个分量的梯度
print(x.grad)
print(x.grad == 4 * x)

x.grad.zero_() #默认pytorch会累计梯度，所以计算新的梯度时，要清除之前的
y = x.sum()
y.backward()
print(x.grad)

#非标量求梯度
#非标量求梯度是，backward（）需要传一个参数gradient，作用是几阶导数
x.grad.zero_()
y = x * x
y.sum().backward()
print(x.grad)

#分离计算
x.grad.zero_()
u = y.detach() #当z是关于x,y的函数，y是关于x的哈函数，可以使用detach（）来分离y，把y当常数看待
z = u * x
z.sum().backward()
print(x.grad == u)

# 执行自动求导的步骤：
# 1. 将梯度附加导想要求偏导的变量上：a = torch.ones((3, 4), requires_grad = True)
# 2. 定义函数：b = a * a
# 3. 对定义的函数执行反向传播算法：b.sum().backward()
# 4. 访问变量的导数：a.grad