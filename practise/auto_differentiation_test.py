# chao
# 时间：2023/11/14 15:19


# 1. 为什么计算二阶导数比一阶导数的开销要更大?
# 2. 在运行反向传播函数之后，立即再次运行它，看看会发生什么。
import torch
x = torch.tensor([2, 3, 4, 5], dtype=torch.float32, requires_grad=True)
y = x * x
y.sum().backward()
print(x.grad)
# y.sum().backward() # 再次执行反向传播函数会报错
# 3. 在控制流的例子中，我们计算d关于a的导数，如果将变量a更改为随机向量或矩阵，会发生什么?
# 4. 重新设计一个求控制流梯度的例子，运行并分析结果。
# 5. 使f(x) = sin(x)，绘制f(x)和df(x)的图像，其中后者不使用f′(x) = cos(x)。 dx