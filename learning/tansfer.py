# chao
# 时间：2023/11/2 14:13

#将tensor转换为numpy中的ndarray
import torch

x = torch.tensor([3,4,5])
print(type(x))

#tensor转换为ndarray
a = x.numpy()
print(type(a))

#ndarray转换为tensor
y = torch.tensor(a)
print(type(y))

#将大小为1的标量转换为python数值类型

b = torch.tensor([True])

by = b.item() #item()函数会根据tensor元素的类型去转换为对应的python标量类型x
byf = float(b)
byi = int(b)
print(by, type(by), byf, type(byf),byi)