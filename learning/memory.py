# chao
# 时间：2023/11/2 11:45

#使用特定的赋值方法来节省内存

import torch

x = torch.ones(2,3)
y = torch.zeros(2,3)
print(x)
print(y)
before = id(y) #记录y的id
before1 = id(x)
y = y + x
print(id(y) == before) #判断执行y = y + x后，y的id和原始y的id是否相同，结果是不同的
#因为执行运算后，为运算结果分配了新的内存，将y指向了这个新的内存地址

# 为了节省内存，使用一下方式去分配

z = torch.zeros_like(y)  #zeros_like(tensor)创建一个形状和已有张量相同，但是元素全是0的张量
print('the id of z is:', id(z))
z[:] = z + y #这中也是一样，不能有重复使用z的地方
print('the new id of z is:', id(z))
#前后z的id是相同的，使用z[:] = <expression> 切片表示法，可以将新的结果分配到原来张量的内存中
print(z)

z = x #如这里，我先将x赋值给z，输出的z是111，111
print(z)
x += y #但是经过+=的计算后，输出的z却成为了222，222；这是因为计算后的结果重新放到了原x的内存中，所以原x变成222，222，z也就被赋值成了222，222
#或者使用+=，使运算的值分配给原先的内存, 注意：这种情况一定是后边或者前边没有用重复使用x，因为+=是改变原对象的值，会影响到其他地方的使用

print(id(x) == before1)
print('x is:', x)
print('z is:', z)


#节省内存空间的赋值方法