# chao
# 时间：2023/11/16 11:16

import random
import torch
from d2l import torch as d2l

# 生成一个数据集
def synthetic_data(w, b, num_examples): #@save
    '''
    定义一个人工合成数据的方法，y=Xw+b+噪声
    :param w: 权重
    :param b: 偏置
    :param num_examples: 样本数量
    :return:
    '''
    X = torch.normal(0, 1, (num_examples, len(w)))
    # 生成均值为0，标准差为1的随机数张量，其中(num_examples, len(w))是作为输出X的shape
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000) #调用生成随机数矩阵方法，生成一个features（1000x2）和labels
print(f'第一个样本数据为：{features[0]}, \n标签为：{labels[0]}')

d2l.set_figsize()
a = features[:, 1].detach().numpy() #按照索引查询features的第二列元素，进行分离，然后转换为numpy数组
b = labels.detach().numpy()
print(a.size, '\n', b.size)
d2l.plt.scatter(a, b, 1);
# d2l.plt.show()

# 读取数据集
def data_iter(batch_size, features, labels): #@save
    '''
    读取数据集，返回小批量的样本数据
    :param batch_size: 批量大小
    :param features: 特征矩阵
    :param lables: 标签矩阵
    :return:
    '''
    num_examples = len(features) # 计算样本数量
    indices = list(range(num_examples)) # 定义所有样本的索引，方便通过索引去循环取样本
    # 通过range方法，返回一个int型的有序对象，然后转换为list形式，方便索引查询
    # print(indices)
    random.shuffle(indices) # 使用shuffle函数随机打乱indices索引列表，该方法直接在原来的输入中改变，不返回新的数据，返回None
    # print('打乱顺序后的样本索引列表为：\n',len(indices))
    # 通过for循环，对全部样本循环取出小批量的样本
    for i in range(0, num_examples, batch_size):
    # 使用range类，获取i的循环范围，：从0开始，到样本数量结束，步长是batch的size
       # 定义需要每次取的批量数量的索引，每次循环更新i的值，都会取batch_size长度的不同的索引数量
         batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        # batch_indices = indices[i: min(i + batch_size, num_examples)] # 这里尝试了把batch_indices存储为列表
        # 对全部样本的索引列表indices进行索引查询，查询位置是从i开始，到i+batch_size，这样正好是batch_size的大小
        # 在取索引的时候，
        # 并且i是循环取值，这样在查询批量索引的时候，不会重复取到之前的索引
        # 这里对batch的索引是存储为张量，但是尝试了把batch_indices存储为list格式，也是可以正常取值
        # print('根据batch_size的数量，取出的索引位置：\n',batch_indices, type(batch_indices))
         yield features[batch_indices], labels[batch_indices] #返回通过batch_indices索引查询到的特征矩阵和标签


# 整体思路：
# 1. 把输入样本矩阵的长度处理为乱序列表，作为查询样本的索引
# 2. 使用for循环不断的按照索引查询一定数量的样本，每次查询不重复，直到所有样本遍历一遍
# 3. 为了不重复查询，是按照顺序查询一定数量的索引，初始化了i，i的取值范围使用range()方法去灵活的按照batch_size改变
# 4. 把选择的索引号存储到一个batch_indices中，为了防止在循环中取索引号超出样本数量，使用min()来控制
# 5. 最后返回features[batch_indices]和labels[indices]，通过取出的一定数量的索引号去特征矩阵和标签矩阵中查询，返回出来
# 6. 通过上边的步骤，就可以按照batch的数量循环的把整个样本集合遍历输出一遍

# batch_size = 10
# for X, y in data_iter(batch_size, features, labels):
#     print(X.shape)
#     break

# 初始化模型参数
w = torch.normal(0, 0.01, (2, 1), requires_grad=True) # 初始化一个均值为0，标准差为0.01的2x1的权重
# w = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True) # 初始化偏置为0
# 因为之后要求梯度，所以在初始化时，都加了requires_grad=True

# 定义模型
def linreg(X, w, b): #@save
    '''
    线性回归模型
    :param X: 输入，样本矩阵
    :param w: 权重
    :param b: 偏置
    :return:
    '''
    return torch.matmul(X, w) + b # Xw+b

# 定义损失函数
def square_loss(y_hat, y): #@save
    '''
    损失函数，均方
    :param y_hat: 预测值
    :param y: 真实值
    :return:
    '''
    return (y_hat - y.reshape(y_hat.shape)) ** 2 /2 # 这里reshape预测值，是为了防止和真实y的形状不同，无法进行计算
    # 书里写的是在实现中，需要把真实标签y给reshape成和预测值y_hat一样的形状，但是预测值和真实值不应该都是一个向量吗，比如nx1的张量

# 定义优化算法
def sgd(params, lr, batch_size): #@save
    '''
    mini_batch gradient descent 小批量梯度下降算法
    :param params: 需要学习的参数，可传列表，多个需要优化求的参数
    :param lr: 学习率
    :param batch_size: 批量数量
    :return:
    '''
    with torch.no_grad(): # 禁止使用梯度跟踪和计算微分，节省计算花费，提高代码运行效率
        for param in params: # 遍历params中的每个参数，执行以下操作
            param -= lr * param.grad / batch_size # 参数更新，param.grad：获取关于参数的梯度值
            param.grad.zero_() # 因为自动求梯度会累计上一次的梯度，所以要添加一个归零函数，这样每一次更新梯度都是对的

# 训练
lr = 0.03
# 增加学习率，三个周期的loss相差很小，说明在第一个周期就已经基本完成了优化；
# 减小学习率，三个周期的loss都很大，且差值也大，说明三个周期根本不足以优化到最佳；
num_epochs = 3
net = linreg
loss = square_loss
batch_size = 10

for epoch in range(num_epochs): # 遍历num_epochs的range列表，执行以下操作，这个循环目的是设置迭代周期的次数
    # 在每个周期中都使用小批量读取数据集的方法，循环遍历样本
    for X, y in data_iter(batch_size, features, labels): # X, y 分别遍历使用data_iter所取的小批量样本
        l = loss(net(X, w, b), y)
        l.sum().backward() # 使用反向传播，计算l的梯度
        sgd([w, b], lr, batch_size) # 使用梯度下降方法，更新参数w和b
    # 使用训练好的模型，评估模型在所有样本上的预测效果
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'w:{w}, b:'
              f'{b}')
        print(f'epoch{epoch + 1}, loss {float(train_l.mean()):f}')
print(f'w的估计误差{true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差值{true_b - b}')