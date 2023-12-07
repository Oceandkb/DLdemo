# chao
# 时间：2023/11/23 15:16
# softmax回归的从零开始实现

import torch
from d2l import torch as d2l
from IPython import display  # 用于实现输出

# 小批量加载数据集
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 初始化模型参数
num_inputs = 784
num_outputs = 10
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

# 定义softmax操作
# 回顾张量求和，按照哪一个轴，是否保持维度
# a = torch.tensor([
#     [1.0, 2.0, 3.0],
#     [4.0, 5.0, 6.0]
# ])
# print(a.sum(0, keepdim=True).shape, a.sum(1, keepdim=True).shape)

# 定义softmax函数
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition # 应用了广播机制

# 测试通过softmax函数计算后的张量值都在0-1之间的非负数，并且每一行的和为1
# b = torch.normal(0, 1, (2, 5))
# b_prob = softmax(b)
# print(b_prob, b_prob.sum(1))

# 定义模型
def net(X):
    '''
    softmax回归模型
    :param X: 输入的样本
    :return:
    '''
    return softmax(torch.matmul(X.reshape(-1, W.shape[0]), W) + b) # 对输入图像样本展平，使用reshape方法

# 定义损失函数-交叉熵损失
y = torch.tensor([0, 2])
y_hat = torch.tensor([
    [0.1, 0.3, 0.6],
    [0.3, 0.2, 0.5]
])
print(y_hat[[0, 1], y]) # 使用y的值作为索引，输出的是y_hat[0][0]和y_hat[1][2]

def cross_entropy(y_hat, y): #@save
    '''
    计算交叉熵，采用真实标签预测概率的负对数似然
    :param y_hat:
    :param y:
    :return:
    '''
    return - torch.log(y_hat[range(len(y_hat)), y])
    # log函数里边的是采用高级索引，取出预测概率y_hat每一行对应的真实标签y的预测值
print(cross_entropy(y_hat, y))

# 计算分类精度
def accuracy(y_hat, y):
    '''
    计算预算准确的数量
    :param y_hat:
    :param y:
    :return:
    '''
    # 自己写的计算精确度，太简单了
    # y_hat_indices = torch.argmax(y_hat, dim=1)
    # y_acc = (y_hat_indices == y)
    # print(y_acc)
    # y_result = sum(y_acc.int()) / sum(y.tolist())
    # return y_result
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1) # 使用argmax方法，按照1轴返回每一行最大值的索引，赋值给y_hat
    cmp = y_hat.type(y.dtype) == y # 将y_hat的类型转换为和y相同后，使用==计算两个张量
    return float(cmp.type(y.dtype).sum()) # 返回cmp的求和，得到预测正确的数量
print(f'预测分类精确度为{accuracy(y_hat, y) / len(y)}')

# 计算指定数据集上模型的精度
def evaluate_accuracy(net, data_iter): #@save
    '''
    计算指定数据集上模型的精度
    :param net: 模型网络
    :param data_iter: 加载的数据
    :return:
    '''
    if isinstance(net, torch.nn.Module): # 检查net是否为nn.Moudle的实例，如果是，执行下边代码
        net.eval() # 将net设置为评估模式，通常用于推理或测试阶段
    metric = Accumulator(2) # 创建Accumulator类的实例，传入n=2，代表正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
            # 返回的是一个了列表，每一个批量中预测正确的个数和元素的总个数
            print('metric_data:', metric.data)
            # 每一次循环输出的都是上一次结果的累加，因为每执行一次循环，metric.data都是记录了上一次的值，再去加这一次计算的值
    return metric[0] / metric[1] # 预测正确个数/元素总个数

# 定义一个类，用于多个变量进行累加
class Accumulator: #@save
    def __init__(self, n):
        self.data = [0.0] * n # 为实例创建一个data属性
    # 创建一个初始化方法（在实例化该类的时候立即调用，目的是为实例的属性赋予初始值）
    # 接收self和n两个参数，self是对类实例的引用，这个参数通常是自动传递，无需手动传参，作用是帮助方法访问类的属性和方法

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)] # 当这个函数被调用时，是对data属性进行重新赋值
        # zip函数将self.data和args中对应位置的元素打成一个元组，返回一个可迭代的对象
        # 遍历zip函数返回的可迭代的对象，每个元组中的a和b，都执行a+float(b)，并存储到一个新的列表中

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

print(evaluate_accuracy(net, test_iter))

# 训练

# 定义一个函数来训练一个迭代周期
def train_epoch_ch3(net, train_iter, loss, updater): #@save
    '''
    训练模型一个迭代周期
    :param net: 神经网络模型
    :param train_iter: 训练集
    :param loss: 损失函数
    :param updater: 一个更新函数，接收batch_size作为参数，可以是内置的更新函数，也可以是手动定义的
    :return:
    '''
    if isinstance(net, torch.nn.Module):
        net.train() # 将模型设置为训练模式
    metric = Accumulator(3) # 训练损失总和、训练准确度总和、样本数
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        # 判断更新函数updater的类型：内置 or 定制
        if isinstance(updater, torch.optim.Optimizer):
            # 使用pytorch内置优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]

# 定义一个绘图类Animator
class Animator:  #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

# 实现一个训练函数
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater): #@save
    '''
    训练模型
    :param net: 网络模型
    :param train_iter:
    :param test_iter:
    :param loss:
    :param num_epochs:
    :param updater:
    :return:
    '''
    # 使用Animator类画图，这个没画出来动态图，不知道为什么，在jupyter notebook可以
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    # 在每个周期中，使用之前定义的方法，计算训练损失、训练集预测精确率和测试集精确率
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    # print(train_loss, train_acc)
    # 这里为什么要验证这几个值的大小
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc >0.7, test_acc

lr = 0.1
# 定义优化函数
def updater(batch_size): #@save
    return d2l.sgd([W, b], lr, batch_size)

num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

# 预测
def predict_ch3(net, test_iter, n = 6): #@save
    '''
    预测标签
    :param net: 模型
    :param test_iter: 测试集
    :param n:
    :return:
    '''
    for X, y in test_iter:
        break
    # 展示真实标签名称和预测标签名称
    trues = d2l.get_fashion_mnist_labels(y) # 使用已有的方法：get_fashion_mnist_labels(labels)获得标签名称
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    # 这里传参net(X).argmax(axis=1)是计算出来的概率，然后给出最大值的索引位置
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n]
    )
predict_ch3(net, test_iter)
d2l.plt.show()
