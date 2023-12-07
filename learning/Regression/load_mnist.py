# chao
# 时间：2023/11/23 17:25
# 读取图像分类数据集

import torch
import torchvision # 提供计算机视觉任务的工具和数据集
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()

# 读取数据集
# 使用transforms模块中的ToTensor()方法将图像数据从PIL转换为float32格式
trans = transforms.ToTensor()
# 通过框架中的内置函数将Fashion-MNIST数据集下载并读取到内存中
# 使用torchvision.datasets模块下载数据集，传参包括数据集根目录、是否是训练集、应用的转换操作和是否自动下载数据集
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)

#Fashion-MNIST数据的大小，10类，训练集6000张，测试集1000张
print(len(mnist_train), len(mnist_test))
# 每张图片通道1，高度和宽度为28px
print(mnist_train[0][0].shape)

def get_fashion_mnist_labels(labels): #@save
    '''
    返回Fashion-MNIST数据集的文本标签
    :param labels: 标签矩阵
    :return:
    '''
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal',
        'shirt', 'sneaker', 'bag', 'ankle boot'
    ]
    # 循环遍历labels中的所有元素，然后拿到该元素去查找text_labels中的对应元素，最后组成一个list
    # 这里labels中元素都是数字0-9
    return [text_labels[int(i)] for i in labels]

# 可视化数据集样本
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5): #@save
    '''
    绘制图像了列表
    :param imgs:
    :param num_rows:
    :param num_cols:
    :param titles:
    :param scale:
    :return:
    '''
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

X, y = next(iter(data.DataLoader(mnist_train, batch_size = 18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
# d2l.plt.show()

# 读取小批量
batch_size = 256
def get_dataloader_workers(): #@save
    '''
    使用四个进程来读取数据
    :return:
    '''
    return 2
# 使用DataLoader类来读取小批量数据，这里相比之前的线性回归多了一个传参：num_workers，作为加载数据的子进程，int
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=get_dataloader_workers())
timer = d2l.Timer()
for X, y in train_iter:
    continue
print(f'四个线程加载训练集6000个样本，时间为 {timer.stop():.2f}sec')


