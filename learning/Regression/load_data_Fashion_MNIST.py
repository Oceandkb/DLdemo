# chao
# 时间：2023/11/24 10:42

import torch
import torchvision
from torchvision import transforms
from torch.utils import data
from d2l import torch as d2l
# from  load_mnist import get_dataloader_workers

# 将下载数据集、批量读取数据集集成在一个方法里
def load_data_fashion_mnist(batch_size, resize=None): #@save
    '''
    下载Fashion-MNIST数据集，并加载到内存中
    :param batch_size: 批量大小
    :param resize: 调整图像大小
    :return:
    '''

    # -----定义数据的转换操作------
    trans = [transforms.ToTensor()] # 创建一个列表trans，包含一个ToTensor的转换操作
    # 如果resize不为空，在列表trans的第一个位置插入一个Resize的操作
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans) # 将列表trans中的两个操作组合为一个整体的转换操作，重新给trans

    # ------下载数据集-----
    mnist_train = torchvision.datasets.FashionMNIST(root='../data', train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root='../data', train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=d2l.get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=True, num_workers=d2l.get_dataloader_workers()))

# 指定resize的传参，使用图像大小调整功能
train_iter, test_iter = load_data_fashion_mnist(32, resize=64) # 指定resize的大小
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break
