# chao
# 时间：2023/11/16 15:28


# 矢量化加速
import math
import time

import torch
import numpy as np
from d2l import torch as d2l

n = 10000
a = torch.ones(n)
b = torch.ones(n)

# 定义一个计时器，为了方便在之后的代码中重复调用
class Timer: #@save
    '''
    记录多次运行的事件
    '''
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        '''
        启动计时器
        :return:
        '''
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()

# 使用for循环的方式，将张量a和b的每一个元素进行加法运算
c = torch.zeros(n)
timer = Timer()
for i in range(n):
    c[i] = a[i] + b[i]
print(f'{timer.stop():.5f} sec')

# 使用重载的+运算法，计算每个元素的和

timer.start()
d = a + b
print(f'{timer.stop():.5f} sec') # 时间大大减少

# 这一节的意思应该是执行矩阵和矩阵之间的运算时，用已有的运算符，而不是自己去写按元素计算的循环