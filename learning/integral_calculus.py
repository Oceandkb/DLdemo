# chao
# 时间：2023/11/7 17:25

# 微积分


import numpy as np
from matplotlib_inline import backend_inline
from d2l import torch as d2l
from matplotlib import pyplot as plt

def f(x):
    return 3 * x ** 2 - 4 * x

def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h

# 极限
h = 0.1
for i in range(5):
    print(f'h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}')
    # h:.5f h精确到小数点后5位的float类型
    h *= 0.1

#画图
def use_svg_display(): #@save
    '''
    使用svg格式在Jupyter中显示绘图
    :return:
    '''
    backend_inline.set_matplotlib_formats('svg') #设置绘图格式

def set_figsize(figsize=(3.5, 2.5)): #@save
    '''
    设置matplotlib的图表大小
    :param figsize:
    :return:
    '''
    use_svg_display()
    d2l.plt.rcParams['figure.figuresize'] = figsize

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend): #@save
    '''
    设置matplotlib的坐标轴
    :param axes:
    :param xlabel:
    :param ylabel:
    :param xlim:
    :param ylim:
    :param xscale:
    :param yscale:
    :param legend:
    :return:
    '''
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

#@save
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None,
         xscale='linear', yscale='linear', fmts=('-', 'm--', 'g-.', 'r:'),
         figsize=(3.5, 2.5), axes=None):
    '''
    绘制数据点
    :param X:
    :param Y:
    :param xlabel:
    :param ylabel:
    :param legend:
    :param xlim:
    :param ylim:
    :param xscale:
    :param yscale:
    :param fmts:
    :param figsize:
    :param axes:
    :return:
    '''
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca()

    #如果x有一个轴，输出True
    def has_one_axis(x):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)

# 绘制函数f(x)=2x-3的函数曲线，以及在x=1处的切线
    x = np.arange(0, 3, 0.1)
    plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
