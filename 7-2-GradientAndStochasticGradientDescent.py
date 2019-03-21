import math
from mxnet import nd
from matplotlib import pyplot as plt
import numpy as np

def gd(eta):
    x = 10
    result = [x]
    for i in range(10):
        x -= eta * 2 * x
        result.append(x)
    print(x)
    return result
# gd(0.2)

def show_trace(res):
    n = max(abs(min(res)),abs(max(res)),10)
    f_line = np.arange(-n, n, 0.1)
    # plt.se
    plt.plot(f_line, [ x*x for x in f_line])
    plt.plot(res, [x*x for x in res], '-o')
    plt.xlabel('x')
    plt.ylabel('f(x)=x^2')
    plt.show()

# show_trace(gd(1.1))


def train_2d(trainer):
    x1, x2, s1, s2 = -5, -2, 0, 0

    # 列表中的元素可以不断地进行添加！
    results = [(x1,x2)]
    for i in range(20):
        x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
        print('epoch %d x1 %f x2 %f' % (i+1, x1, x2))
    return results

def show_trace_2d(f, results):
    plt.plot(*zip(*results), '-o',color = 'orange')

    # meshgrid函数是指定xy轴的取值范围（以生成网格点坐标矩阵！）
    x1, x2 = np.meshgrid(np.arange(-5.5,1.0,0.1), np.arange(-3.0,1.0,0.1))
    # contour函数是指画出二元函数，并以等高线图多彩色画出来！
    plt.contour(x1,x2,f(x1,x2))
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

lr = 0.1
def f_2d(x1,x2):
    return x1 **2+ 2*x2**2

def gd_2d(x1,x2,s1,s2):
    return (x1 - lr*2*x1,x2 -lr*4*x2,0,0)

show_trace_2d(f_2d,train_2d(gd_2d))


def sgd_trace_2d(x1,x2,s1,s2):
    # 记住ndarray和nparray都有同样的方法生成符合正态分布的随机数
    # 差别在于：
    # 1.np生成的是一个scalar标量，而nd生成的是一个NDArray
    return (x1 - lr*(2*x1 + np.random.normal(scale=0.1)),
            x2 - lr*(4*x2 + np.random.normal(scale=0.1)),0,0)

# show_trace_2d(f_2d,train_2d(sgd_trace_2d))