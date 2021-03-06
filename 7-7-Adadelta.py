import math
from mxnet import nd,autograd
from matplotlib import pyplot as plt
import numpy as np
from mxnet.gluon import data as gdata
import time





#【生成数据】
data= np.genfromtxt('./data/airfoil_self_noise.dat',delimiter='\t') #因为数据中每个数据之间使用tab隔开，所以在这里要告诉numpy!
# Z-SCORE标准化处理数据
data = (data - data.mean(axis=0))/ data.std(axis=0)

features, labels = nd.array(data[:1500,:-1]),nd.array(data[:1500,-1])

#【读取数据】
batch_size = 10
data_iter = gdata.DataLoader(gdata.ArrayDataset(features, labels),batch_size,shuffle=True)

#【定义计算模型】
def net(X,w,b):
    return nd.dot(X,w) + b

#【初始化模型函数】
w = nd.random.normal(scale=0.01,shape=(features.shape[1],1))
b = nd.zeros(shape=(1,))
w.attach_grad()
b.attach_grad()

#【定义损失函数 - L2范数损失函数】
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape))**2/2

#【定义优化算法】
def init_adadelta_states():
    # s 是一种状态变量，和真正的权重和偏差参数的形状一样
    # 记住！这里的w和b相当于是两种自变量，而不是计算模型中的参数，可以把它当作x1 ,x2
    s_w = nd.zeros((features.shape[1],1))
    s_b = nd.zeros(1)
    delta_w = nd.zeros((features.shape[1],1))
    delta_b = nd.zeros(1)
    return (s_w, delta_w), (s_b, delta_b)


def rmsprop(params,states,hyperparams):
    # 在这里设置一下超参数！
    rho, eps = hyperparams['rho'], 1e-5
    for p , (s, delta) in zip(params,states):
        s[:] = rho * s + (1 - rho) * p.grad**2
        g = (delta + eps).sqrt() / (s + eps).sqrt()* p.grad
        p[:] -= g
        delta[:] = rho * delta + (1- rho)*g**2
        # 这里存在问题，网上的求平方根的方法都是math.sqrt()，但是在这里却行不通？！

#【训练模型】
num_epochs = 2
ls = []

def train():
    start = time.time()
    for _ in range(num_epochs):
        for epoch_i, (X, y) in enumerate(data_iter):
            with autograd.record():
                y_hat = net(X,w,b)
                l = squared_loss(y_hat,y)
            l.backward()
            rmsprop([w,b], init_adadelta_states(), {'rho':0.9})
            if (epoch_i + 1 )* batch_size %100 ==0:
                ls.append(squared_loss(net(features,w,b),labels).mean().asscalar())
    print('loss %f , %f sec per epoch' % (ls[-1], (time.time() - start)/num_epochs))

train()
plt.plot(np.linspace(0,num_epochs,len(ls)), ls)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()