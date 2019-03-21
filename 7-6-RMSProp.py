import math
from mxnet import nd,autograd
from matplotlib import pyplot as plt
import numpy as np
from mxnet.gluon import data as gdata
import time



def f_2d(x1,x2):
    return 0.1*x1**2 + 2*x2**2


eta, gamma = 0.4, 0.9
def rmsprop_2d(x1,x2,s1,s2):
    g1,g2,eps = 0.2*x1, 4*x2,1e-6
    s1 = gamma * s1 + (1-gamma)*g1**2
    s2 = gamma * s2 + (1-gamma)*g2**2
    x1 -= eta*g1/math.sqrt(s1 + eps)
    x2 -= eta*g2/math.sqrt(s2 + eps)
    return x1, x2, s1, s2

def get_result(trainer):
    x1,x2,s1,s2 = -5,-3,0,0
    result = []
    for i in range(20):
        x1,x2,s1,s2 = trainer(x1,x2,s1,s2)
        result.append((x1,x2))
    return result

# x1,x2 = np.meshgrid(np.arange(-5.0,1.0,0.1),np.arange(-3.0,1.0,0.1))
# plt.contour(x1,x2,f_2d(x1,x2))
# plt.plot(*zip(*get_result(rmsprop_2d)),'-o',color='orange')
# plt.xlabel('x1')
# plt.ylabel('x2')
# plt.show()

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
def init_rmsprop_states():
    # s 相当与预参数，和真正的权重和偏差参数的形状一样
    s_w = nd.zeros((features.shape[1],1))
    s_b = nd.zeros(1)
    return s_w, s_b


def rmsprop(params,states,hyperparams):
    # 在这里设置一下超参数！
    gamma, eps = hyperparams['gamma'], 1e-6
    for p , s in zip(params,states):
        s[:] = gamma * s + (1 - gamma) * p.grad **2
        p[:] -= hyperparams['lr'] * p.grad / (s + eps).sqrt()
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
            rmsprop([w,b], init_rmsprop_states(), {'lr':0.01,'gamma':0.9})
            if (epoch_i + 1 )* batch_size %100 ==0:
                ls.append(squared_loss(net(features,w,b),labels).mean().asscalar())
    print('loss %f , %f sec per epoch' % (ls[-1], (time.time() - start)/num_epochs))

train()
plt.plot(np.linspace(0,num_epochs,len(ls)), ls)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
