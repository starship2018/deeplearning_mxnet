import math
import numpy as np
from matplotlib import pyplot as plt
from mxnet import nd, autograd
from mxnet.gluon import data as gdata
import time


def f_2d(x1,x2):
    return 0.1 * x1**2 + 2 * x2**2



def adagrad_2d(x1,x2,s1,s2,eta):
    #自变量梯度
    g1, g2, eps= 0.2*x1, 4*x2, 1e-6
    s1 += g1**2
    s2 += g2**2
    x1 -= eta /math.sqrt(s1 + eps) * g1
    x2 -= eta /math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def train():
    x1, x2, s1, s2 = -5, -3, 0, 0
    result = []
    for i in range(20):
        x1, x2, s1, s2 = adagrad_2d(x1, x2, s1, s2, 2)
        result.append((x1, x2))
    return result

result = train()
def show_grace():
    x1, x2 = np.meshgrid(np.arange(-5.0,1.0,0.1),np.arange(-3.0,1.0,0.1))
    plt.contour(x1,x2,f_2d(x1,x2))

    plt.plot(*zip(*result),'-o',color = 'orange')
    plt.show()

# show_grace()



#【生成数据】
data = np.genfromtxt('./data/airfoil_self_noise.dat',delimiter='\t')
# Z-SCORE标准化处理数据
# 因为特征之间的量级相差较大
data = (data - data.mean(axis=0))/data.std(axis=0)

features, labels = nd.array(data[:1500,:-1]),nd.array(data[:1500,-1])

#【读取数据】
data_iter = gdata.DataLoader(gdata.ArrayDataset(features, labels),batch_size=10,shuffle=True)


#【定义模型】
def net(X,w,b):
    return nd.dot(X,w) + b

#【初始化模型参数】
w = nd.random.normal(scale=0.01, shape=(features.shape[1],1))
b = nd.zeros(shape=(1,))
w.attach_grad()
b.attach_grad()

#【定义损失函数】
def squared_loss(y_hat,y):
    return (y_hat - y.reshape(y_hat.shape))**2/2

#【定义优化算法】
def init_adagrad_states():
    s_w = nd.zeros((features.shape[1],1))
    s_b = nd.zeros(1)
    return (s_w, s_b)

def adagrad(params,states,hyperparams):
    eps = 1e-6
    for p,s in zip(params,states):
        s[:] += p.grad.square()
        p[:] -= hyperparams['lr'] * p.grad / (s + eps).sqrt()


#【训练模型】

num_epochs =2
start = time.time()

batch_size = 10
ls = []
def eval_loss():
    return squared_loss(net(features,w,b), labels).mean().asscalar()


for _ in range(num_epochs):
    for epoch_i , (X, y) in enumerate(data_iter):
        with autograd.record():
            y_hat = net(X,w,b)
            l = squared_loss(y_hat,y)
        l.backward()
        adagrad([w,b], init_adagrad_states(), {'lr':0.01})
        if (epoch_i+1)*batch_size%100 ==0:
            ls.append(eval_loss())
print('loss %f  %f sec per epoch' %(ls[-1], (time.time() - start)/num_epochs))
plt.plot(np.linspace(0,num_epochs,len(ls)),ls)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()