from mxnet import autograd, gluon, init, nd
from mxnet.gluon import nn, data as gdata ,loss as gloss
import time
import numpy as np
from matplotlib import pyplot as plt

#【获取数据】
data = np.genfromtxt('./data/airfoil_self_noise.dat', delimiter='\t')
# 标准化数据
data = (data - data.mean(axis=0))/data.std(axis=0)


#【读取数据】
features, labels = nd.array(data[:1500,:-1]), nd.array(data[:1500,-1])


def f2d(x1,x2):
    return 0.1 * x1 **2 + 2*x2**2

#【定义模型】
def linreg(X,w,b):
    return nd.dot(X,w) + b

#【定义损失函数】
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape))**2/2
#【优化算法】
def gd(x1,x2):
    return (x1 - eta*0.2*x1, x2-eta*4*x2)

gamma,eta = 0.5,0.6

def momentum(x1,x2,v1,v2):
    v1 = gamma * v1 + eta*0.2*x1
    v2 = gamma * v2 + eta*4*x2
    return x1 - v1, x2 - v2,v1,v2

#【将每一步的优化结果存储起来】
def train(x1, x2,v1,v2):
    result = []
    for i in range(20):
        x1, x2, v1, v2= momentum(x1,x2,v1,v2)
        result.append((x1, x2))
    return result

x1,x2,v1,v2= -5,-3,0,0
result = train(x1,x2,v1,v2)

def show_trace():
    # 这个meshgrid函数是画二元函数图必备，用于将不等长的xy坐标数组统一
    x1, x2 = np.meshgrid(np.arange(-5.0,1.0,0.1), np.arange(-3.0,1.0,0.1))
    plt.contour(x1, x2, f2d(x1, x2))
    plt.xlabel('x1')
    plt.ylabel('x2')

    plt.plot(*zip(*result), '-o', color = 'orange')

    plt.show()

# show_trace()

def init_momentum_states():
    v_w = nd.zeros((features.shape[1],1))
    v_b = nd.zeros(1)
    return (v_w,v_b)

def sgd_momentum(params, states, hyperparams):
    for p,v in zip(params,states):
        v[:] = hyperparams['momentum']*v + hyperparams['lr']*p.grad
        p[:] -= v

def train_ch7(trainer,states,hyperparams, batch_size = 10,num_epochs = 2):
    net,loss = linreg,squared_loss
    #【初始化模型参数】
    w = nd.random.normal(scale=0.01,shape=(features.shape[1],1))
    b = nd.zeros(shape=(1,))

    w.attach_grad()
    b.attach_grad()

    ls =[]
    #【读取数据】
    data_iter = gdata.DataLoader(gdata.ArrayDataset(features, labels),batch_size,shuffle=True)

    start = 0

    for _ in range(num_epochs):
        start = time.time()
        for epoch_i, (X, y) in enumerate(data_iter):
            with autograd.record():
                y_hat =net(X,w,b)
                l = loss(y_hat, y).mean()
            l.backward()
            trainer([w,b],states, hyperparams)
            if (epoch_i+1)*batch_size%100 ==0:
                ls.append(loss(net(features,w,b), labels).mean().asscalar())
    print('loss %f,%f sec  per epoch ' %(ls[-1],(time.time()-start)/num_epochs))

    plt.plot(np.linspace(0,num_epochs,len(ls)),ls)
    plt.xlabel('epoch_i')
    plt.ylabel('loss')
    plt.show()

# train_ch7(sgd_momentum,init_momentum_states(),{'lr':0.004,'momentum':0.9})
# lr 和 momentum 的变化最好呈现出反比例关系


def train_gluon(trainer_name,hyperparams,batch_size):
    #【读取数据】
    data_iter = gdata.DataLoader(gdata.ArrayDataset(features, labels),batch_size, shuffle=True)
    #【定义模型】
    net = nn.Sequential()
    net.add(nn.Dense(1))
    #【初始化模型参数】
    net.initialize(init=init.Normal(sigma=0.01))
    #【定义损失函数】
    loss = gloss.L2Loss()
    #【定义优化算法】
    trainer = gluon.Trainer(net.collect_params(),trainer_name,hyperparams)
    #【训练模型】

    num_epochs = 2
    start = 0
    ls= []
    for _ in range(num_epochs):
        start = time.time()
        for epoch_i , (X, y) in enumerate(data_iter):
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            trainer.step(batch_size)
            if (epoch_i+1)*batch_size%100 ==0:
                ls.append(loss(net(features),labels).mean().asscalar())
    print('loss %f,%f sec per epoch' %(ls[-1], (time.time()-start)/len(ls)))

    plt.plot(np.linspace(0,num_epochs,len(ls)),ls)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

train_gluon('sgd',{'learning_rate':0.004,'momentum':0.9},10)
