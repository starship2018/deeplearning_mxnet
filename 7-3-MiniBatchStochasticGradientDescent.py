from mxnet import autograd, gluon, init, nd
from mxnet.gluon import nn, data as gdata ,loss as gloss
import time
import numpy as np
from matplotlib import pyplot as plt


def get_data():
    # 关于相对路径！上一级 ../+path  同一级 ./+path
    data = np.genfromtxt('./data/airfoil_self_noise.dat', delimiter='\t')
    # 对数据进行标准化处理

    # 这里的mean和std函数中参数axis=0表示对data进行求平均值计算和标准差计算后输出的结果只有一行！
    # 若axis=1，表示计算后的结果只有一列！
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    # 返回data前5列，作为特征。最后一列作为标签！都只要前1500个样本

    # 再返回之前需要使用nd.array包装成NDArray类型的数据
    return nd.array(data[:1500,:-1]),nd.array(data[:1500,-1])

features, labels = get_data()
# print(features.shape, type(features))

#【定义模型】
def linreg(X,w,b):
    return nd.dot(X,w) + b
#【定义损失函数】
def squared_loss(y_hat, y):
    # 注意！这里的y_hat的shape是（1500，1），是一个二维数组，是竖着的
    # 这里的y的shape是（1500，），是一个数组，是横着的
    # 如果直接使用y_hat - y得到的是一个1500*1500的矩阵，二维矩阵和这种一维矩阵进行加减运算一定要记得将
    # 一位矩阵变形为二维矩阵，否者计算结果会极度变形！
    return (y_hat - y.reshape(y_hat.shape))**2/2  # y_hat - y.reshape(y_hat.shape) 这里我们先不要对y进行变形!

#【定义优化算法】
def sgd(params, states, hyperparams):
    for param in params:
        param[:] -= hyperparams['lr'] * param.grad


def train_ch7(trainer, states, hyperparams, features, labels, batch_size=10, num_epochs=2):
    net, loss = linreg, squared_loss
    #【初始化模型参数】
    w = nd.random.normal(scale=0.01, shape=(features.shape[1], 1))
    b = nd.zeros(shape=(1,))

    w.attach_grad()
    b.attach_grad()

    def eval_loss():
        return loss(net(features,w,b),labels).mean().asscalar()

    ls = []

    data_iter = gdata.DataLoader(gdata.ArrayDataset(features, labels),batch_size,shuffle=True)

    start = 0

    for _ in range(num_epochs):
        start = time.time()
        # 这里使用for enumerate来迭代data_iter，除了和直接迭代data_iter得到X，y之外，还能额外得到索引标号！
        for batch_i, (X, y) in enumerate(data_iter):
            with autograd.record():
                y_hat = net(X,w,b)
                # print(y_hat.shape, y.shape)
                # print(type(y_hat),type(y))
                l = loss(y_hat, y).mean()
            l.backward()
            trainer([w, b],states, hyperparams)
            # print(batch_i+1)
            if(batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())
    print('loss %f, %f sec per epoch' % (ls[-1], (time.time() - start)/num_epochs))
    plt.plot(np.linspace(0, num_epochs, len(ls)),ls)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

def train_sgd(lr, batch_size, num_epochs =2):
    train_ch7(sgd,None,{'lr':lr},features,labels,batch_size,num_epochs)





# train_sgd(0.05,10,2)

'''
    这里的batch_size 是指将原始的数据以batch_size为大小进行分批次处理，
    我们知道现在的data_iter里面一共有1500个数据
    当设置batch_size为1时，for in data_iter可以执行1500/1=1500次，
    每次迭代返回的X，y都只有batch_size个样本，所以在一次epoch中，
    参数学习进化了1500次
    当batch_size为1500时，for in data_iter只能执行1500/1500=1次，
    这时返回的X，y有1500行全部数据，在一次epoch中，参数同时在1500
    个数据下面学习，最终能进化一次！
'''

#【定义模型】
net = nn.Sequential()
net.add(nn.Dense(1))

#【初始化模型参数】
net.initialize(init=init.Normal(sigma=0.01))

#【定义损失函数】
loss = gloss.L2Loss()


def train_gluon_ch7(trainer_name, trainer_hyperparams, features, labels, batch_size = 10, num_epochs = 2):
    #【定义优化算法】
    trainer = gluon.Trainer(net.collect_params(), trainer_name, trainer_hyperparams)
    #【生成数据】
    data = gdata.ArrayDataset(features, labels)
    #【读取数据】
    data_iter = gdata.DataLoader(data, batch_size, shuffle=True)
    #【训练模型】

    ls = []

    for _ in range(num_epochs):
        start = time.time()
        for batch_i, (X,y) in enumerate(data_iter):
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            trainer.step(batch_size)
            if (batch_i+1)*batch_size%100 ==0:
                # 使用当前得到的参数来计算所有的样本的损失函数
                ls.append(loss(net(features), labels).mean().asscalar())
        print('loss:%f , %f sec per epoch' % (ls[-1], time.time() - start))
    # np的两个函数arange(),linespace()都可以用来创建一维数组来构建坐标轴
    # 区别在与arange()的参数是起点，终点，精度（越小越圆滑）  linespace()的参数是起点，终点，点个数
    #个人建议：若要将数组元素在图上表示观察，请使用linespace;若是想单纯画出一个函数图像，请使用arange,调整第三个参数以表示画图的精度
    plt.plot(np.linspace(0, num_epochs, len(ls)),ls)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

train_gluon_ch7('sgd', {'learning_rate':0.05}, features,labels)

'''
    随机梯度下降，批量梯度下降，小批量随机梯度下降的差别就在batch_size的大小上。
    当batch_size为1时。每一次迭代数据集只收集一个样本进行训练
    当batch_size为num_examples时，每一次迭代数据集收集全部样本进行训练
    当batch_size >1 && batch_size <<num_examples时，即为小批量随机梯度下降
'''
