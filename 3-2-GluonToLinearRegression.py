from mxnet import autograd, nd, init, gluon
from mxnet.gluon import data as gdata
from mxnet.gluon import nn
from mxnet.gluon import loss as loss
import numpy as np

num_inputs = 2
num_examples = 2000

true_w = [2, -3.4]
true_b = 4.2

# 【生成数据集】
features = nd.random.normal(scale=1, shape = (num_examples, num_inputs))
# labels = features[:,0] * true_w[0] + features[:,1] * true_w[1] + true_b
labels = nd.dot(features, nd.array(np.transpose(true_w))) + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)

# 【读取数据】


batch_size = 10
# 利用gluon 中的data模块将特征集和标签集整合为一个完整的数据集
dataset = gdata.ArrayDataset(features, labels)
# 随机读取小批量(可迭代!)
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)

# 试读取数据集

for X, y in data_iter:
    # print(X, y)
    break


# 【定义模型】

net = nn.Sequential()

# 因为线性回归输出层中的神经元和输入层中各个输入完全连接,所以线性回归的输出层又叫
# 全连接层

net.add(nn.Dense(1))


# 【初始化模型参数】


# 默认初始化的是权重参数,采用标准差为0.01的正态分布
# 偏差参数默认初始化为0

# 这时net已经有了w和b，后续操作中只需要将输入输入层即可
# 这个net(X)的操作就是求出预测值的过程！
net.initialize(init.Normal(sigma = 0.01))


# 【定义损失函数】

loss = loss.L2Loss() # 平方损失又称为L2范数损失

# 【定义优化算法】

# 定义一个模型参数优化算法一共需要4个参数，w,b,learning_rate,batch_size
# 其中batch_size这个参数在最后的step函数中给出！
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})


# 【训练模型】

num_epochs = 3

for epoch in range(num_epochs):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        # 打印net第一层中的权重和偏差信息！
        print(net[0].weight.data(), net[0].bias.data())
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print('epoch %d, loss %f' % (epoch, l.mean().asnumpy()))