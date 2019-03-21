from mxnet import init, nd
from mxnet.gluon import nn


class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)

net = nn.Sequential()
net.add(nn.Dense(256,activation='relu'), nn.Dense(10))


# 在初始化函数中传入的是初始化器的实例，而不是函数本身！！！！！
net.initialize(MyInit())

x =nd.random.uniform(shape=(2,20))
y = net(x)

