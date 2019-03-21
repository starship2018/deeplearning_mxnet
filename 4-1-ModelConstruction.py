from mxnet import nd
from mxnet.gluon import nn


class MLP(nn.Block):
    def __init__(self,**kargs):
        # 在继承其他基类后，初始化函数中要记得调用基类的初始化函数
        # nn.Block.__init__(**kargs)
        super(MLP,self).__init__(**kargs)
        self.hidden = nn.Dense(256, activation='relu') # 隐藏层
        self.output = nn.Dense(10) # 输出层

    def forward(self, x):
        # 定义模型的向前计算，即根据输入X计算并返回所需要的模型输出
        return self.output(self.hidden(x))


x = nd.random.uniform(shape=(2,20))

# net = MLP()
# net.initialize()

class MySequential(nn.Block):
    def __init__(self,**kwargs):
        super(MySequential,self).__init__(**kwargs)

    def add(self,block):
        self._children[block.name] = block

    def forward(self, x):
        for block in self._children.values():
            x = block(x)
        return x

net = MySequential()
net.add(nn.Dense(256,activation='relu'))
net.add(nn.Dense(10))
net.initialize()
print(net(x))

#  未完待续！