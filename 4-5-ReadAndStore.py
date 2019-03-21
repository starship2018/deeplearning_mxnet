from mxnet import nd
from mxnet.gluon import nn

x = nd.ones(3)
# print(x,type(x))
# nd.save('x',x)

x2 = nd.load('x')
# print(x2)

y = nd.zeros(4)
nd.save('xy',[x,y])
x2,y2 = nd.load('xy')
# print((x2, y2))


class MLP(nn.Block):
    # 自定义模型
    def __init__(self,**kwargs):
        # 继承block的构造函数
        super(MLP, self).__init__(**kwargs)
        # 开始自定义网络层构造，省去了再添加layer的麻烦，内置两个全连接层
        self.hidden = nn.Dense(256,activation='relu')
        self.output = nn.Dense(10)


    def forward(self, x):
        return self.output(self.hidden(x))

net = MLP()
net.initialize()
x = nd.random.uniform(shape=(2,20))
y = net(x)

filename = 'mlp.params'
net.save_parameters(filename)

net2 = MLP()
net2.load_parameters(filename)

y2 = net2(x)

print(y == y2)