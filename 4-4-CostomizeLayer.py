from mxnet import gluon, nd
from mxnet.gluon import nn


class CenteredLayer(nn.Block):
    def __init__(self,**kwargs):
        super(CenteredLayer,self).__init__(**kwargs)

    def forward(self, x):
        return x- x.mean()

layer = CenteredLayer()
# print(layer(nd.array([1,2,3,4,5])))

class MyDense(nn.Block):
    def __init__(self,units,in_units,**kwargs):
        super(MyDense, self).__init__(**kwargs)
        self.weight = self.params.get('weight',shape=(in_units,units))
        self.bias = self.params.get('bias', shape=(units,))

    def forward(self, x):
        linear = nd.dot(x,self.weight.data()) + self.bias.data()
        return nd.relu(linear)

dense = MyDense(3,5)
# print(dense.params)
dense.initialize()

# print(dense(nd.random.uniform(shape=(2,5))))

net = nn.Sequential()
net.add(MyDense(8,64),MyDense(1,8))
net.initialize()
print(net(nd.random.uniform(shape=(2,64))))