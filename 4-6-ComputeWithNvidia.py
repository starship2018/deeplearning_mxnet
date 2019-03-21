import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn

# print(mx.cpu(),mx.gpu())

x = nd.array([1,2,3], ctx=mx.gpu())
y = nd.array([1,2,3,4,5,6,7,8,9], ctx=mx.gpu())
# print(x,x.context)
# print(y,y.context)

z =y.copyto(mx.gpu())
# print(z, z.context)



# 切记，在使用gpu进行运算时，参与运算的所有数据必须存储在同一块设备上
# 也就是说CPU上的数据只能和CPU之间进行运算，CPU上的数据和GPU上的数据相计算会报错！！
# print((z + 2 ).exp() * y)

net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(ctx=mx.gpu())

print(net(y))

print(net[0].weight.data())