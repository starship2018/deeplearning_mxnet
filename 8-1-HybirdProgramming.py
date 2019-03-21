from mxnet import nd, sym
from mxnet.gluon import nn
import time




def get_net():
    net = nn.HybridSequential()
    net.add(nn.Dense(256,activation='relu'),
            nn.Dense(128,activation='relu'),
            nn.Dense(2))
    net.initialize()
    return net

x = nd.random.normal(shape=(1,512))
# net = get_net()
# print(net(x))
# net.hybridize()
# print(net(x))

def benchmark(net,x):
    start =  time.time()
    for i in range(1000):
        _ = net(x)
    nd.waitall() # 等待所有计算完成时方便计时
    return time.time() - start

net = get_net()
print('before hybrizing: %.4f sec' %(benchmark(net,x))) # before hybrizing: 0.4130 sec
net.hybridize()
print('after hybrizing: %.4f sec' %(benchmark(net,x)))  # after hybrizing: 0.1700 sec