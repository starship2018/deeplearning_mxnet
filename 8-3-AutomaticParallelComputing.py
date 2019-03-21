import mxnet as mx
from mxnet import nd
import time

def run(x):
    return [nd.dot(x,x) for _ in range(10)]

x_cpu = nd.random.uniform(shape=(2000,2000))
x_gpu = nd.random.uniform(shape=(6000,6000), ctx=mx.gpu())

run(x_cpu)
run(x_gpu)
nd.waitall()


def benchmark(net,x):
    start =  time.time()
    for i in range(1000):
        _ = net(x)
    nd.waitall() # 等待所有计算完成时方便计时
    return time.time() - start

#
# with benchmark('Run on CPU.'):
#     run(x_cpu)
#     nd.waitall()
#
# with benchmark('Then run on GPU.'):
#     run(x_gpu)
#     nd.waitall()