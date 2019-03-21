from mxnet.gluon import loss as gloss, nn
import os
import subprocess
import time
from mxnet import nd, gluon, autograd

def data_iter():
    start = time.time()
    num_batches, batch_size = 100, 1024
    for i in range(num_batches):
        x = nd.random.normal(scale=0.01, shape=(batch_size,512))
        y = nd.ones((batch_size,))
        #将data_iter这个函数变成了迭代函数，可以直接进行for in 迭代！
        yield x, y
        if (i+1) %50 ==0:
            print('batch %d,time %f sec' %(i+1,time.time() - start))

net = nn.Sequential()
net.add(nn.Dense(2048,activation='relu'),
        nn.Dense(512,activation='relu'),
        nn.Dense(1))
net.initialize()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.005})
loss = gloss.L2Loss()

for X, y in data_iter():
    loss(net(X), y).wait_to_read()
    break

l_sum = 0
for X, y in data_iter():
    with autograd.record():
        l = loss(net(X), y)
    l_sum += l.mean().asscalar()
    l.backward()
    trainer.step(X.shape[0])
nd.waitall()
