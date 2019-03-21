from mxnet import gluon, init, nd, autograd
from mxnet.gluon import data as gdata, nn, loss as gloss
import mxnet as mx
import os
import sys
import time

# 【生成数据集】

train_data = gdata.vision.FashionMNIST(train = True)
test_data = gdata.vision.FashionMNIST(train = False)

# 【读取数据集】

batch_size = 128
transformer = [gdata.vision.transforms.Resize(224), gdata.vision.transforms.ToTensor()]
transformer = gdata.vision.transforms.Compose(transformer)

train_iter = gdata.DataLoader(train_data.transform_first(transformer), batch_size=batch_size, shuffle= True)
test_iter = gdata.DataLoader(test_data.transform_first(transformer), batch_size=batch_size, shuffle= False)


# 【定义模型】
net = nn.Sequential()
net.add(nn.Conv2D(96, kernel_size=11, strides=4, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=3),
        nn.Conv2D(256, kernel_size=5, padding=2, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        nn.Conv2D(384, kernel_size=3, padding=1,activation='relu'),
        nn.Conv2D(384,kernel_size=3, padding=1, activation='relu'),
        nn.Conv2D(256, kernel_size=3,padding=1, activation='relu'),
        nn.MaxPool2D(pool_size=3,strides=2),
        nn.Dense(4096,activation='relu'),nn.Dropout(0.5),
        nn.Dense(4096,activation='relu'),nn.Dropout(0.5),
        nn.Dense(10))

# X = nd.random.uniform(shape=(1,1,224,224))
# net.initialize()
# for layer in net:
#     X = layer(X)
#     print(layer.name,'output shape:\t',X.shape)

#【初始化模型参数】
net.initialize(init= init.Xavier(), ctx=mx.gpu())

#【定义损失函数】
loss = gloss.SoftmaxCrossEntropyLoss()

#【定义优化算法】
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.01})

#【训练模型】

# 批量准确率计算
def batch_accuracy(y_hat, y):
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()

# 总体准确率计算
def all_accuracy(data_iter, net):
    acc = 0
    for X, y in data_iter:
        X, y = X.as_in_context(mx.gpu()), y.as_in_context(mx.gpu())
        acc += batch_accuracy(net(X), y)
    return acc/ len(data_iter)



num_epochs = 5

def train():
    string = ''
    for epoch in range(num_epochs):
        train_l_sum, train_acc, start = 0, 0, time.time()
        for X, y in train_iter:
            X , y = X.as_in_context(mx.gpu()), y.as_in_context(mx.gpu())
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            trainer.step(batch_size)

            train_l_sum += l.mean().asscalar()
            train_acc += batch_accuracy(y_hat, y)
        test_acc = all_accuracy(test_iter, net)
        print('epoch %d loss %.3f train_acc %.3f test_acc %.3f time %.1f' % (epoch+1, train_l_sum/len(train_iter),
                                                                   train_acc/len(train_iter),
                                                                   test_acc, time.time() - start))
        string += str('epoch %d loss %.3f train_acc %.3f test_acc %.3f time %.1f' % (epoch+1, train_l_sum/len(train_iter),
                                                                   train_acc/len(train_iter),
                                                                   test_acc, time.time() - start))
    return string


with open('5-6-result.txt', 'w') as f:
    f.write(train())


'''
    以后遇到海量的计算时，将打印信息str()后返回来，再使用open将其写入文件中！
'''
