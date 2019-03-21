import mxnet as mx
from mxnet import autograd,gluon,init,nd
from mxnet.gluon import loss as gloss, nn, data as gdata
import time



# 【获取数据集】
train_data = gdata.vision.FashionMNIST(train=True)
test_data = gdata.vision.FashionMNIST(train=False)



# 【读取数据】
batch_size = 256
transformer = gdata.vision.transforms.ToTensor()
train_iter = gdata.DataLoader(train_data.transform_first(transformer),batch_size=batch_size, shuffle=True)
test_iter = gdata.DataLoader(test_data.transform_first(transformer),batch_size=batch_size, shuffle=False,)



# 【定义模型】
net = nn.Sequential()
# 在添加层的时候只需要指定输出的信息 卷积层需要输出的通道数 全连接层需要输出的神经元个数
net.add(nn.Conv2D(channels=6, kernel_size=5, activation='sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=5, activation='sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        # 这里的全连接层和我们之前的WX+b是一样的，最后通过W的列数来控制输出的个数
        nn.Dense(120, activation='sigmoid'),
        nn.Dense(84,activation='sigmoid'),
        # 最后一层不需要激活函数
        nn.Dense(10))




# 【初始化模型参数】
net.initialize(init = init.Xavier(), ctx=mx.gpu())




# 【定义损失函数】
loss = gloss.SoftmaxCrossEntropyLoss()



# 【定义优化算法】
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.9})



# 【定义准确率的计算】
def batch_accuracy(y_hat, y):
    # 将判断的对错结果以0和1表示出来，并求出了均值以表示该批量下的准确率！
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()

def all_accuracy(data_iter, net):
    acc = nd.array([0],ctx=mx.gpu())
    for X ,y in data_iter:
        X, y = X.as_in_context(mx.gpu()), y.as_in_context(mx.gpu())
        acc += batch_accuracy(net(X), y)
    return acc.asscalar() / len(data_iter)




# 【训练模型】

num_epochs = 5

def train_ch5():
    for epoch in range(num_epochs):
        train_l_sum, train_acc, start = 0, 0, time.time()
        for X, y in train_iter:
            X = X.as_in_context(mx.gpu())
            y = y.as_in_context(mx.gpu())
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            trainer.step(batch_size)

            train_l_sum += l.mean().asscalar()
            train_acc += batch_accuracy(y_hat, y)
        test_acc = all_accuracy(test_iter,net)
        print('epoch %d loss %.3f train_acc %.3f test_acc %.3f time %.1f' % (epoch+1, train_l_sum / len(train_iter),
                                                              train_acc / len(train_iter),
                                                              test_acc,time.time() - start))
        return str('epoch %d loss %.3f train_acc %.3f test_acc %.3f time %.1f' % (epoch, train_l_sum / len(train_iter),
                                                              train_acc / len(train_iter),
                                                              test_acc,time.time() - start))

# net.initialize(force_reinit=True,ctx=mx.gpu(),init=init.Xavier())
with open('5-5-result.txt', 'w') as f:
    f.write(train_ch5())


'''
若在gpu上完成全部的计算，需要注意的是
1.在第一次初始化模型参数时指定gpu！
2.在迭代所有的批量数据的时候将X和y全部转存到gpu上去！ 记住as_in_context(mx.gpu()) 这个函数！
3.在整体精度计算中所迭代出来的X，y也需要转存到gpu上去
'''