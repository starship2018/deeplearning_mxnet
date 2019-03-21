from mxnet import gluon, init, nd, autograd
from mxnet.gluon import nn, data as gdata, loss as gloss
import mxnet as mx
import time

#【生成数据集】
train_data = gdata.vision.FashionMNIST(train=True)
test_data = gdata.vision.FashionMNIST(train=False)

#【读取数据集】
batch_size = 128
'''这个批量的大小也会影响到学习的准确率！'''
transformer = [gdata.vision.transforms.Resize(96), gdata.vision.transforms.ToTensor()]
transformer = gdata.vision.transforms.Compose(transformer)

train_iter = gdata.DataLoader(train_data.transform_first(transformer), batch_size, shuffle=True)
test_iter = gdata.DataLoader(test_data.transform_first(transformer), batch_size, shuffle=False)






def nin_block(num_channels, kernel_size, strides, padding):
    blk = nn.Sequential()
    blk.add(nn.Conv2D(num_channels, kernel_size, strides, padding, activation='relu'),
            # 卷积层无论你的输入有多少个通道
            nn.Conv2D(num_channels, kernel_size=1, activation='relu'),
            nn.Conv2D(num_channels, kernel_size=1, activation='relu'))
    return blk




# 【定义模型】
net = nn.Sequential()
net.add(nin_block(96,kernel_size=11, strides=4, padding=0),
        nn.MaxPool2D(pool_size=3, strides=2),
        nin_block(256, kernel_size=5, strides=1, padding=2),
        nn.MaxPool2D(pool_size=2, strides=2),
        nin_block(384, kernel_size=3, strides=1, padding=1),
        nn.MaxPool2D(pool_size=3, strides=2),
        nn.Dropout(0.5),
        nin_block(10, kernel_size=3, strides=1, padding=1),
        # 全局平均池化将窗口形状自动设置成输入的高和宽
        nn.GlobalAvgPool2D(),
        # 将四维的输出转化为二维的输出，其形状为（批量大小，10）
        nn.Flatten())


# X = nd.random.uniform(shape=(1,1,224,224))
# net.initialize()
# for layer in net:
#     X = layer(X)
#     print(layer.name, 'output shape:\t', X.shape)


# 【初始化模型参数】

net.initialize(init= init.Xavier(), ctx=mx.gpu())

#【定义损失函数】

loss = gloss.SoftmaxCrossEntropyLoss()

#【定义优化算法】

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.1})

# 【训练模型】

def batch_accuracy(y_hat, y):
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()

def all_accuracy(data_iter, net):
    acc = 0
    for X, y in data_iter:
        X, y = X.as_in_context(mx.gpu()), y.as_in_context(mx.gpu())
        acc += batch_accuracy(net(X), y)
    return acc / len(data_iter)

num_epochs = 5


def train():
    string = ''
    for epoch in range(num_epochs):
        train_l_sum, train_acc, start = 0, 0, time.time()
        for X, y in train_iter:
            X, y = X.as_in_context(mx.gpu()), y.as_in_context(mx.gpu())
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            trainer.step(batch_size)

            train_l_sum += l.mean().asscalar()
            train_acc += batch_accuracy(y_hat, y)
        test_acc = all_accuracy(test_iter, net)
        print('epoch %d train_loss %.3f train_acc %.3f test_acc %.3f time %.1f' % (epoch+1,
                                                                                   train_l_sum/len(train_iter),
                                                                                   train_acc / len(train_iter),
                                                                                   test_acc,
                                                                                   time.time() - start))
        string += str('epoch %d train_loss %.3f train_acc %.3f test_acc %.3f time %.1f \r' % (epoch+1,
                                                                                   train_l_sum/len(train_iter),
                                                                                   train_acc / len(train_iter),
                                                                                   test_acc,
                                                                                   time.time() - start))
    return string




with open('5-8-result.txt' ,'w') as f:
    f.write(train())


'''
    batch_size的大小也会影响到学习的准确率！256下的准确率要小于128
'''