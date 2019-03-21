from mxnet import gluon, init, nd,autograd
from mxnet.gluon import nn, data as gdata, loss as gloss
import mxnet as mx
import time



#【生成数据集】
train_data = gdata.vision.FashionMNIST(train = True)
test_data = gdata.vision.FashionMNIST(train= False)

#【读取数据集】

batch_size = 256
transformer = [gdata.vision.transforms.Resize(96), gdata.vision.transforms.ToTensor()]
transformer = gdata.vision.transforms.Compose(transformer)

train_iter = gdata.DataLoader(train_data.transform_first(transformer), batch_size, shuffle=True)
test_iter = gdata.DataLoader(test_data.transform_first(transformer), batch_size, shuffle=False)


# 残差块的定义
class Residual(nn.Block):
    def __init__(self, num_channels, use_1x1conv = False, strides = 1, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1, strides=strides) # 不变形卷积
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1, strides=strides) # 使用1x1卷积，以调整通道数
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    # 这个向前传播的函数实质上就是核心计算函数！
    def forward(self, x):
        Y = nd.relu(self.bn1(self.conv1(x)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            x = self.conv3(x)
        return nd.relu(Y + x)


#【定义模型】
net = nn.Sequential()
net.add(nn.Conv2D(64, kernel_size=7, padding=3, strides=3),
        nn.BatchNorm(),
        nn.Activation('relu'),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1))

def resnet_block(num_channels, num_residuals, first_block=False):
    blk = nn.Sequential()
    for i in range(num_residuals):
        if i ==0 and not first_block:
            blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
        else:
            blk.add(Residual(num_channels))
    return blk

net.add(resnet_block(64,2,first_block=True),
        resnet_block(128,2),
        resnet_block(256,2),
        resnet_block(512,2),
        nn.GlobalAvgPool2D(),
        nn.Dense(10))


#【初始化模型参数】
net.initialize(init=init.Xavier(), ctx=mx.gpu())

#【定义损失函数】
loss = gloss.SoftmaxCrossEntropyLoss()

#【定义优化算法】
trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.05})

#【训练模型】
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

with open('5-11-result.txt', 'w') as f:
    f.write(train())
