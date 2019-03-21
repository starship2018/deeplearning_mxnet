from mxnet import gluon, init, autograd, nd
from mxnet.gluon import nn, data as gdata, loss as gloss
import time
import mxnet as mx



#【生成数据集】
train_data = gdata.vision.FashionMNIST(train = True)
test_data = gdata.vision.FashionMNIST(train= False)

#【读取数据集】

batch_size = 128
transformer = [gdata.vision.transforms.Resize(96), gdata.vision.transforms.ToTensor()]
transformer = gdata.vision.transforms.Compose(transformer)

train_iter = gdata.DataLoader(train_data.transform_first(transformer), batch_size, shuffle=True)
test_iter = gdata.DataLoader(test_data.transform_first(transformer), batch_size, shuffle=False)




def conv_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(),
            nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=3, padding=1))
    return blk


# 由于densenet的计算方式需要定制，所以将计算和结构一起定义在类中！

class DenseBlock(nn.Block):
    def __init__(self, num_convs, num_channels, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        self.net = nn.Sequential()

        for _ in range(num_convs):
            self.net.add(conv_block(num_channels))

    def forward(self, x):
        for blk in self.net:
            y = blk(x)
            x = nd.concat(x,y,dim =1)
        return x

def transition_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(), nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=1),
            nn.AvgPool2D(pool_size=2, strides=2))
    return blk


#【定义模型】
net = nn.Sequential()
net.add(nn.Conv2D(64,kernel_size=7,padding=3,strides=2),
        nn.BatchNorm(),
        nn.Activation('relu'),
        nn.MaxPool2D(pool_size=3,strides=2,padding=1))

num_channels, grow_rate = 64, 32
num_convs_in_dense_blocks = [4,4,4,4]

for i,num_convs in enumerate(num_convs_in_dense_blocks):
    net.add(DenseBlock(num_convs, grow_rate))
    num_channels += num_convs * grow_rate
    if i != len(num_convs_in_dense_blocks) - 1:
        net.add(transition_block(num_channels // 2))

net.add(nn.BatchNorm(),
        nn.Activation('relu'),
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

with open('5-12-result.txt', 'w') as f:
    f.write(train())
