from mxnet import gluon, init, nd, autograd
from mxnet.gluon import nn, loss as gloss, data as gdata
import mxnet as mx
import time


#【生成数据集】
train_data = gdata.vision.FashionMNIST(train = True)
test_data = gdata.vision.FashionMNIST(train= False)

#【读取数据集】

batch_size = 128
transformer = [gdata.vision.transforms.Resize(96), gdata.vision.transforms.ToTensor()]
transformer = gdata.vision.transforms.Compose(transformer)

train_iter = gdata.DataLoader(train_data.transform_first(transformer), batch_size, shuffle=True)
test_iter = gdata.DataLoader(test_data.transform_first(transformer), batch_size, shuffle=False)


class Inception(nn.Block):
    def __init__(self,c1,c2,c3,c4,**kwargs):
        super(Inception, self).__init__(**kwargs)

        # 在这种模式下，将神经层写在属性里面

        # 线路1 单 1*1 卷积层
        self.p1_1 = nn.Conv2D(c1, kernel_size=1, activation='relu')

        # 线路2 1*1 卷积层后接 3*3 卷积层
        self.p2_1 = nn.Conv2D(c2[0], kernel_size=1, activation='relu')
        self.p2_2 = nn.Conv2D(c2[1], kernel_size=3, padding=1,activation='relu')


        # 线路3 1*1卷积层后接 5*5 卷积层
        self.p3_1 = nn.Conv2D(c3[0], kernel_size=1,activation='relu')
        self.p3_2 = nn.Conv2D(c3[1], kernel_size=5, padding=2, activation='relu')

        # 线路4 3*3最大池化层后接 1*1卷积层
        self.p4_1 = nn.MaxPool2D(pool_size=3, strides=1, padding=1)
        self.p4_2 = nn.Conv2D(c4, kernel_size=1,activation='relu')

    def forward(self, x):
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))
        return nd.concat(p1,p2,p3,p4,dim = 1)

# 【定义模型】

b1 = nn.Sequential()
b1.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3, activation='relu'),
       nn.MaxPool2D(pool_size=3, strides=2, padding=1))

b2 = nn.Sequential()
b2.add(nn.Conv2D(54, kernel_size=1),
       nn.Conv2D(192, kernel_size=3, padding=1),
       nn.MaxPool2D(pool_size=3, strides=2, padding=1))

b3 = nn.Sequential()
b3.add(Inception(64,(96,128),(16,32),32),
       Inception(128,(128,192),(32,96), 64),
       nn.MaxPool2D(pool_size=3, strides=2, padding=1))

b4 = nn.Sequential()
b4.add(Inception(192,(96,208),(16,48),64),
       Inception(160,(112,224),(24,24),64),
            Inception(128,(128,256),(24,64),64),
                 Inception(112,(144,288),(32,64),64),
                 Inception(256,(160,320),(32,128),128),
                nn.MaxPool2D(pool_size=3,strides=2,padding=1)
                 )

b5 = nn.Sequential()
b5.add(Inception(256,(160,320),(32,128),128),
       Inception(384,(192,384),(48,128),128),
       nn.GlobalAvgPool2D())

net = nn.Sequential()
net.add(b1,b2,b3,b4,b5,nn.Dense(10))

#【初始化模型参数】

net.initialize(init=init.Xavier(), ctx=mx.gpu())

# 【定义损失函数】
loss = gloss.SoftmaxCrossEntropyLoss()

# 【定义优化算法】
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.1})

# 【开始训练模型】

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
        train_l_loss, train_acc, start = 0, 0, time.time()
        for X, y in train_iter:
            X, y = X.as_in_context(mx.gpu()), y.as_in_context(mx.gpu())
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            trainer.step(batch_size)

            train_l_loss += l.mean().asscalar()
            train_acc += batch_accuracy(y_hat, y)
        test_acc = all_accuracy(test_iter, net)
        print('epoch %d train_loss %.3f train_acc %.3f test_acc %.3f time %.1f' % (epoch+1,
                                                                          train_l_loss/len(train_iter),
                                                                          train_acc/len(train_iter),
                                                                          test_acc,
                                                                                   time.time() - start))
        string += str('epoch %d train_loss %.3f train_acc %.3f test_acc %.3f time %.1f \r' % (epoch+1,
                                                                          train_l_loss/len(train_iter),
                                                                          train_acc/len(train_iter),
                                                                          test_acc,
                                                                                   time.time() - start))
    return string

with open('5-9-result.txt', 'w') as f:
    f.write(train())




