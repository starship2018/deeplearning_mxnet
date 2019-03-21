from mxnet import autograd, gluon, init, nd
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



def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 通过auto grad来判断当前模式为训练模式还是预测模式
    if not autograd.is_training():
        #在预测模式下,直接使用传入的移动平均所得的均值和方差

        # 标准化
        X_hat = (X - moving_mean) / nd.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2,4)
        if len(X.shape) == 2: # 若为2维，则上面连接的是全连接层
            # 在使用全连接层的情况下，计算特征维上的均值和方差
            mean = X.mean(axis = 0)
            var =( (X - mean) **2 ).mean(axis = 0) # axis = 0 就是指在竖直方向上
        else: # 若不是二维，则上面连接的就是卷积层

            # 在使用二维卷积层的情况下，计算通道维上（axis=1）的均值和方差。
            # 这里我们需要保持X的形状以便后面可以做广播运算


            # 卷积层中的输入由4个维度，从1到4分别是 样本序号，通道数，高，宽
            # 这里对通道数这一维并不做均值计算

            mean = X.mean(axis = (0,2,3), keepdims = True)
            var = ((X - mean)**2).mean(axis = (0,2,3), keepdims = True)
        # 训练模式下用当前的均值和方差做标准化
        X_hat = (X - mean) / nd.sqrt(var + eps)
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta
    return Y, moving_mean, moving_var

class BatchNorm(nn.Block):
    def __init__(self,num_features, num_dims, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)
        if num_dims ==2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉升和偏移参数，分别初始化为0 和 1
        self.gamma =self.params.get('gamma', shape= shape, init = init.One())
        self.beta = self.params.get('beta', shape = shape, init = init.Zero())
        #不参与求梯度和迭代的变量，全在CPU上初始化成 0
        self.moving_mean = nd.zeros(shape)
        self.moving_var = nd.zeros(shape)

    def forward(self, x):
        # 若x不在CPU上，将moving_mean和moving_var复制到x所在的设备之上
        if self.moving_mean.context != x.context:
            self.moving_mean = self.moving_mean.copyto(x.context)
            self.moving_var = self.moving_var.copyto(x.context)

        # 保存更新过的moving_mean和moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(x, self.gamma.data(), self.beta.data(), self.moving_mean,
                                                          self.moving_var, eps=1e-5, momentum=0.9)
        return Y

#【定义模型】
net = nn.Sequential()
net.add(nn.Conv2D(6, kernel_size=5),
        BatchNorm(6,num_dims=4),


        #以往的激活函数都是在Conv2D或者是Dense中实现的，这里要求出现在一个特殊的位置，所以是直接调用的nn中的激活函数#
        nn.Activation('sigmoid'),

        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(16,kernel_size=5),
        BatchNorm(16,num_dims=4),
        nn.Activation('sigmoid'),
        nn.MaxPool2D(pool_size=2,strides=2),
        nn.Dense(120),
        BatchNorm(120,num_dims=2),
        nn.Activation('sigmoid'),
        nn.Dense(84),
        BatchNorm(84,num_dims=2),
        nn.Activation('sigmoid'),
        nn.Dense(10))

#【初始化模型参数】
net.initialize(init=init.Xavier(),ctx=mx.gpu())

#【定义损失函数】
loss = gloss.SoftmaxCrossEntropyLoss()

#【定义优化算法】
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':1})

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

with open('5-10-result.txt', 'w') as f:
    f.write(train())