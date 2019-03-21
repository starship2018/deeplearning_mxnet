from mxnet import autograd, init, gluon
from mxnet.gluon import loss as gloss, nn, data as gdata


# 【生成数据】

train_data = gdata.vision.FashionMNIST(train=True)
test_data = gdata.vision.FashionMNIST(train=False)


# 【读取数据】

batch_size = 256
transformer = gdata.vision.transforms.ToTensor()

train_iter = gdata.DataLoader(train_data.transform_first(transformer), batch_size, shuffle= True)
test_iter = gdata.DataLoader(test_data.transform_first(transformer), batch_size, shuffle= False)

# 【定义初始化模型】
net = nn.Sequential()
net.add(nn.Dense(10))

# 【初始化模型参数】

net.initialize(init.Normal(sigma= 0.01))

#【损失函数】

loss = gloss.SoftmaxCrossEntropyLoss()

# 【定义优化算法】

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

# 【训练模型】

# softmax结果归一化
def normalize(y_hat, y):
    # 将计算结果归一化（归一化后的结果除以数据长度就是准确率！）
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()

# 对测试集专门计算准确率
def test_accuracy(test_iter, net):
    nor = 0
    for X, y in test_iter:
        nor += normalize(net(X), y)
    return nor/len(test_iter)


num_epochs = 5

def train(num_epochs, train_iter, test_iter, batch_size, net, loss, trainer):
    for epoch in range(num_epochs):
        train_l_sum = 0
        train_acc_sum = 0

        for X, y in train_iter:
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            trainer.step(batch_size)
            train_l_sum += l.mean().asscalar()
            train_acc_sum += normalize(y_hat, y)
            # print('w %.3f  b %.3f' % (net[0].weight.data(), net[0].bias.data()))
        test_acc = test_accuracy(test_iter, net)
        print('epoch %d loss %.3f  train_acc %.3f test_acc %.3f' % (epoch,
                                                                    train_l_sum/len(train_iter),
                                                                    train_acc_sum/len(train_iter),
                                                                    test_acc))

train(num_epochs, train_iter, test_iter, batch_size, net, loss, trainer)

