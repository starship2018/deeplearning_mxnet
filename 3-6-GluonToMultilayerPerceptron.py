from mxnet import gluon, init,autograd
from mxnet.gluon import loss as gloss, nn ,data as gdata


# 【生成数据】

train_data = gdata.vision.FashionMNIST(train=True)
test_data = gdata.vision.FashionMNIST(train=False)

# 【读取数据】

batch_size = 256

transformer = gdata.vision.transforms.ToTensor()
train_iter = gdata.DataLoader(train_data.transform_first(transformer), batch_size, shuffle=True)
test_iter =gdata.DataLoader(test_data.transform_first(transformer), batch_size ,shuffle=False)

# 【定义模型】

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'),nn.Dense(10))

# 【初始化模型参数】

net.initialize(init.Normal(sigma=0.01))

# 【定义损失函数】

loss =gloss.SoftmaxCrossEntropyLoss()

# 【定义优化算法】
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.5})

# 【训练模型】

num_epochs = 20

# 定义归一化处理和计算准确率
def normal(y_hat, y):
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()

def accuracy(data_iter,net):
    acc = 0
    for X, y in data_iter:
        acc += normal(net(X), y)
    return acc / len(data_iter)

for epoch in range(num_epochs):
    loss_sum = 0
    train_acc_sum = 0
    for X ,y in train_iter:
        with autograd.record():
            y_hat = net(X)
            l = loss(y_hat, y)
        l.backward()
        trainer.step(batch_size)
        loss_sum += l.mean().asscalar()
        train_acc_sum += normal(y_hat, y)
    test_acc = accuracy(test_iter, net)
    print('epoch %d loss %.3f train_loss %.3f test_acc %.3f' % (epoch,
                                                                loss_sum/len(train_iter),
                                                                train_acc_sum/len(train_iter),
                                                                test_acc))


