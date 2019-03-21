from matplotlib import pyplot as plt
from mxnet.gluon import loss as gloss, data as gdata
from mxnet import autograd, nd

def xyplot(x_vals, y_vals, y_name):
    plt.plot(x_vals.asnumpy(), y_vals.asnumpy())
    plt.xlabel('x')
    plt.ylabel(y_name)
    plt.show()

x = nd.arange(-8, 8,0.1)
x.attach_grad()
with autograd.record():
    # 这里的record函数的作用不仅仅是求loss函数，而且还可以启动激活函数
    y = x.relu()

    # 这里的backward函数的作用就是将x的grad写入到上面的x.attach_grad()开辟的空间中去
    # 换句话说就是存储x.grad，使得后面的x.grad可以得到更新和使用！
    y.backward()

    z = x.sigmoid()
    z.backward()
# xyplot(x,x.grad,'grad of relu')
# xyplot(x,z,'sigmoid')
# xyplot(x,x.grad,'grad of sigmoid')

# 【生成数据】

train_data = gdata.vision.FashionMNIST(train=True)
test_data = gdata.vision.FashionMNIST(train=False)

# 【读取数据】
batch_size = 256
transformer = gdata.vision.transforms.ToTensor()

train_iter = gdata.DataLoader(train_data.transform_first(transformer), batch_size, shuffle=True)
test_iter = gdata.DataLoader(test_data.transform_first(transformer), batch_size ,shuffle= False)

# 【定义模型参数】

num_inputs = 28*28
num_outputs = 10
num_hiddens = 256

W1 = nd.random.normal(scale=0.01, shape=(num_inputs,num_hiddens))
W1.attach_grad()
b1 = nd.zeros(num_hiddens)
b1.attach_grad()
W2 = nd.random.normal(scale=0.01, shape=(num_hiddens, num_outputs))
W2.attach_grad()
b2 = nd.zeros(num_outputs)
b2.attach_grad()

params = [W1,b1,W2,b2]

# 【定义激活函数】

def relu(X):
    return nd.maximum(X, 0)


# 【定义模型】

def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(nd.dot(X, W1) + b1)
    return nd.dot(H,W2) + b2

# 【定义损失函数】

loss = gloss.SoftmaxCrossEntropyLoss()

# 【定义优化算法】

def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size


# 【训练模型】

num_epochs = 5
learning_rate = 0.5

# 定义归一化
def normal(y_hat, y):
   return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()

# 定义准确率的计算

def accuracy(data_iter, net):
    acc = 0
    for X, y in data_iter:
        acc += normal(net(X),y)
    return acc / len(data_iter)

for epoch  in range(num_epochs):
    loss_l_sum = 0
    train_acc_sum = 0

    for X, y in train_iter:
        with autograd.record():
            y_hat = net(X)
            l = loss(y_hat, y)
        l.backward()
        sgd(params, learning_rate, batch_size)

        loss_l_sum += l.mean().asscalar()
        train_acc_sum += normal(y_hat,y)
    test_acc = accuracy(test_iter,net)
    print('epoch %d loss %f train_acc %.3f test_acc %.3f' % (epoch,
                                                           loss_l_sum/len(train_iter),
                                                           train_acc_sum/len(train_iter),
                                                           test_acc))