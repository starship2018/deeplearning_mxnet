import time
from mxnet.gluon import data as gdata
import matplotlib.pyplot as plt
from mxnet import autograd, nd

# import gluonbook as gb


# 【生成数据集】
mnist_train = gdata.vision.FashionMNIST(train=True)
mnist_test = gdata.vision.FashionMNIST(train=False)



features, labels = mnist_train[0]


# 根据int32类型的label值返回对应的种类信息
def get_fashion_mnist_labels(label):
    text_labels = ['T-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return text_labels[label]


# 展示图像和标签信息
def show_fashion_mnist(images, labels):
    plt.imshow(images.reshape((28,28)).asnumpy())
    plt.show()
    print(get_fashion_mnist_labels(labels))

# X, y = mnist_train[52123]
# show_fashion_mnist(X, y)



# 【读取小批量】
batch_size = 256
transformer = gdata.vision.transforms.ToTensor() # 定义一个ToTensor转化器

# transform_first()函数用于对所有的数据样本应用某种转化器
train_iter = gdata.DataLoader(mnist_train.transform_first(transformer), batch_size, shuffle=True)
test_iter = gdata.DataLoader(mnist_test.transform_first(transformer), batch_size, shuffle=False)


start = time.time()

# 测试读取测试机所需的时间！
for X, y in train_iter:
    continue
# print('%.2f' % (time.time()-start))



# 【初始化模型参数】

num_input = 28*28
num_output = 10

# 这里将图像的features化为一个横向量，那么与之对应相乘的权重的高度和其长度相等！
w = nd.random.normal(scale=0.01, shape=(num_input, num_output))
# 又因为图像特征是一个横向量，那么Xw的结果也务必是一个横向量！
b = nd.zeros(num_output)

# 开辟梯度空间
w.attach_grad()
b.attach_grad()

# 实现Softmax运算

def softmax(X):
    X_EXP = X.exp()
    sum = X_EXP.sum(axis=1, keepdims= True)
    return X_EXP/sum

# 【定义模型算法】

def net(X):
    return softmax(nd.dot(X.reshape(-1, num_input), w) + b)

# 【定义损失函数】

def cross_entropy(y_hat, y):
    # 这里的y_hat是序号，在y中按照Y_Hat的序号进行取值！
    return  - nd.pick(y_hat, y).log()

# 【计算出分类的准确率】

def accuracy(y_hat, y):
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()

def evaluate_accuracy(data_iter, net):
    acc = 0
    for X, y in data_iter:
        acc += accuracy(net(X), y)
    return acc / len(data_iter)

# 【定义优化算法】

def sdg(params,lr, batch_size):
    for param in params:
        param[:] = param - lr*param.grad/batch_size




# 【训练模型】
num_epochs, lr = 5, 0.1

def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params = None, lr = None, trainer = None):
    for epoch in range(num_epochs):
        train_l_sum = 0
        train_acc_sum = 0

        for X, y in train_iter:
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            if trainer is None:
                sdg(params, lr, batch_size)
            else:
                trainer.step(batch_size)
            train_l_sum += l.mean().asscalar()
            train_acc_sum += accuracy(y_hat, y)
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d ,loss %f, train acc %.3f, test acc %.3f' % (epoch,train_l_sum/len(train_iter),
                                                                    train_acc_sum/len(train_iter),
                                                                    test_acc))

train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [w,b], lr)