from mxnet import nd, autograd
from time import time
from matplotlib import pyplot as plt
import random



# 比较矢量计算和标量积算的速度的差异性！
def compute_rate():

    a = nd.ones(shape=1000)
    b = nd.ones(shape=1000)
    start = time()
    c = nd.zeros(shape=1000)
    for i in range(1000):
        c[i] = a[i] + b[i]
    print('所有元素逐一进行标量计算！', time() - start)
    start = time()
    c = a + b
    print('向量直接进行标量运算！', time() - start)


# compute_rate()

# 【生成数据】
num_inputs = 2
num_examples = 1000
true_w = [2,-3.4]
true_b = 4.2

features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0]*features[:, 0]+true_w[1]*features[:, 1]+true_b

labels += nd.random.normal(scale=0.01, shape=labels.shape)




# 画出散点图！
# 这里使用的asnumpy是mxnet自带的函数，将自己转化为numpy的格式！
plt.scatter(features[:, 1].asnumpy(), labels.asnumpy(), 1)
plt.show()



# 读取一批次的数据
def data_iter (batch_size, features, labels):
    num_examples = len(features)

    indices=list(range(num_examples))
    # 随机打乱样本序号
    random.shuffle(indices)

    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i:min(i+batch_size,num_examples)])
        #take()根据索引返回元素！
        #yeild生成器，在每一次的迭代过程中返回指定的值
        yield features.take(j), labels.take(j)

batch_size=10


# 打印一批次的数据
for X, y in data_iter(batch_size, features, labels):
    # print(X, y)
    break



# 【初始化模型参数】

w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
b = nd.zeros(shape=(1,))
print(w, b)
w.attach_grad()
b.attach_grad()


# 【定义计算模型，计算出的就是预测值】

def linereg(X, w, b):
    return nd.dot(X, w) + b

# 【定义损失函数-平方损失】

def squared_loss(y_pre, y):
    return (y_pre - y.reshape(y_pre.shape))**2/2


# 【定义优化算法】


def sgd(params, lr, batch_size):
    # 参数优化的核心步骤，每执行一次，参数就会朝着优化方向进化一次！
    for param in params:
        # param[:] 是深复制，梯度下降优化不仅仅针对的是权重，而且也有偏差！
        param[:] = param - lr * param.grad / batch_size


# 【开始模型的训练】


lr = 0.3  # 设定学习速率为0.3，迭代次数为3
num_epochs = 3


net = linereg  # 模型计算函数即为神经网络层

loss = squared_loss

# 开始迭代计算
for epoch in range(num_epochs):# 一共计算num_epochs次
    for X, y in data_iter(batch_size, features, labels):
        with autograd.record():
            l = loss(net(X, w, b), y)
        # 这个backward()函数的作用是求出损失函数loss在
        # 其参数位置处的导数，这样在sgd中的param.grad的值才能取到！
        # 若没有backward()，w和b的grad取值均为0，这样模型参数就无法进行优化
        # 始终为初始值！在训练之中没有吸取到任何的教训，训练总是失败的！

        l.backward()
        sgd([w, b], lr, batch_size)
    train_loss = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch+1, train_loss.mean().asnumpy()))

print(true_w, w)
print(true_b, b)



