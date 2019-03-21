from mxnet import gluon, init, nd, autograd
from mxnet.gluon import nn, loss as gloss, data as gdata
import mxnet as mx
import time



# 【生成数据集】

train_data = gdata.vision.FashionMNIST(train = True)
test_data = gdata.vision.FashionMNIST(train = False)

# 【读取数据集】

batch_size = 128
transformer = [gdata.vision.transforms.Resize(96), gdata.vision.transforms.ToTensor()]
transformer = gdata.vision.transforms.Compose(transformer)

train_iter = gdata.DataLoader(train_data.transform_first(transformer), batch_size=batch_size, shuffle= True)
test_iter = gdata.DataLoader(test_data.transform_first(transformer), batch_size=batch_size, shuffle= False)





# 这时一个构建卷积块的快速函数，输入卷积层的层数和输出通道数，就会按照默认的方式
# 构造一个卷积块，以将输入全部减半的最大池化层结尾
def vgg_block(num_convs, num_channels):
    blk = nn.Sequential()
    for _ in range(num_convs):
        # blk中的卷积层不改变输入的大小，仅仅是对通道数进行扩充！
        blk.add(nn.Conv2D(num_channels, kernel_size=3,padding=1,activation='relu'))
        # 每一个blk块的最后通过一个最大池化层将输入全部减半，但维持原通道数（最大池化层不能改变通道数！）
    blk.add(nn.MaxPool2D(pool_size=2,strides=2))
    return blk




# 在比较初级的神经块中，我们只需要定义一个函数来进行神经层的搭建即可！
# 但是若出现了更加复杂的结构，则需要我们定义神经块类来自定出更加复合神经层之间的计算

def vgg(conv_arch):
    net = nn.Sequential()
    # 卷积层部分
    for (num_convs, num_channels) in conv_arch:
        net.add(vgg_block(num_convs, num_channels))

    # 全连接层部分

    net.add(nn.Dense(4096,activation='relu'),
            nn.Dropout(0.5),
            nn.Dense(4096,activation='relu'),
            nn.Dropout(0.5),
            nn.Dense(10))
    return net

conv_arch = ((1,64),(1,128),(2,256),(2,512),(2,512))
# net = vgg(conv_arch)

# net.initialize()

# X = nd.random.uniform(shape=(1,1,224,224))

# for blk in net:
#     X = blk(X)
#     print(blk.name, 'output shape:\t', X.shape)

ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]




# 【定义模型】
net = vgg(small_conv_arch)

# 【初始化模型参数】

net.initialize(init= init.Xavier(), ctx=mx.gpu())

# 【定义损失函数】

loss = gloss.SoftmaxCrossEntropyLoss()

#【定义优化算法】
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.05})

#【开始训练】

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
            train_acc +=batch_accuracy(y_hat, y)
        test_acc = all_accuracy(test_iter, net)
        print('epoch %d train_loss %.3f train_acc %.3f test_acc %.3f time %.1f' % (epoch+1,
                                                                               train_l_sum/len(train_iter),
                                                                               train_acc/len(train_iter),
                                                                               test_acc,
                                                                               time.time() - start))
        string += str('epoch %d train_loss %.3f train_acc %.3f test_acc %.3f time %.1f \r' % (epoch+1,
                                                                               train_l_sum/len(train_iter),
                                                                               train_acc/len(train_iter),
                                                                               test_acc,
                                                                               time.time() - start))
    return string




with open('5-7-result.txt','w') as f:
    f.write(train())

'''
    1.当图像的大小为224时，显存爆出导致实验失败！
    2.修改了实验的结果输出方法！既在控制台进行输出，又将结果写入到txt文件中去！
    
'''

