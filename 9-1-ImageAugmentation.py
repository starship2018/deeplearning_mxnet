import sys
import mxnet as mx
from mxnet import autograd, gluon, image, init, nd
from mxnet.gluon import data as gdata, loss as gloss, utils as gutils, nn
from time import time
from matplotlib import pyplot as plt

img = image.imread('./image/luffy.jpg')
# plt.imshow(img.asnumpy())
# plt.show()

def show_images(imgs, num_rows, num_cols, scale =2):
    figsize = (num_cols*scale , num_rows*scale)
    # subplots返回两个参数，第一个是figure图对象，另一个是axes轴对象
    #对图像的处理应该对axes对象操作！

    # 若不设置这个figsize参数，完整体的图像大小和原始单张图像的大小一致
    # 设置了figsize后，原始的图像呈现不变，各个小图之间更为紧密
    _, axes = plt.subplots(num_rows, num_cols,figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i*num_cols+j].asnumpy())
            # 隐藏x , y轴
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    # 记住要show出这个图像
    plt.show()
    return axes

def apply(img, aug, num_rows =2,num_cols =4,scale =1.5):
    # 这个for的作用是复制增殖的过程！
    Y = [aug(img) for _ in range(num_rows*num_cols)]
    show_images(Y,num_rows,num_cols,scale)

# apply(img,gdata.vision.transforms.RandomFlipLeftRight())
shape_aug = gdata.vision.transforms.RandomResizedCrop((200,200),scale=(0.1,1),ratio=(0.5,2))
# apply(img, shape_aug)
# apply(img, gdata.vision.transforms.RandomBrightness(0.5))
# apply(img,gdata.vision.transforms.RandomHue(0.5))

# show_images(gdata.vision.CIFAR10(train=True)[0:32][0], 4, 8, scale=2)





train_augs = gdata.vision.transforms.Compose([gdata.vision.transforms.RandomFlipLeftRight(), gdata.vision.transforms.ToTensor()])
test_augs = gdata.vision.transforms.Compose([gdata.vision.transforms.ToTensor()])

#【生成数据】
def load_cifar10(is_train, augs, batch_size):
    return gdata.DataLoader(gdata.vision.CIFAR10(train=is_train).transform_first(augs),batch_size=batch_size,shuffle=is_train)

# 准确率的计算
def batch_accuracy(y_hat, y):
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()


def evaluate_accuracy(data_iter, net):
    acc = nd.array([0])
    # n = 0
    # for batch in data_iter:
    #     features, labels = batch

        # 如果labels的子元素类型 和features的不一样，则改为一致的类型！
        # type()是指的labels和features整个的数据类型，不是单个元素的类型
    # if labels.dtype != features.dtype:
    #     labels = labels.astype(features.dtype)
    for X, y in data_iter:
        X, y = X.as_in_context(mx.gpu()), y.as_in_context(mx.gpu())
        y = y.astype('float32')
        # acc += (net(X).argmax(axis=1) == y).sum()
        acc += batch_accuracy(net(X), y)
        # n += y.size          # 改为len(y)如何？
    acc.wait_to_read()
    return acc.asscalar() / len(data_iter)

#【开始训练模型】
def train(train_iter, test_iter, net, loss, trainer, num_epochs):
    string = ''
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, m =0.0,0.0,0.0,0.0
        start = time()
        for X, y in train_iter:
            # if y.dtype != X.dtype:
            #     y = y.astype(X.dtype)
            X = X.as_in_context(mx.gpu())
            y = y.as_in_context(mx.gpu())
            batch_size = 128
            ls = []
            with autograd.record():
                # for X in Xs:
                #     print(X.shape)
                # y_hats = [net(X) for X in Xs]
                y_hat = net(X)
                # ls = [loss(y_hat, y) for y_hat, y in zip(y_hats, ys)]
                l = loss(y_hat, y)
            l.backward()
            train_acc_sum += batch_accuracy(y_hat, y)
            train_l_sum += l.mean().asscalar()
            trainer.step(batch_size)
        test_acc = evaluate_accuracy(test_iter,net)
        print('epoch %d  loss %.4f  train_acc %.4f  test_acc %.3f  time %.1f' %(epoch+1,
                                                                             train_l_sum/len(train_iter),
                                                                             train_acc_sum/len(train_iter),
                                                                             test_acc,time() -start))
        string += str('epoch %d  loss %.4f  train_acc %.4f  test_acc %.3f  time %.1f \t' %(epoch+1,
                                                                             train_l_sum/len(train_iter),
                                                                             train_acc_sum/len(train_iter),
                                                                             test_acc,time() -start))
    return string


#【定义模型】

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

def resnet18(num_classes):
    def resnet_block(num_channels, num_residuals, first_block =False):
        blk = nn.Sequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                blk.add(Residual(num_channels))
        return blk

    net = nn.Sequential()
    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
            nn.BatchNorm(), nn.Activation('relu'))
    net.add(resnet_block(64, 2, first_block=True),
            resnet_block(128, 2),
            resnet_block(256, 2),
            resnet_block(512, 2),
            nn.GlobalAvgPool2D(),
            nn.Dense(num_classes))
    return net


def train_with_data_aug(train_augs, test_augs, lr=0.001):
    # 【读取数据】
    batch_size = 128  # 256
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    #【定义计算模型】
    net = resnet18(10)
    #【初始化模型参数】
    net.initialize(ctx=mx.gpu(), init=init.Xavier())
    #【定义优化算法】
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate':lr})
    #【定义损失函数】
    loss = gloss.SoftmaxCrossEntropyLoss()
    #【开始训练】
    return train(train_iter, test_iter, net, loss, trainer, num_epochs=5)

with open('9-1-result.txt', 'w') as f:
    f.write(train_with_data_aug(train_augs,test_augs))

'''
    在使用cifar10数据集的时候，关于精确度的计算和训练过程，和fashion_mnist的过程是一模一样的
    1.gluon上面的过程会出错误，暂时无法解决！所以以后依照原始的计算过程进行计算！
'''

