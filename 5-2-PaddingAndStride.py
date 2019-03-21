from mxnet import nd
from mxnet.gluon import nn

def comp_conv2d(conv2d, x):
    # 这里给x的维度进行了添加，添加了一个批量数和通道数！
    x = x.reshape((1,1) + x.shape)
    y = conv2d(x)
    # 这里将y除掉了所添加的新的二维
    return y.reshape(y.shape[2:])

# 注意！！！！！声明卷积层时一定要指定卷积核的通道数！！！！！


# 定义模型
conv2d = nn.Conv2D(1, kernel_size=(3,3), padding=1)

# 初始化模型
conv2d.initialize()


x = nd.random.uniform(shape=(8,8))

print(comp_conv2d(conv2d, x))

