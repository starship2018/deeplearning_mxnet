from mxnet import autograd, gluon, image, init, nd
from mxnet.gluon import model_zoo, nn
from matplotlib import pyplot as plt
import mxnet as mx
import time

# 分别显示风格图像和内容图像
plt.figure()

style_image = image.imread('./image/fangao.jpg')


content_image = image.imread('./image/scene.jpg')

plt.imshow(style_image.asnumpy())
# plt.show()
plt.imshow(content_image.asnumpy())
# plt.show()

#定义预处理函数和后处理函数
rgb_mean = nd.array([0.485,0.456,0.406])
rgb_std = nd.array([0.229,0.224,0.225])

def preprocess(img, image_shape):
    img =image.imresize(img, *image_shape)
    # 标准化数据在0-1之间
    img = (img.astype('float32')/255 -rgb_mean)/rgb_std
    # 将图像的通道维也就是第三维提到了最前面，同时还在最前面加上了一维
    return img.transpose((2,0,1)).expand_dims(axis=0)

def postprocess(img):
    img = img[0].as_in_context(rgb_std.context)
    return (img.transpose((1,2,0)) * rgb_std + rgb_mean).clip(0,1)


pretrained_net = model_zoo.vision.vgg19(pretrained=True)


# 定义序号为0 5 10 19 28 的层数为风格层 26号为内容层
style_layers, content_layers = [0,5,10,19,28],[25]
net = nn.Sequential()
for i in range(max(content_layers + style_layers) + 1):
    net.add(pretrained_net.features[i])


# 取出样式层和内容层的输出
def extract_features(x, content_layers, style_layers):
    # 普通的net(X)得到的是最终的输出结果，但是我们要获取到输出层之前的几层信息！
    contents = []
    styles = []
    for i in range(len(net)):
        x = net[i](x)
        if i in style_layers:
            # 拿到风格层的结果
            styles.append(x)
        if i in content_layers:
            # 拿到内容层的结果
            contents.append(x)
    return contents, styles

def get_contents(image_shape):
    content_x = preprocess(content_image, image_shape).copyto(mx.gpu())
    content_y, _ = extract_features(content_x, content_layers, style_layers)
    return content_x, content_y

def get_styles(image_shape):
    style_x = preprocess(style_image, image_shape).copyto(mx.gpu())
    _, style_y = extract_features(style_x, content_layers, style_layers)
    return style_x, style_y

#【定义内容损失函数】
def content_loss(y_hat, y):
    return (y_hat - y).square().mean()

# 计算协方差矩阵
def gram(x):
    c, n = x.shape[1], x.size // x.shape[1]
    y = x.reshape((c, n))
    return nd.dot(y,y.T) / (c*n)

#【定义风格损失函数】
def style_loss(y_hat, gram_y):
    return (gram(y_hat) - gram_y).square().mean()

#总变差降噪
def tv_loss(y_hat):
    return 0.5 * ((y_hat[:,:,1:,:] - y_hat[:,:,:-1,:]).abs().mean() + (y_hat[:,:,:,1:] - y_hat[:,:,:,:-1]).abs().mean())

style_channels = [net[l].weight.shape[0] for l in style_layers]
style_weights = [1e3] * len(style_channels)
content_weights, tv_weights = [1], 10


class TransferImage(nn.Block):
    def __init__(self, img_shape, **kwargs):
        super(TransferImage, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=img_shape)

    def forward(self, *args):
        return self.weight.data()

def train(x, content_y, style_y, lr, max_epochs, lr_decey_epoch):
    net = TransferImage(x.shape)
    net.initialize(init.Constant(x),ctx=mx.gpu(), force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate':lr})

    x = net()

    style_y_gram = [gram(y) for y in style_y]
    for i in range(max_epochs):
        tic = time.time()
        with autograd.record():
            # 对x抽取样式和内容特征
            content_y_hat, style_y_hat = extract_features(x, content_layers, style_layers)
            # 分别计算内容，样式和噪音损失
            content_L = [w*content_loss(y_hat, y) for w, y_hat, y in zip(content_weights, content_y_hat, content_y)]
            style_L =[w* style_loss(y_hat, y) for w, y_hat, y in zip(style_weights, style_y_hat, style_y_gram)]
            tv_L = tv_weights*tv_loss(x)
            # 对所有的损失求和
            l = nd.add_n(*style_L) + nd.add_n(*content_L) + tv_L
        l.backward()
        trainer.step(1)

        nd.waitall()

        if i %50 ==0:
            print('batch %.3d:content %.2f, style %.2f,TV %.2f, %.1f sec per batch' %(i, nd.add_n(*content_L).asscalar(),
                                                                                      nd.add_n(*style_L).asscalar(),
                                                                                      tv_L.asscalar(),
                                                                                      time.time() - tic))
        if i % lr_decey_epoch ==0 and i !=0:
            trainer.set_learning_rate(trainer.learning_rate*0.1)
            print('change lr to %.1e' % (trainer.learning_rate))
    return net()

# 设置输出图像的尺寸
image_shape = (600,400)
net.collect_params().reset_ctx(mx.gpu())
content_x, content_y = get_contents(image_shape)
style_x, style_y = get_styles(image_shape)

x = content_x
y = train(x, content_y, style_y, 0.01,500,200)
plt.imsave('./image/neural_style_1.png', postprocess(y).asnumpy())


