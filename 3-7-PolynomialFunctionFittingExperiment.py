from mxnet import autograd,gluon,nd
from mxnet.gluon import data as gdata ,loss as gloss ,nn
from matplotlib import pyplot as plt


#  y=1.2x-3.4x^2+5.6x^3+5+c
# 根据这个标准函数式来生成许多（x,y）点，然后添加上一定的噪声，
# 最后训练模型看是否能够仅仅通过大量的添加了噪声的点还原得到原始的标准函数式

n_train,n_test,true_w,true_b = 100, 100, [1.2, -3.4, 5.6],5

# 生成X
# 单个特征x（生成X轴）
features = nd.random.normal(shape=(n_test + n_train, 1))

# 多个特征-多项式
poly_features = nd.concat(features, nd.power(features, 2), nd.power(features, 3))

# 生成标准点
labels = (true_w[0] * poly_features[:,0] + true_w[1] * poly_features[:,1] + true_w[2] *
          poly_features[:,2] +true_b)

# 加上噪声生成测试点
labels += nd.random.normal(scale=0.1, shape=labels.shape)

# print(features[:5], poly_features[:2], labels[:8],poly_features.shape,labels.shape)

# 定义作图函数(一图同轴多线)
def semilogy(x_vals, y_vals, x_label, y_label,x2_vals = None,y2_vals=None,legend=None):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals,':')
        #legend 列表表示对画出的两幅图进行分类展示
        plt.legend(legend)
    plt.show()


num_epochs, loss = 500, gloss.L2Loss()

def fit_and_plot(train_features, test_features, train_labels, test_labels):
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize()
    batch_size = min(10, train_labels.shape[0])
    train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features, train_labels),batch_size,shuffle=True)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.01})
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            # 神经网络自生成的参数不用attach_grad()
            l.backward()
            trainer.step(batch_size)
            # 每一次训练完成之后，使用记录下来的Wb来计算出此时的loss
        train_ls.append(loss(net(train_features),train_labels).mean().asscalar())
        test_ls.append(loss(net(test_features), test_labels).mean().asscalar())
    print('final epoch : train_loss',train_ls[-1],'test_loss',test_ls[-1])

    # 画出train_Loss和test_loss随着epoch的增加而变化的情况
    semilogy(range(1,num_epochs+1),train_ls,'epoch','loss', range(1,num_epochs+1),
             test_ls,['train', 'test'])
    print('weight:',net[0].weight.data().asnumpy(),'\nbias:',net[0].bias.data().asnumpy())


fit_and_plot(poly_features[:n_train,:],poly_features[n_train:,:],labels[:n_train],labels[n_train:])





