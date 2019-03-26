import sys
from mxnet import contrib, gluon, image, nd
import numpy as np
from matplotlib import pyplot as plt


# matplotlib在显示图像要以numpy格式的进行显示，直接读取得到的图像无法显示出来！
img = image.imread('./image/targetdetection.jpg').asnumpy()
h, w = img.shape[0:2]

# print(h,w)

X = nd.random.uniform(shape=(1,3,h,w))   # 返回的数组是在0和1之间均匀分布！
Y = contrib.nd.MultiBoxPrior(X,sizes=[0.75,0.5,0.25],ratios=[1,2,0.5])
# print(Y.shape)

boxes = Y.reshape((h, w, 5, 4))
# print(boxes[250,250,0,:])

def get_rectangle(bbox,color):
    return plt.Rectangle(xy=(bbox[0],bbox[1]),width=bbox[2]-bbox[0],height=bbox[3]-bbox[1],fill=False,edgecolor=color,linewidth=2)


def show_boxes(axes, bboxes, labels=None, colors = None):
    def _make_list(obj,default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj,(list,tuple)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ['b','g','r','m','c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = get_rectangle(bbox.asnumpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) >1:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va = 'center', ha = 'center', fontsize =9,
                      color=text_color,bbox=dict(facecolor = color,lw=0))
bbox_scale = nd.array((w,h,w,h))
fig = plt.imshow(img)
# show_boxes(fig.axes, boxes[250,250,:,:] * bbox_scale,
#            ['s=0.75,r=1','s=0.5,r=1','s=0.25,r=1','s=0.75,r=2','s=0.75,r=0.5'])
# plt.show()

ground_truth = nd.array([[0,0.25,0.1,0.45,0.42],[1,0.55,0.1,0.75,0.4]])
anchor = nd.array([[0,0.1,0.2,0.3],[0.2,0.1,0.5,0.9],[0.6,0.1,0.8,0.5],[0.55,0.3,0.7,0.5],[0.65,0.15,0.8,0.9]])
show_boxes(fig.axes, ground_truth[:,1:] * bbox_scale,labels=['dog','cat'],colors='k')
show_boxes(fig.axes, anchor*bbox_scale,['0','1','2','3','4'])
# plt.show()

labels = contrib.nd.MultiBoxTarget(anchor.expand_dims(axis=0), ground_truth.expand_dims(axis=0),nd.zeros((1,3,5)))