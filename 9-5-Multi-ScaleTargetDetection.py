from mxnet import contrib, image, nd
from matplotlib import pyplot as plt


img = image.imread("./image/targetdetection1.png")
h, w = img.shape[0:2]
# print(img.shape)   # 高为312 宽为500 通道数为3

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

def display_anchors(fmap_w, fmap_h,s):
    fmap = nd.zeros((1,10,fmap_w, fmap_h))
    anchors = contrib.nd.MultiBoxPrior(fmap,sizes =s, ratios=[1,2,0.5])
    bbox_scale = nd.array((w,h,w,h))
    show_boxes(plt.imshow(img.asnumpy()).axes, anchors[0] * bbox_scale)
    plt.show()

display_anchors(fmap_h=2, fmap_w=3, s=[0.5])
