import sys
from mxnet import image
from matplotlib import pyplot as plt

# 使用matplotlib显示图片
img = image.imread("./image/targetdetection.jpg").asnumpy()
figure = plt.imshow(img)  #注意这里获取imshow()函数的返回值，得到一个画板，方便二次作图！


zi_bbox ,shan_bbox = [160,40,280,170], [350,32,460,180]


# 在matplotlib中画出矩形需要以下参数：原点（左上角）+宽+高
def get_rectangle_from_dots(bbox, color):
    return plt.Rectangle(xy=(bbox[0],bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
                         fill=False, edgecolor=color,
                         linewidth=2)

figure.axes.add_patch(get_rectangle_from_dots(zi_bbox, color='red'))
figure.axes.add_patch(get_rectangle_from_dots(shan_bbox, color='blue'))
# plt.plot()
# plt.plot()
plt.show()