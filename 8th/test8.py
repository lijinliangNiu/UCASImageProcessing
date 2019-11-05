import cv2
from matplotlib import pyplot as plt
import numpy as np

# 灰度级数
GRAYLEVEL = 256


def calHist(I):
    hist = np.zeros(GRAYLEVEL, np.int16)
    for line in I:
        for pixel in line:
            hist[pixel] += 1
    return hist


def histequal4e(I):
    g = np.zeros_like(I)
    height, width = I.shape

    # 直方图统计
    hist = calHist(I)

    # 直方图归一化
    prob = np.zeros(GRAYLEVEL, np.float)
    for i in range(GRAYLEVEL):
        prob[i] = hist[i] / (width * height)

    # 频率累计
    prob = np.cumsum(prob)

    for i in range(height):
        for j in range(width):
            #映射到[0, 255]
            g[i, j] = prob[I[i, j]] * (GRAYLEVEL - 1)

    return g

# 对图像进行填充 size为一个tuple


def padding(f, size, padding='zero'):
    fHeight, fWidth = f.shape
    wHeight, wWidth = size

    # 每条边外需要填充的部分
    paddingHeight = int(wHeight / 2)
    paddingWidth = int(wWidth / 2)

    # 创建一个全0的数组并填充上f，这样便实现了0填充方法
    g = np.zeros([fHeight + paddingHeight * 2,
                  fWidth + paddingWidth * 2], np.uint8)
    g[paddingHeight:fHeight + paddingHeight,
        paddingWidth:fWidth + paddingWidth] = f

    if padding == 'replicate':
        g[0:paddingHeight] = g[paddingHeight]
        g[:, 0:paddingWidth] = np.array([g[:, paddingWidth]]).T
        g[fHeight + paddingHeight:] = g[fHeight + paddingHeight - 1]
        g[:, fWidth +
            paddingWidth:] = np.array([g[:, fWidth + paddingWidth - 1]]).T
    elif padding != 'zero':
        print('Invalid padding method!')
        exit()

    return g


# 传像素的(5,5)邻域
def smooth_one_pixel(neighbors):
    # 正方形作为初值
    square = neighbors[1:4, 1:4]
    minMean = np.mean(square)
    minVar = np.var(square)

    # 五边形掩模
    for y, x in [[0, 1], [1, 2, 3]], [[1, 2, 3], [0, 1]], [[3, 4], [1, 2, 3]], [[1, 2, 3], [3, 4]]:
        pentagon = []
        pentagon.append(neighbors[2][2])
        for i in y:
            for j in x:
                pentagon.append(neighbors[i][j])
        if np.var(pentagon) < minVar:
            minMean = np.mean(pentagon)
            minVar = np.var(pentagon)

    # 六边形掩模
    for y, x in [[0, 1, 2], [0, 1, 2]], [[0, 1, 2], [2, 3, 4]], [[2, 3, 4], [0, 1, 2]], [[2, 3, 4], [2, 3, 4]]:
        hexagon = []
        for i in y:
            for j in x:
                if (i, j) != (0, 2) and (i, j) != (2, 0) and (i, j) != (2, 4) and (i, j) != (4, 2):
                    hexagon.append(neighbors[i][j])
        if np.var(hexagon) < minVar:
            minMean = np.mean(hexagon)
            minVar = np.var(hexagon)

    return minMean


def smoother(f):
    height, width = f.shape
    g = padding(f, (5, 5), 'replicate')
    for i in range(2, 2 + height):
        for j in range(2, 2 + width):     
            g[i][j] = smooth_one_pixel(g[i - 2:i + 3, j - 2:j + 3])
    return g[2:2 + height, 2:2 + width]


if __name__ == '__main__':
    I = cv2.imread("einstein.png", cv2.IMREAD_GRAYSCALE)

    # 第一题 直方图均衡化
    # g = histequal4e(I)
    # plt.subplot(221), plt.imshow(I, cmap='gray'), plt.title('original'), plt.axis('off')
    # plt.subplot(222), plt.bar([i for i in range(GRAYLEVEL)], calHist(I), width=1), plt.title('Histogram')
    # plt.subplot(223), plt.imshow(g, cmap='gray'), plt.title('eqalHist'), plt.axis('off')
    # plt.subplot(224), plt.bar([i for i in range(GRAYLEVEL)], calHist(g), width=1), plt.title('Histogram')
    # plt.show()

    # 第二题 有选择保边缘平滑法
    # g = smoother(I)
    # plt.subplot(211), plt.imshow(I, cmap='gray')
    # plt.subplot(212), plt.imshow(g, cmap='gray')
    # plt.show()

    #第三题 拉普拉斯增强
    