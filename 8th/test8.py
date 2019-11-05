import cv2
from matplotlib import pyplot as plt
import numpy as np

# 灰度级数
GRAYLEVEL = 256

# 统计图像灰度直方图
def calHist(I):
    hist = np.zeros(GRAYLEVEL, np.int16)
    for line in I:
        for pixel in line:
            hist[pixel] += 1
    return hist

# 直方图均衡化
def histequal4e(I):
    g = np.zeros_like(I)
    height, width = I.shape

    # 直方图统计
    hist = calHist(I)

    # 直方图归一化
    prob = np.zeros(GRAYLEVEL, np.float)
    for i in range(GRAYLEVEL):
        prob[i] = hist[i] / (width * height)

    # 累计频率
    prob = np.cumsum(prob)

    # 将灰度值映射到uint8并赋值给g
    for i in range(height):
        for j in range(width):
            g[i, j] = prob[I[i, j]] * (GRAYLEVEL - 1)

    return g


# 加入椒盐噪声
# snr：信噪比
def add_noise(f, snr):
    height, width = f.shape
    g = f.copy()
    
    noiseNum = int(height * width * (1 - snr))

    for i in range(noiseNum):
        randx = np.random.randint(0, height - 1)
        randy = np.random.randint(0, width - 1)
        if np.random.random() <= 0.5:
            g[randx, randy] = 0
        else:
            g[randx, randy] = 255
    
    return g


# 图像边缘填充
# size：tuple
# method：零填充zero 最近复制填充replicate
def padding(f, size, method='zero'):
    fHeight, fWidth = f.shape
    wHeight, wWidth = size

    # 每条边外需要填充的部分
    paddingHeight = wHeight // 2
    paddingWidth = wWidth // 2

    # 创建一个全0的数组并填充上f，这样便实现了0填充方法
    g = np.zeros([fHeight + paddingHeight * 2,
                  fWidth + paddingWidth * 2], np.uint8)
    g[paddingHeight:fHeight + paddingHeight,
        paddingWidth:fWidth + paddingWidth] = f

    if method == 'replicate':
        g[0:paddingHeight] = g[paddingHeight]
        g[:, 0:paddingWidth] = np.array([g[:, paddingWidth]]).T
        g[fHeight + paddingHeight:] = g[fHeight + paddingHeight - 1]
        g[:, fWidth +
            paddingWidth:] = np.array([g[:, fWidth + paddingWidth - 1]]).T
    elif method != 'zero':
        print('Invalid padding method!')
        exit()

    return g


# 计算单个像素平滑后的结果
# 参数neighbors：像素的(5, 5)邻域
def smooth_one_pixel(neighbors):
    # 正方形作为初值
    square = neighbors[1:4, 1:4]
    minMean = np.mean(square)
    minVar = np.var(square)

    # 五边形掩模
    # 2*3矩形加上中心点
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
    # 正方形去掉2个点
    for y in [0, 1, 2], [2, 3, 4]:
        for x in [0, 1, 2], [2, 3, 4]:
            hexagon = []
            for i in y:
                for j in x:
                    if (i, j) != (0, 2) and (i, j) != (2, 0) and (i, j) != (2, 4) and (i, j) != (4, 2):
                        hexagon.append(neighbors[i][j])
            if np.var(hexagon) < minVar:
                minMean = np.mean(hexagon)
                minVar = np.var(hexagon)

    return minMean


# 有选择保边缘平滑法 
def smoother(f):
    height, width = f.shape
    g = padding(f, (5, 5), 'replicate')
    for i in range(2, 2 + height):
        for j in range(2, 2 + width):
            g[i][j] = smooth_one_pixel(g[i - 2:i + 3, j - 2:j + 3])
    return g[2:2 + height, 2:2 + width]


# 矩阵翻转180°，以进行卷积操作
def flip180(arr):
    new_arr = arr.ravel()
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    return new_arr


# 进行卷积
def twodConv(f, w, method='zero'):
    fHeight, fWidth = f.shape
    wHeight, wWidth = w.shape

    # 每条边外需要填充的部分
    paddingHeight = wHeight // 2
    paddingWidth = wWidth // 2

    f_padding = padding(f, w.shape, 'replicate')

    # 进行卷积
    g = np.zeros(f.shape)
    w = flip180(w)
    for i in range(paddingHeight, fHeight + paddingHeight):
        for j in range(paddingWidth, fWidth + paddingWidth):
            convZone = f_padding[i - paddingHeight:i +
                                 paddingHeight + 1, j - paddingWidth:j + paddingWidth + 1]
            g[i - paddingHeight][j - paddingWidth] = np.sum(convZone * w)

    return g


# 数据映射到[0, 255]
def map2Uint8(f):
    f -= np.min(f)  # 归零
    f /= np.max(f)
    f *= (GRAYLEVEL - 1)
    return f


if __name__ == '__main__':
    I = cv2.imread('rose512.tif', cv2.IMREAD_GRAYSCALE)

    # 第一题 直方图均衡化
    g = histequal4e(I)
    plt.subplot(221), plt.imshow(I, cmap='gray'), plt.title(
        'Original'), plt.axis('off')
    plt.subplot(222), plt.bar([i for i in range(GRAYLEVEL)], calHist(
        I), width=1), plt.title('Histogram')
    plt.subplot(223), plt.imshow(g, cmap='gray'), plt.title(
        'EqualHist'), plt.axis('off')
    plt.subplot(224), plt.bar([i for i in range(GRAYLEVEL)], calHist(
        g), width=1), plt.title('Histogram')
    # plt.imshow(cv2.equalizeHist(I), cmap='gray'), plt.axis('off')
    plt.show()

    # 第二题 有选择保边缘平滑法
    noise = add_noise(I, 0.6)
    g = smoother(noise)
    plt.subplot(131), plt.imshow(I, cmap='gray'), plt.title(
        'Original'), plt.axis('off')
    plt.subplot(132), plt.imshow(noise, cmap='gray'), plt.title(
        'NoiseAdded'), plt.axis('off')    
    plt.subplot(133), plt.imshow(g, cmap='gray'), plt.title(
        'Smoothing'), plt.axis('off')
    plt.show()

    # 第三题 拉普拉斯增强
    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.int16)
    g = twodConv(I, laplacian, 'replicate')
    g = map2Uint8(g)
    plt.subplot(121), plt.imshow(I, cmap='gray'), plt.title(
        'Original'), plt.axis('off')
    plt.subplot(122), plt.imshow(g, cmap='gray'), plt.title(
        'Laplacian'), plt.axis('off')
    # plt.imshow(cv2.Laplacian(I, cv2.CV_16S, 3), cmap='gray'), plt.axis('off')
    plt.show()