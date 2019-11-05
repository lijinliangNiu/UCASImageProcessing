import numpy as np
import cv2

##################
#更改main函数中的f和w即可运行
##################

def twodConv(f, w, padding='zero'):
    fHeight, fWidth = f.shape
    wHeight, wWidth = w.shape

    # 每条边外需要填充的部分
    paddingHeight = int(wHeight / 2)
    paddingWidth = int(wWidth / 2)

    # 创建一个全0的数组并填充上f，这样便实现了0填充方法
    paddingF = np.zeros([fHeight + paddingHeight * 2, fWidth + paddingWidth * 2], np.float32)
    paddingF[paddingHeight:fHeight + paddingHeight, paddingWidth:fWidth + paddingWidth] = f

    if padding == 'replicate':
        paddingF[0:paddingHeight] = paddingF[paddingHeight]
        paddingF[:, 0:paddingWidth] = np.array([paddingF[:, paddingWidth]]).T
        paddingF[fHeight + paddingHeight:] = paddingF[fHeight + paddingHeight - 1]
        paddingF[:, fWidth + paddingWidth:] = np.array([paddingF[:, fWidth + paddingWidth - 1]]).T
    elif padding != 'zero':
        print('Invalid padding method!')
        exit()

    # 进行卷积
    g = np.zeros(f.shape, np.uint8)
    for i in range(paddingHeight, fHeight + paddingHeight):
        for j in range(paddingWidth, fWidth + paddingWidth):
            convZone = paddingF[i - paddingHeight:i + paddingHeight + 1, j - paddingWidth:j + paddingWidth + 1]
            g[i - paddingHeight][j - paddingWidth] = np.sum(convZone * w)

    return g


def gaussKernel(sig, m=0):
    kernelSize = int(np.ceil(3 * sig) * 2 + 1)
    if m == 0:
        m = kernelSize
    if m < kernelSize:
        print('Provided m is small!')
        exit()

    kernel = np.zeros((m, m), np.float32)
    center = int(m / 2)
    sum = 0
    for i in range(m):
        x = (i - center) * (i - center)
        for j in range(m):
            y = (j - center) * (j - center)
            kernel[i][j] = np.exp(-(x + y) / (2 * sig * sig))
            kernel[i][j] /= (2 * np.pi * sig * sig)
            sum += kernel[i][j]
    kernel /= sum

    return kernel


if __name__ == '__main__':
    f = cv2.imread('lena512color.tiff', cv2.IMREAD_GRAYSCALE)
    w = gaussKernel(5)
    g1 = twodConv(f, w, 'replicate')

    # 与opencv中的高斯滤波进行比较
    #g2 = cv2.GaussianBlur(f, (7, 7), 1)
    #g = np.abs(g2.astype(np.int32) - g1.astype(np.int32))
    #print(g)
    #cv2.imshow('w', g.astype(np.uint8))

    cv2.imshow('window', g1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()