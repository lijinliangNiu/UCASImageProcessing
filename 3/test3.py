import numpy as np
from matplotlib import pyplot as plt

import cv2


def dft2D(f):
    F = np.zeros_like(f, dtype=complex)
    for i in np.arange(f.shape[0]):
        F[i, :] = np.fft.fft(f[i, :])
    for i in np.arange(f.shape[1]):
        F[:, i] = np.fft.fft(F[:, i])
    return F


def idft2D(F):
    f = dft2D(F.conjugate())
    f /= (F.shape[0] * F.shape[1])
    f = f.conjugate()
    return f


# 把数据映射到[0, 255]
def map2Uint8(f):
    f -= np.min(f)  # 归零
    f /= np.max(f)
    f *= 255
    return f


# 中心化频谱
def centered_spectrum(F):
    height, width = F.shape
    centered = np.zeros_like(F)
    for i in range(height // 2):
        for j in range(width // 2):
            centered[i, j] = F[i + height // 2, j + width // 2]  # 左上
            centered[i + height // 2, j + width // 2] = F[i, j]  # 左下

            centered[i, j + width // 2] = F[i + height // 2, j]  # 右下
            centered[i + height // 2, j] = F[i, j + width // 2]  # 右上
    return centered


if __name__ == "__main__":
    f = cv2.imread('rose512.tif', cv2.IMREAD_GRAYSCALE)
    f = f / 255.

    F = dft2D(f)
    g = idft2D(F)

    npF = np.fft.fft2(f)
    npf = np.fft.ifft2(npF)
    print('正变换与numpy中函数的差：')
    print(F - npF)
    print('逆变换与numpy中函数的差：')
    print(f - npf)

    plt.imshow(f * 255 - map2Uint8(np.abs(g)), cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.show()

    f = np.zeros((512, 512), dtype=np.uint8)
    f[226:285, 251:260] = 1
    F = dft2D(f)  # 傅里叶变换
    fShift = centered_spectrum(F)  # 中心化
    result = np.log(1 + np.abs(fShift))  # 对数化

    #  plt.imshow()可以将数值序列直接映射为灰度图
    plt.subplot(221), plt.imshow(
        f * 255, cmap='gray'), plt.title('original'), plt.axis('off')
    plt.subplot(222), plt.imshow(np.abs(F), cmap='gray'), plt.title(
        'spectrum'), plt.axis('off')
    plt.subplot(223), plt.imshow(np.abs(fShift), cmap='gray'), plt.title(
        'centered spectrum'), plt.axis('off')
    plt.subplot(224), plt.imshow(result, cmap='gray'), plt.title(
        'log transform'), plt.axis('off')
    plt.show()
