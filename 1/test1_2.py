import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def rgb1gray(f, method='NTSC'):
    r, g, b = [f[:, :, i] for i in range(3)]
    gray = []
    if method == 'average':
        # 对于uint8，相加 / 3会溢出
        gray = r / 3 + g / 3 + b / 3
    elif method == 'NTSC':
        gray = r * 0.2989 + g * 0.5870 + b * 0.1140
    else:
        print('invalid method')
    return gray


def drawOneImg(res, imgPath, num):
    f = mpimg.imread(imgPath)

    originalImg = res.add_subplot(2, 3, 3 * num + 1)
    averageImg = res.add_subplot(2, 3, 3 * num + 2)
    NTSCImg = res.add_subplot(2, 3, 3 * num + 3)

    originalImg.set_title('Original Image')
    originalImg.axis('off')
    originalImg.imshow(f)

    g = rgb1gray(f, 'average')
    averageImg.set_title('Average Grayscale')
    averageImg.axis('off')
    averageImg.imshow(g, cmap='gray')

    g = rgb1gray(f)
    NTSCImg.set_title('NTSC Grayscale')
    NTSCImg.axis('off')
    NTSCImg.imshow(g, cmap='gray')


if __name__ == '__main__':
    res = plt.figure(figsize=(16, 9))

    drawOneImg(res, 'mandril_color.tif', 0)
    drawOneImg(res, 'lena512color.tiff', 1)

    # plt.savefig('test1_2_histogram.png', format='png')
    plt.show()