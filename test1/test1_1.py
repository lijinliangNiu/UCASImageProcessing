import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def scanLine4e(f, I, loc):
    # 取中心除以2后会变成小数
    I = int(I)
    if loc == 'row':
        return f[I]
    if loc == 'column':
        return f[:, I]
    else:
        print('invalid parameter')


# 返回 原图 中心行向量 中心列向量
def extractRowOrColomn(imgPath):
    img = mpimg.imread(imgPath)
    height, width = img.shape

    # 尺寸为偶数时，取的是后一个向量
    row = scanLine4e(img, height / 2, 'row')
    column = scanLine4e(img, width / 2, 'column')
    return img, row, column

# 最终图 图像路径 图像位于最终结果的行数
def drawOneImg(res, imgPath, num):
    img, row, column = extractRowOrColomn(imgPath)

    # 子图，分别绘制原图、中心行折线、中心列折线
    originalImg = res.add_subplot(2, 3, 3 * num + 1)
    centralRow = res.add_subplot(2, 3, 3 * num + 2)
    centralColumn = res.add_subplot(2, 3, 3 * num + 3)

    originalImg.set_title('Original Image')
    originalImg.imshow(img, cmap='gray')

    centralRow.set_title('Central Row')
    centralRow.plot(row)

    centralColumn.set_title('Central Column')
    centralColumn.plot(column)


if __name__ == '__main__':
    # *100为最终结果的size
    res = plt.figure(figsize=(16, 9))

    drawOneImg(res, 'cameraman.tif', 0)
    drawOneImg(res, 'einstein.tif', 1)

    # plt.savefig('test1_1.png', format='png')
    plt.show()