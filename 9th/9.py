import numpy as np
from matplotlib import pyplot as plt
import cv2


def erode(a, b):
    # 结构元反射后再卷积，相当于直接求相关
    # opencv中该函数其实是求的相关
    dst = cv2.filter2D(a, -1, b, borderType=cv2.BORDER_CONSTANT)
    sum_b = np.sum(b)
    dst = np.where(dst == sum_b, 1, 0)
    return dst.astype(np.uint8)


def dilate(a, b):
    # 结构元进行卷积，需要旋转180°
    b_reflect = np.rot90(b, 2)
    dst = cv2.filter2D(a, -1, b_reflect, borderType=cv2.BORDER_CONSTANT)
    dst = np.where(dst > 0, 1, 0)
    return dst.astype(np.uint8)


def hit_miss(a, b):
    b1 = np.where(b == 1, 1, 0)
    b2 = np.where(b == 0, 1, 0)
    # 填充一下以解决边界
    padding = cv2.copyMakeBorder(a, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

    eroded = erode(padding, b1)

    a_not = 1 - padding
    eroded2 = erode(a_not, b2)

    # 去除填充边界
    dst = cv2.bitwise_and(eroded, eroded2)[1:-1, 1:-1]
    return dst.astype(np.uint8)


def thin(f, b):
    hit_miss_res = hit_miss(f, b)

    # 记录每个像素是不是连通的
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    neighbor_num = cv2.filter2D(f, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    connected = np.where(neighbor_num == 0, 0, 1)

    # 击中击不中变换中连通的像素才需要被删除
    deleted = cv2.bitwise_and(hit_miss_res, connected.astype(np.uint8))

    return cv2.subtract(f, deleted)


def morphological_skeleton_extract(binary):
    b = []
    b.append(np.array([[0, 0, 0], [-1, 1, -1], [1, 1, 1]]))
    b.append(np.array([[-1, 0, 0], [1, 1, 0], [1, 1, -1]]))
    b.append(np.array([[1, -1, 0], [1, 1, 0], [1, -1, 0]]))
    b.append(np.array([[1, 1, -1], [1, 1, 0], [-1, 0, 0]]))
    b.append(np.array([[1, 1, 1], [-1, 1, -1], [0, 0, 0]]))
    b.append(np.array([[-1, 1, 1], [0, 1, 1], [0, 0, -1]]))
    b.append(np.array([[0, -1, 1], [0, 1, 1], [0, -1, 1]]))
    b.append(np.array([[0, 0, -1], [0, 1, 1], [-1, 1, 1]]))

    dst = binary.copy()
    # 迭代次数
    thin_num = 0
    # 利用b中的核不断进行细化直到细化前后无变化
    while True:
        isConverged = False
        for bi in b:
            thinned = thin(dst, bi)
            if (thinned == dst).all():
                isConverged = True
                break
            else:
                dst = thinned
                thin_num += 1
        if isConverged:
            break
    return dst.astype(np.uint8), thin_num


# 利用腐蚀膨胀提取边缘
def edge_extract(a):
    b = np.ones((3, 3), np.uint8)
    return cv2.subtract(a, erode(a, b))


# 距离变换，改写自matlab文件
def distance_transform(img):
    height, width = img.shape
    A = np.where(img == 0, np.Inf, 1)
    padding = cv2.copyMakeBorder(A, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=np.inf)
    for i in range(1, height): 
        for j in range(1, width - 1):  
            temp1 = min(padding[i][j-1] + 3, padding[i][j])
            temp2 = min(padding[i-1][j-1] + 4, temp1)
            temp3 = min(padding[i-1][j] + 3, temp2)
            padding[i][j] = min(padding[i-1][j+1]+4, temp3)
    for i in range(height - 2, -1, -1): 
        for j in range(width - 2, 0, -1): 
            temp1 = min(padding[i][j+1] + 3, padding[i][j])
            temp2 = min(padding[i+1][j+1] + 4, temp1)
            temp3 = min(padding[i+1][j] + 3, temp2)
            padding[i][j] = min(padding[i+1][j+1]+4, temp3)
    D = np.round(padding[:, 1:width-1]/3)
    return D


def get_local_max_img(img):
    dst = np.zeros_like(img)
    height, width = img.shape
    padding = img.copy()
    padding = cv2.copyMakeBorder(
        padding, 3, 3, 3, 3, borderType=cv2.BORDER_CONSTANT, value=np.inf)

    # 每个像素的7*7邻域极大
    for i in range(height):
        for j in range(width):
            neighbor = padding[i:i+7, j:j+7]
            if img[i][j] == np.max(neighbor):
                dst[i][j] = 1
    return dst.astype(np.uint8)

def distance_skeleton_extract(binary):
    edge_img = edge_extract(binary)
    dis_img = distance_transform(edge_img)
    distance_skeleton = get_local_max_img(dis_img)
    return distance_skeleton

def cut(a):
    b = []
    b.append(np.rot90(np.array([[0, 0, 0], [-1, 1, -1], [1, 1, 1]]), 1))
    b.append(np.rot90(np.array([[-1, 0, 0], [1, 1, 0], [1, 1, -1]]), 1))
    b.append(np.rot90(np.array([[1, -1, 0], [1, 1, 0], [1, -1, 0]]), 1))
    b.append(np.rot90(np.array([[1, 1, -1], [1, 1, 0], [-1, 0, 0]]), 1))
    b.append(np.rot90(np.array([[1, 1, 1], [-1, 1, -1], [0, 0, 0]]), 1))
    b.append(np.rot90(np.array([[-1, 1, 1], [0, 1, 1], [0, 0, -1]]), 1))
    b.append(np.rot90(np.array([[0, -1, 1], [0, 1, 1], [0, -1, 1]]), 1))
    b.append(np.rot90(np.array([[0, 0, -1], [0, 1, 1], [-1, 1, 1]]), 1))

    x1 = a.copy()
    for bi in b:
        x1 = thin(x1, bi)

    x2 = np.zeros_like(x1)
    for bi in b:
        x2_component = hit_miss(x1, bi)
        x2 = np.bitwise_or(x2, x2_component)

    H = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    # 裁剪中进行腐蚀的次数
    ERODE_NUM = 3
    eroded = x2.copy()
    for i in range(ERODE_NUM):
        eroded = erode(eroded, H)

    x3 = np.bitwise_and(eroded, a)

    return np.bitwise_or(x1, x3)


if __name__ == "__main__":
    img = cv2.imread('smallfingerprint.jpg', cv2.IMREAD_GRAYSCALE)

    _, binary = cv2.threshold(img, 150, 1, cv2.THRESH_BINARY_INV)
    binary = binary.astype(np.uint8)

    morphological_skeleton, thin_num = morphological_skeleton_extract(binary)
    morphological_skeleton_cut = cut(morphological_skeleton)

    distance_skeleton = distance_skeleton_extract(binary)
    distance_skeleton_cut = cut(distance_skeleton)

    fig = plt.figure(figsize=(8, 6))
    plt.subplot2grid((2, 4), (0, 0), rowspan=2), plt.imshow(img, cmap='gray'), plt.title(
        'Original', fontsize=6), plt.axis('off')
    plt.subplot2grid((2, 4), (0, 1), rowspan=2), plt.imshow(binary, cmap='gray'), plt.title(
        'Binary', fontsize=6), plt.axis('off')

    plt.subplot2grid((2, 4), (0, 2)), plt.imshow(morphological_skeleton, cmap='gray'), plt.title(
        'Morphological Skeleton, iteration=' + str(thin_num), fontsize=6), plt.axis('off')
    plt.subplot2grid((2, 4), (0, 3)), plt.imshow(morphological_skeleton_cut, cmap='gray'), plt.title(
        'Cut', fontsize=6), plt.axis('off')

    plt.subplot2grid((2, 4), (1, 2)), plt.imshow(distance_skeleton, cmap='gray'), plt.title(
        'Distance Skeleton', fontsize=6), plt.axis('off')
    plt.subplot2grid((2, 4), (1, 3)), plt.imshow(distance_skeleton_cut, cmap='gray'), plt.title(
        'cut', fontsize=6), plt.axis('off')
    plt.show()
