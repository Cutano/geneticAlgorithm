from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import random


def compareHist(tar, curr):
    tar = Image.open(tar)
    curr = Image.open(curr)
    mat_tar = np.asarray(tar, dtype=np.int32)
    mat_curr = np.asarray(curr, dtype=np.int32)
    result = abs(mat_tar - mat_curr)
    plt.imshow(result)
    plt.show()
    x_norm = np.linalg.norm(result)
    print(x_norm)
    # return imgocr[0, 1] > 0.96


if __name__ == '__main__':
    tar = 'edge_small.png'
    curr = 'result/gen50.png'
    compareHist(tar, curr)
