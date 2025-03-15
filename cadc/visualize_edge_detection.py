import numpy as np
import pandas as pd
import PIL.Image as Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.ion()
import cv2

BASE = "/home/datasets/CADC/cadcd/"
BASE_mod = "/home/datasets_mod/CADC/cadcd/"
cadc_stats = pd.read_csv('cadc_dataset_route_stats.csv', header=0, usecols=[0, 1, 2, 18, 19, 20, 21])
start_row = 0  # 25
cam = 0


def sobel(img_grayscale):
    sobelx = cv2.Sobel(img_grayscale, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(img_grayscale, cv2.CV_64F, 0, 1)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.convertScaleAbs(sobely)
    return cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)


def scharr(img_grayscale):
    scharrx = cv2.Scharr(img_grayscale, cv2.CV_64F, 1, 0)
    scharry = cv2.Scharr(img_grayscale, cv2.CV_64F, 0, 1)
    scharrx = cv2.convertScaleAbs(scharrx)
    scharry = cv2.convertScaleAbs(scharry)
    return cv2.addWeighted(scharrx, 0.5, scharry, 0.5, 0)


def laplacian(img_grayscale):
    laplacian = cv2.Laplacian(img_grayscale, cv2.CV_64F)
    return cv2.convertScaleAbs(laplacian)


for row in range(start_row, len(cadc_stats)):
    print('\n------ Date %s, Sequence %d ------' % (cadc_stats.iloc[row, 0], cadc_stats.iloc[row, 1]))
    date = cadc_stats.iloc[row, 0]
    seq = cadc_stats.iloc[row, 1]
    n_frame = cadc_stats.iloc[row, 2]
    start_frame = 10

    for frame in range(start_frame, n_frame):
        # visualization
        img_path = BASE + date + '/' + format(seq, '04') + "/labeled" + "/image_0" + str(cam) + \
                   "/data/" + format(frame, '010') + ".png"
        img = plt.imread(img_path)
        img_grayscale = cv2.cvtColor(np.array(Image.open(img_path).convert("RGB"))[...,::-1], cv2.COLOR_BGR2GRAY)
        img_sobel = sobel(img_grayscale)
        img_scharr = scharr(img_grayscale)
        img_laplacian = laplacian(img_grayscale)
        img_canny = cv2.Canny(img_grayscale, 0, 100)


        fig1 = plt.figure(1)
        ax1 = fig1.subplots(2, 2)
        ax1[0, 0].clear()
        ax1[0, 0].imshow(img, cmap='gray')
        ax1[0, 0].set_title('RGB image')

        ax1[0, 1].clear()
        ax1[0, 1].imshow(img_sobel, cmap='gray')
        ax1[0, 1].set_title('sobel')

        ax1[1, 0].clear()
        ax1[1, 0].imshow(img_scharr, cmap='gray')
        ax1[1, 0].set_title('scharr')

        ax1[1, 1].clear()
        ax1[1, 1].imshow(img_canny, cmap='gray')
        ax1[1, 1].set_title('canny')

        plt.pause(0.1)
        plt.show()
