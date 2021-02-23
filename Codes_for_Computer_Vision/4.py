from cv2 import *
import numpy as np

if __name__ == "__main__":
    img = imread('rectangle.jpg')
    img_gray = cvtColor(img, COLOR_BGR2GRAY)
    img_sobel_x = Sobel(img_gray, CV_32F, 1, 0)
    img_sobel_y = Sobel(img_gray, CV_32F, 0, 1)

    IXIX = img_sobel_x * img_sobel_x
    IYIY = img_sobel_y * img_sobel_y
    IXIY = img_sobel_x * img_sobel_y

    height, width = img.shape[:2]

    window_size = 5
    offset = int(window_size/2)

    r = np.zeros(img_gray.shape)

    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            window_ixix = IXIX[y-offset:y-offset+1, x-offset: x+offset+1]
            window_iyiy = IYIY[y-offset:y+offset+1, x-offset: x+offset+1]
            window_ixiy = IXIY[y-offset:y+offset+1, x-offset: x+offset+1]

            mxx = window_ixix.sum()
            myy = window_iyiy.sum()
            mxy = window_ixiy.sum()


            dst = mxx*myy - mxy*myy
            trace = mxx + myy

            r[y,x] = dst - 0.04 * (trace ** 2)

    normalize(r,r, 0.0, 1.0, NORM_MINMAX)

    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            if r[y, x] > 0.4:
                img.itemset((y, x, 0), 0)
                img.itemset((y, x, 1), 0)
                img.itemset((y, x, 2), 255)

    imshow('original', img)
    waitKey(0)