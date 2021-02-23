from cv2 import *
import numpy as np

if __name__ == '__main__':
    img = imread('image1.png')
    filter_for_image = (5,5)
    kernel = np.ones(filter_for_image, np.float32)/(filter_for_image[0]*filter_for_image[1])
    blur = blur(img, filter_for_image) #box filtering
    gaussian = GaussianBlur(img, filter_for_image, 0)

    imshow('blur',blur)
    imshow('gaussian',gaussian)

    waitKey(0)
    destroyAllWindows()