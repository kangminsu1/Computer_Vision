import numpy as np
from cv2 import *


def rotation(src):
    height, width, channel = src.shape
    matrix = getRotationMatrix2D((width/2, height/2), 45, 1)
    dst = warpAffine(src, matrix, (width, height))
    imshow("move to align the coins", dst)
    waitKey(0)
    destroyAllWindows()

def translation(img, height, width):
    M = np.float32([[1,0,-50],[0,1,60]])
    dst = warpAffine(img, M,(width, height))
    rotation(dst)

def hough_circle(img):
    img = medianBlur(img,5)
    find_image = cvtColor(img, COLOR_GRAY2BGR)

    circles = HoughCircles(img, HOUGH_GRADIENT,1,30,
                                param1=40,param2=50,minRadius=5,maxRadius=0)

    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw the outer circle
        circle(find_image,(i[0],i[1]),i[2],(255,0,0),2)
        # draw the center of the circle
        circle(find_image,(i[0],i[1]),2,(0,0,255),3)

    imshow('Detacting circles',find_image)
    waitKey(0)
    destroyAllWindows()

    height, width = find_image.shape[:2]
    translation(find_image, height, width)

if __name__ == "__main__":
    img = imread('image2_1.png', 0)
    hough_circle(img)