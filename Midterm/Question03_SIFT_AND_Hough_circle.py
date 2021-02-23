import numpy as np
from cv2 import *

#first i choosed sift discriptor.
def sift(img):
    img_gray = cvtColor(img, COLOR_GRAY2BGR)
    sift = xfeatures2d.SIFT_create()
    kernel = sift.detect(img_gray, None)
    image_02 = None 

    image_02 = drawKeypoints(img_gray, kernel, image_02)

    imshow('sift', image_02)
    waitKey(0)
    destroyAllWindows()
# but sift dosen't detected accuracy so i decided hough circle discriptor
def hough_circle(img):
    find_image = cvtColor(img, COLOR_GRAY2BGR)

    circles = HoughCircles(img, HOUGH_GRADIENT,1,60,
                                param1=50,param2=60,minRadius=10,maxRadius=0)

    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw the outer circle
        circle(find_image,(i[0],i[1]),i[2],(255,0,0),2)
        # draw the center of the circle
        circle(find_image,(i[0],i[1]),2,(0,0,255),3)

    imshow('Detacting circles',find_image)
    waitKey(0)
    destroyAllWindows()

if __name__ == "__main__":
    img = imread('image2_2.png', 0)
    sift(img)
    hough_circle(img)