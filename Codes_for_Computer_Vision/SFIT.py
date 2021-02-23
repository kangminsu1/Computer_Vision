from cv2 import *
import numpy as np

img = imread('lion.jpg')
imggray = cvtColor(img, COLOR_BGR2GRAY)
img2, img3 = None, None

sift = xfeatures2d.SIFT_create()
kp = sift.detect(imggray, None)
img2 = drawKeypoints(imggray, kp, img2)
img3 = drawKeypoints(imggray, kp, img3, DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

imshow('sifi',img2)
imshow('sifi2',img3)
waitKey(0)
destroyAllWindows()