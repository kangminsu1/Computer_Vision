import numpy as np
from cv2 import *
import glob
# 종료 기준을 정한다
criteria = (TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#Object point(3d) 준비
objp = np.zeros((6*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1,2)

objpoint = []
imgpoint = []

images = glob.glob('*.jpg')

for i in images:
    img = imread(i)
    gray = cvtColor(img, COLOR_BGR2GRAY)

    ret, corner = findChessboardCorners(gray, (7, 6), None)

    if ret == True:
        objpoint.append(objp)

        corner = cornerSubPix(gray, corner, (11,11), (-1,-1), criteria)
        imgpoint.append(corner)

        img = drawChessboardCorners(img, (7,6), corner, ret)
        imshow('img',img)
        waitKey(0)
destroyAllWindows()