from cv2 import * 
import numpy as np
from matplotlib import pyplot as plt

def scaling(img, height, width):
    # 이미지 축소
    shrink = resize(img, None, fx=0.5, fy=0.5, interpolation=INTER_AREA)

# Manual Size지정
    zoom1 = resize(img, (width*2, height*2), interpolation=INTER_CUBIC)

# 배수 Size지정
    zoom2 = resize(img, None, fx=2, fy=2, interpolation=INTER_CUBIC)


    imshow('Origianl', img)
    imshow('Shrink', shrink)
    imshow('Zoom1', zoom1)
    imshow('Zoom2', zoom2)

    waitKey(0)
    destroyAllWindows()

def translation(img, height, width):
    # 변환 행렬, X축으로 10, Y축으로 20 이동
    M = np.float32([[1,0,10],[0,1,20]])

    dst = warpAffine(img, M,(width, height))
    imshow('Original', img)
    imshow('Translation', dst)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def affine(img):
    rows, cols, ch = img.shape

    pts1 = np.float32([[200,100],[400,100],[200,200]])
    pts2 = np.float32([[200,300],[400,200],[200,400]])

    # pts1의 좌표에 표시. Affine 변환 후 이동 점 확인.
    cv2.circle(img, (200,100), 10, (255,0,0),-1)
    cv2.circle(img, (400,100), 10, (0,255,0),-1)
    cv2.circle(img, (200,200), 10, (0,0,255),-1)

    M = cv2.getAffineTransform(pts1, pts2)

    dst = cv2.warpAffine(img, M, (cols,rows))

    plt.subplot(121),plt.imshow(img),plt.title('image')
    plt.subplot(122),plt.imshow(dst),plt.title('Affine')
    plt.show()

def perspective_transformation(img):
        # [x,y] 좌표점을 4x2의 행렬로 작성
    # 좌표점은 좌상->좌하->우상->우하
    pts1 = np.float32([[504,1003],[243,1525],[1000,1000],[1280,1685]])

    # 좌표의 이동점
    pts2 = np.float32([[10,10],[10,1000],[1000,10],[1000,1000]])

    # pts1의 좌표에 표시. perspective 변환 후 이동 점 확인.
    cv2.circle(img, (504,1003), 20, (255,0,0),-1)
    cv2.circle(img, (243,1524), 20, (0,255,0),-1)
    cv2.circle(img, (1000,1000), 20, (0,0,255),-1)
    cv2.circle(img, (1280,1685), 20, (0,0,0),-1)

    M = cv2.getPerspectiveTransform(pts1, pts2)

    dst = cv2.warpPerspective(img, M, (1100,1100))

    plt.subplot(121),plt.imshow(img),plt.title('image')
    plt.subplot(122),plt.imshow(dst),plt.title('Perspective')
    plt.show()

def rotation():
    src = cv2.imread("lion.jpg", cv2.IMREAD_COLOR)

    height, width, channel = src.shape
    matrix = cv2.getRotationMatrix2D((width/2, height/2), 45, 1)
    dst = cv2.warpAffine(src, matrix, (width, height))

    cv2.imshow("src", src)
    cv2.imshow("dst", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def transform(src):
    height, width, channel = src.shape
    dst = cv2.pyrUp(src, dstsize=(width*2, height*2), borderType=cv2.BORDER_DEFAULT)
    dst2 = cv2.pyrDown(src)

    cv2.imshow("src", src)
    cv2.imshow("dst", dst)
    cv2.imshow("dst2", dst2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':

    img = imread('lane.jpg')

    # 행 : Height, 열:width
    height, width = img.shape[:2]
    #translation(img, height, width)
    #affine(img)
    #perspective_transformation(img)
    #rotation()
    transform(img)

