from cv2 import *
import numpy as np


if __name__ == "__main__":

    filters = (5,5)
    kernel = np.ones(filters, np.float32)/(filters[0]*filters[1])
    img = imread('lion.jpg')
    blur = blur(img, filters) #box filtering
    gaussian = GaussianBlur(img, filters, 0)
    convolution = filter2D(img, -1, kernel)

    #Sobel은 prewitt 보다 수평, 수직, 대각선 검출에 강함
    #Sobel(src, ddepth, dx, dy)
    # -1은 입력 이미지와 동일한 출력을 하게 하는 것
    Horizential = Sobel(img, -1, 1, 0)
    Vertical = Sobel(img, -1, 0, 1)
    Laplacians = Laplacian(img, -1, filters)

    imshow('blur',blur)
    imshow('gaussian',gaussian)
    imshow('convolution', convolution)
    imshow('Sobel_Horizential', Horizential)
    imshow('Sobel_Vertical', Vertical)

    imshow('Laplacian',Laplacians)
    waitKey(0)
    destroyAllWindows()
    #while(1):
        #if cv2.waitKey(0) == ord('q'):
            #break
        #a = 5
        #kernel = np.ones((a,a), np.float32)/(a^2)


        

        

