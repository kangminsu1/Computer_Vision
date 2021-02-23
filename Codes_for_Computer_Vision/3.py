from cv2 import *
import numpy as np


def houghlines(gray):
    #canny(image, min threshold, max threshold, argument)
    edges = Canny(gray, 150, 500, 3)
    imshow('edgez', edges)
    lines = HoughLines(edges, 1, np.pi/180, 100)

    for r, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x = a*r
        y = b*r
        x1 = int(x+ 1000*(-b))
        y1 = int(y+ 1000*(a))
        x2 = int(x- 1000*(-b))
        y2 = int(y- 1000*(a))

        line(img,(x1,y1),(x2, y2), (0,255,255),2)
    imshow('lines',img)

    waitKey(0)
    destroyAllWindows()

def houghcircle():
    img = imread('circles.jpg')
    img = medianBlur(img,3)
    grays = cvtColor(img, COLOR_GRAY2BGR)
    circle = HoughCircles(img, HOUGH_GRADIENT, 1, 20, 50, 30, 0, 0)
    circle = np.unit16(np.around(circle))
    for i in circle[0,:]:
        circle(grays,(i[0],i[1]),i[2],(0,0,255),2)

    imwrite('teected', grays)
    imshow('deteected', grays)
    waitKey(0)
    destroyAllWindows()

if __name__ == '__main__':
    img = imread('circles.jpg')
    imshow('origine', img)

    gray = cvtColor(img, COLOR_BGR2GRAY)
    imshow('GRAY', gray)
    #houghlines(gray)
    houghcircle()

    