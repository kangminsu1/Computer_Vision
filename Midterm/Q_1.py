import numpy as np
import cv2
#OPENCV VERSION 4.4

a = cv2.imread('image_2.2.png')
b = cv2.imread('image_2.1.png')

A_GRAY = cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
B_GRAY = cv2.cvtColor(b,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
ka, da = sift.detectAndCompute(A_GRAY, None)
kb, db = sift.detectAndCompute(B_GRAY, None)

buff = cv2.BFMatcher()
m = buff.match(da, db)

sorting = sorted(m, key = lambda x : x.distance)
rest = cv2.drawMatches(a, ka, b, kb, sorting[:30], None, flags=2)

src = np.float32([ka[i.queryIdx].pt for i in m]).reshape((-1, 1, 2))
dst = np.float32([kb[i.trainIdx].pt for i in m]).reshape((-1, 1, 2))
H, stat = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

bef = []
for x in range(a.shape[1], a.shape[1]*2):
    for y in range(a.shape[0]):
        point = [x,y,1]
        bef.append(point)
bef = np.array(bef).transpose()
Hinv = np.linalg.inv(H)
aft = np.matmul(Hinv, bef)
aft = aft / aft[2, :]
aft = aft[:2, :]
aft = np.round(aft, 0).astype(np.int)

height, width, _ = a.shape
temp1 = np.zeros((height, width * 2, 3), dtype=np.uint8)
for pt1, pt2 in zip(bef[:2, :].transpose(), aft.transpose()):
    if pt2[1] >= height or pt2[0] >= width:
        continue

    if np.sum(pt2 < 0) >= 1:
        continue

    temp1[pt1[1], pt1[0]] = a[pt2[1], pt2[0]]
temp1[0: height, 0: width] = b
cv2.imshow('temp1', temp1)

temp2 = cv2.warpPerspective(a, H, (a.shape[1] + b.shape[1], b.shape[0]))
temp2[0: a.shape[0], 0: b.shape[1]] = b
cv2.imshow('result', temp2)
cv2.waitKey(0)
cv2.destroyAllWindows()