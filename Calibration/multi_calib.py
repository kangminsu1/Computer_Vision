#https://www.aiismath.com/pages/c_07_camera_calibration/multi_plane_calib_nb/

from cv2 import *
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

def draw(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-1)

    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)

    return img


square_size = 2.8
pattern_size = (9, 6)
figure_size = (20, 20)
img_mask = '*.jpg'
imgs = glob(img_mask)
num_image = len(imgs)

pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size
odj_points = []
img_points = []

h, w = imread(imgs[0]).shape[:2]

plt.figure(figsize=figure_size)

for i, f in enumerate(imgs):
    print("processing..%s"% f)
    imgBGR = imread(f)

    if imgBGR is None:
        print("failed", f)
        continue

    imgRGB = cvtColor(imgBGR, COLOR_BGR2RGB)
    img = cvtColor(imgBGR, COLOR_BGR2GRAY)

    assert w == img.shape[1] and h == img.shape[0], ("size: %d x %d ... " % (img.shape[1], img.shape[0]))
    found, corners = cv2.findChessboardCorners(img, pattern_size)

    if not found:
        print('chessboard not found')
        continue

    if i<12:
        img_w_corners = cv2.drawChessboardCorners(imgRGB, pattern_size, corners, found)
        plt.subplot(4,3,i+1)
        plt.imshow(img_w_corners)



    print('            %s... OK' % f)
    img_points.append(corners.reshape(-1, 2))
    odj_points.append(pattern_points)

plt.show()

rms, camera_matrix, dist_coefs, rvecs, tvecs = calibrateCamera(odj_points, img_points, (w, h), None, None)
print("\nRMS:", rms)
print("camera matrix:\n", camera_matrix)
print("distortion coefficients: ", dist_coefs.ravel())

# undistort the image with the calibration
plt.figure(figsize=figure_size)
for i,fn in enumerate(imgs):

    imgBGR = cv2.imread(fn)
    imgRGB = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2RGB)

    dst = cv2.undistort(imgRGB, camera_matrix, dist_coefs)

    if i<12:
        plt.subplot(4,3,i+1)
        plt.imshow(dst)

plt.show()
print('Done')

objectPoints = 3*square_size*np.array([[0,0,0], [0,1,0], [1,1,0], [1,0,0],[0,0,-1],[0,1,-1],[1,1,-1],[1,0,-1]])
plt.figure(figsize=figure_size)
for i, fn in enumerate(imgs):

    imgBGR = cv2.imread(fn)
    imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)

    dst = cv2.undistort(imgRGB, camera_matrix, dist_coefs)

    imgpts = cv2.projectPoints(objectPoints, rvecs[i], tvecs[i], camera_matrix, dist_coefs)[0]
    drawn_image = draw(dst, imgpts)

    if i < 12:
        plt.subplot(4, 3, i + 1)
        plt.imshow(drawn_image)

plt.show()