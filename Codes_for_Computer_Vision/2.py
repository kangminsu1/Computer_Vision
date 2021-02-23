from cv2 import *
import numpy as np
import matplotlib.pyplot as plt


def Gaussian_pyrimid(img):
    copy_img = img.copy()

    titles = ['Original', 'Level1', 'Level2', 'Level3']
    g_down = []
    g_up = []

    g_down.append(copy_img)

    for i in range(3):
        temp1 = pyrDown(copy_img)
        g_down.append(temp1)
        copy_img = temp1

    imshow('Level3', copy_img)

    for i in range(4):
        imshow(titles[i], g_down[i])

    waitKey(0)
    destroyAllWindows()

def Gaussian_upsampling(img):
    copy_img = img.copy()

    titles = ['Original', 'Level1', 'Level2', 'Level3']
    g_down = []
    g_up = []

    g_down.append(copy_img)

    for i in range(3):
        temp1 = pyrDown(copy_img)
        g_down.append(temp1)
        copy_img = temp1

    imshow('Level3', copy_img)

    for i in range(3):
        copy_img = g_down[i+1]
        temp1 = pyrUp(copy_img)
        g_up.append(temp1)

    for i in range(3):
        imshow(titles[i], g_up[i])

    waitKey(0)
    destroyAllWindows()

def Laplacian_pyramid(img):
    copy_img = img.copy()

    titles = ['Original', 'Level1', 'Level2', 'Level3']
    g_down = []
    g_up = []
    image_append = []

    g_down.append(copy_img)
    image_append.append(copy_img.shape)

    for i in range(3):
        temp1 = pyrDown(copy_img)
        g_down.append(temp1)
        image_append.append(temp1.shape)
        copy_img = temp1 

    for i in range(3):
        copy_img = g_down[i+1]
        temp1 = pyrUp(copy_img)
        copy_img = resize(temp1, dsize=(image_append[i][1], image_append[i][0]), interpolation=INTER_CUBIC)
        g_up.append(copy_img)

    for i in range(3):
        copy_img = subtract(g_down[i], g_up[i])
        imshow(titles[i], copy_img)

        #cc = add(g_up[i], copy_img)
        #imshow(titles[i], cc)
        #imshow("titles[i]", img)

    waitKey(0)
    destroyAllWindows()

def fourier(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f) #주파수 영역 한군데로 모으기
    m_specturm = 20 * np.log(np.abs(fshift))

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('input image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(m_specturm, cmap='gray')
    plt.title('m_specturm'), plt.xticks([]), plt.yticks([])
    plt.show()
    #저주파 영역 60*60

    #imshow('title', img)
    #imshow('changed', m_specturm)
    #waitKey(0)
    #destroyAllWindows()


def LPF(img):
    dft = cv2.dft(np.float32(img), flags=DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))
    

if __name__ == "__main__":
    #IMREAD_GRAYSCALE
    img = imread('123.jpg', IMREAD_GRAYSCALE)
    #Gaussian_pyrimid(img)
    #Gaussian_upsampling(img)
    #Laplacian_pyramid(img)
    #fourier(img)
    LPF(img)

