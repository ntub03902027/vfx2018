#! /usr/bin/env python

import os
import numpy as np
import cv2
import scipy.spatial.ckdtree as kdTree

testpath = 'test/parrington'
nImages = 18
pre = 'prtn'
suf = '.jpg'

def getImageFilename(i, prefix='', suffix='', digit=0):
    if digit == 0:
        return prefix + str(i) + suffix
    
    n = digit - len(str(i))
    if n < 0:
        n = 0
    return prefix + n * '0' + str(i) + suffix
    


"""
    the descriptor vector will be (2 * size - 1)**2
"""
def harrisDescriptor(image, fPoint, size):
    if size < 1:
        print("harrisDescriptor: size must be larger than 1!!")
        exit(1)

    des = []
    for point in fPoint:
        vec = []
        for i in range(point[0] - size + 1, point[0] + size):
            for j in range(point[1] - size + 1, point[1] + size):
                if i < 0 or i >= image.shape[0] or j < 0 or j >= image.shape[1]:
                    vec.append(0)
                else:
                    vec.append(image[i,j])

        des.append(vec)

    return des



"""
Harris corner detection:
    INPUT: an rgb image
    OUTPUT:
"""
def harrisCornerDetection(image):

    print("Harris corner detecting...")
    # k: empricially 0.04 - 0.06
    k = 0.05
    
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   
    # 1. x, y derivatives
    # Sobel operators
    I_x = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, ksize=-1)
    I_y = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, ksize=-1)

    # 2. products of derivatives
    I_xx = I_x * I_x
    I_yy = I_y * I_y
    I_xy = I_x * I_y

    # 3. sums of products at each pixel
    S_xx = cv2.GaussianBlur(I_xx, (5, 5), 1.)
    S_yy = cv2.GaussianBlur(I_yy, (5, 5), 1.)
    S_xy = cv2.GaussianBlur(I_xy, (5, 5), 1.)


    # 4. response map R 
    R = np.zeros(S_xx.shape)
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            Mij = np.array([[S_xx[i,j], S_xy[i,j]], [S_xy[i,j], S_yy[i,j]]])
            R[i,j] = np.linalg.det(Mij) - k * (np.trace(Mij) ** 2)

    # find the N largest points as features

    N = 1024
    
    arr = abs(R.reshape([R.shape[0] * R.shape[1]]))
    largest = arr.argsort()[-N:][::-1] # argsort: sort corresponding index from lowest to largest; [-N:]: the last N (i.e. largest) elements; [::-1]: reverse ordering
    x = np.floor_divide(largest, R.shape[1])
    y = largest % R.shape[1]
    fPoint = list(zip(x.tolist(), y.tolist()))


    des = harrisDescriptor(grayscale, fPoint, 3)
    
    return fPoint, des

    # output featured image
    fImage = image
    for point in fPoint:
        fImage[point[0],point[1],0] = 255
        fImage[point[0],point[1],1] = 0
        fImage[point[0],point[1],2] = 0
        
    

    #dst = cv2.cornerHarris(grayscale, 5, 19, 1)
    
    #m = np.zeros(S_xx.shape)
    #for i in range(m.shape[0]):
    #    for j in range(m.shape[1]):
    #        if abs(R[i,j]) >= 1000000000000000:
    #            m[i,j] = 255

    #print(np.min(R), np.min(dst))
    #print(np.max(R), np.max(dst))

    cv2.imshow('image', fImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    images = {}
    fPoints = {}
    descriptors = {}
    for i in range(nImages):
        images[i] = cv2.imread(os.path.join(testpath, getImageFilename(i, prefix=pre, suffix=suf, digit=2)))
        fPoints[i], descriptors[i] = harrisCornerDetection(images[i])
    
