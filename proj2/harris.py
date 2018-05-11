#! /usr/bin/env python

import os
import numpy as np
import cv2

testpath = 'test/'


k = 1.



if __name__ == '__main__':
    
    image = cv2.imread(os.path.join(testpath, 'parrington/prtn00.jpg'))
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Sobel operators
    I_x = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, ksize=-1)
    I_y = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, ksize=-1)

    I_xx = I_x * I_x
    I_yy = I_y * I_y
    I_xy = I_x * I_y


    S_xx = cv2.GaussianBlur(I_xx, (5, 5), 1.)
    S_yy = cv2.GaussianBlur(I_yy, (5, 5), 1.)
    S_xy = cv2.GaussianBlur(I_xy, (5, 5), 1.)


    

    R = np.zeros(S_xx.shape)
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            Mij = np.array([[S_xx[i,j], S_xy[i,j]], [S_xy[i,j], S_yy[i,j]]])
            R[i,j] = np.linalg.det(Mij) - k * (np.trace(Mij) ** 2)

    dst = cv2.cornerHarris(grayscale, 5, 19, 1)

    m = np.zeros(S_xx.shape)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if abs(R[i,j]) >= 1000000000000000:
                m[i,j] = 255

    print(np.min(R), np.min(dst))
    print(np.max(R), np.max(dst))

    cv2.imshow('image', m)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
