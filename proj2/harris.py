#! /usr/bin/env python

import os
import csv
import numpy as np
import cv2
import scipy.spatial.ckdtree as KDTree
import sys
import math
import random
import copy

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
    
def cylindricalProjection(image, f, showResult=False):
    xCen = int(image.shape[1] / 2)
    yCen = int(image.shape[0] / 2)
    cImage = np.zeros([image.shape[0], image.shape[1], image.shape[2]], dtype=np.uint8)

    s = f
    for i in range(cImage.shape[1]):
        for j in range(cImage.shape[0]):
            x = i - xCen
            y = j - yCen
            theta = math.asin(x / math.sqrt(x**2 + f**2))
            h = y / math.sqrt(x**2 + f**2)

            xc = int(s * theta + xCen)
            yc = int(s * h + yCen)
            
            if xc >= 0 and xc < cImage.shape[1] and yc >= 0 and yc < cImage.shape[0]:
                cImage[yc,xc,0] = image[j,i,0]
                cImage[yc,xc,1] = image[j,i,1]
                cImage[yc,xc,2] = image[j,i,2]


    # crop left/right/up/down blankings
    left = 0
    right = cImage.shape[1] - 1
    up = 0
    down = cImage.shape[0] - 1
    while isBlank(cImage, left):
        left += 1
    while isBlank(cImage, right):
        right -= 1
    while isBlank(cImage, up, horizontal=True):
        up += 1
    while isBlank(cImage, down, horizontal=True):
        down -= 1

    cImage = cImage[up:down+1,left:right+1,:]

    if showResult:
        cv2.imshow('image', cImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    
    return cImage

def isBlank(image, i, horizontal=False):
    flag = True
    if horizontal:
        for j in range(image.shape[1]):
            if image[i,j,0] != 0 or image[i,j,1] != 0 or image[i,j,2] != 0:
                flag = False
                break
    else:
        for j in range(image.shape[0]):
            if image[j,i,0] != 0 or image[j,i,1] != 0 or image[j,i,2] != 0:
                flag = False
                break

    return flag

    pass

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
        
        # put the vertical coordinate into the descriptor is based on the assumption that the vertical differences between the two feature points should be small
        vec.append(point[0])

        des.append(vec)

    return des



"""
Harris corner detection:
    INPUT: an rgb image
    OUTPUT:
"""
def harrisCornerDetection(image, showResult=False):

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

    # 5.
    # find the N largest points as features

    #N = 2048
    N = int(R.shape[0] * R.shape[1] * 0.01 * 5) # highest i%
    
    arr = abs(R.reshape([R.shape[0] * R.shape[1]]))
    largest = arr.argsort()[-N:][::-1] # argsort: sort corresponding index from lowest to largest; [-N:]: the last N (i.e. largest) elements; [::-1]: reverse ordering
    x = np.floor_divide(largest, R.shape[1])
    y = largest % R.shape[1]
    fPoint = list(zip(x.tolist(), y.tolist()))

    # remove edge features
    # remove features due to cylindrical projection
    maxDistort = 0
    while image[maxDistort,0,0] == 0 and image[maxDistort,0,1] == 0 and image[maxDistort,0,2] == 0: 
        maxDistort += 1

    
    tmp = []
    for p in fPoint:
        if p[0] < maxDistort + 5 or p[1] < 2 or p[0] >= R.shape[0] - maxDistort - 6 or p[1] >= R.shape[1] - 2:
            tmp.append(p)
    for p in tmp:
        fPoint.remove(p)
    del tmp


    # 2*2 max pooling
    maxPool = set({})
    for i in range(2,R.shape[0]-1, 2):
        for j in range(2,R.shape[1]-1, 2):
            tmp = [(R[i,j], (i,j)), (R[i+1,j], (i+1,j)), (R[i,j+1], (i,j+1)), (R[i+1, j+1], (i+1, j+1))]
            maxPool.add(max(tmp)[1])

    if showResult:
        fPointOld = fPoint
    fPoint = list(maxPool.intersection(set(fPoint)))

    # find descriptors
    des = harrisDescriptor(grayscale, fPoint, 3)
    


    if showResult:
    # output featured image
        fImage = copy.copy(image)
        for point in fPointOld:
            if point in fPoint:
                fImage[point[0],point[1],0] = 255
                fImage[point[0],point[1],1] = 0
                fImage[point[0],point[1],2] = 0
            else:
                fImage[point[0],point[1],0] = 0
                fImage[point[0],point[1],1] = 255
                fImage[point[0],point[1],2] = 255

    

        #dst = cv2.cornerHarris(grayscale, 5, 19, 1)
    

        #print(np.min(R), np.min(dst))
        #print(np.max(R), np.max(dst))

        cv2.imshow('image', fImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return fPoint, des

def featureMatching(des1, des2, k=4):
    kdtree2 = KDTree.cKDTree(des2)

    nn = []

    n = 0
    for d1 in des1:
        d, i = kdtree2.query([d1], k=k)
        nn.append((n, d, i))
        #print(n, d) 
        n += 1

    nn = sorted(nn, key=lambda a: a[1][0][0])
    
    
    matchedPoints = []
    for x in nn:
        print(x)
        if x[1][0][0] < 175. and x[1][0][0] <= 0.7 * x[1][0][1]:
            matchedPoints.append((x[0], x[2][0][0]))
    
    # return a list of matched points of format 2-tuple: ([ind of 1st fPoint], [ind of 2nd fPoint])
    return matchedPoints

   
def printFeatureMatchPoints(matchedPoints, image1, image2, fPoint1, fPoint2):

    a = np.zeros([image1.shape[0], 2 * image1.shape[1], image1.shape[2]], dtype=np.uint8)
    grayscale1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    grayscale2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    for i in range(a.shape[0]):
        for j in range(int(a.shape[1] / 2)):
            a[i,j,0] = grayscale1[i,j]
            a[i,j,1] = grayscale1[i,j]
            a[i,j,2] = grayscale1[i,j]
            a[i,j+image1.shape[1],0] = grayscale2[i,j]
            a[i,j+image1.shape[1],1] = grayscale2[i,j]
            a[i,j+image1.shape[1],2] = grayscale2[i,j]
   
    for fPoint in fPoint1:
        a[fPoint[0],fPoint[1],0] = 255
        a[fPoint[0],fPoint[1],1] = 0
        a[fPoint[0],fPoint[1],2] = 0
    for fPoint in fPoint2:
        a[fPoint[0],fPoint[1]+ image1.shape[1],0] = 255
        a[fPoint[0],fPoint[1]+ image1.shape[1],1] = 0
        a[fPoint[0],fPoint[1]+ image1.shape[1],2] = 0

    # caution, when drawing a line, the coord is (y, x), not (x, y)!!!!!!!!!!!!!!!!
    
    for x in matchedPoints:
        cv2.line(a, (fPoint1[x[0]][1], fPoint1[x[0]][0]), (fPoint2[x[1]][1] + image1.shape[1], fPoint2[x[1]][0]), (55, 255, 155) )
        pass

    cv2.imshow('image', a)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def imageStitch(trans, image1, image2, printResult=True):
    result = np.zeros([max(image1.shape[0] + abs(trans[0]), image2.shape[0]), max(image1.shape[1] + abs(trans[1]), image2.shape[1]), image1.shape[2]], dtype=np.uint8)

    startX1 = abs(min(trans[0], 0))
    startY1 = abs(min(trans[1], 0))

    startX2 = abs(max(trans[0], 0))
    startY2 = abs(max(trans[1], 0))

    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            result[i+startX1,j+startY1,0] = image1[i,j,0]
            result[i+startX1,j+startY1,1] = image1[i,j,1]
            result[i+startX1,j+startY1,2] = image1[i,j,2]
   
    yOverlap = image1.shape[1] + image2.shape[1] - result.shape[1]
    for i in range(image2.shape[0]):
        for j in range(image2.shape[1]):
            if result[i+startX2,j+startY2,0] == 0 and result[i+startX2,j+startY2,1] == 0 and result[i+startX2,j+startY2,2] == 0:
                result[i+startX2,j+startY2,:] = image2[i,j,:]
            else:
                # blending
                coef = float(trans[1] > 0) + (float(trans[1] <= 0) - float(trans[1] > 0))* ((j+startY2) - abs(trans[1])) / yOverlap 
                result[i+startX2,j+startY2,:] = (coef * image2[i,j,:] + (1. - coef) * result[i+startX2,j+startY2,:]).astype(np.uint8)
                pass

    if printResult:
        cv2.imshow('image', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return result



def ransac(matchedPoints, fPoints1, fPoints2, n=2, p=0.6, P=0.99, kMin=64):
    k = max(math.ceil(math.log(1-P)/math.log(1-p**n)), kMin)

    transX = 0
    transY = 0
    
    cMax = 0
    dMin = math.inf
    for i in range(k):
        random.shuffle(matchedPoints)
        tmpTransX = 0
        tmpTransY = 0
        for j in range(n):
            tmpTransX += fPoints1[matchedPoints[j][0]][0] - fPoints2[matchedPoints[j][1]][0] 
            tmpTransY += fPoints1[matchedPoints[j][0]][1] - fPoints2[matchedPoints[j][1]][1]

        tmpTransX = int(tmpTransX / n)
        tmpTransY = int(tmpTransY / n)

        # check inliers
        c = 0
        dist = 0.
        for j in range(n, len(matchedPoints)):
            d = math.sqrt( ( (np.array(fPoints2[matchedPoints[j][1]]) + np.array([tmpTransX, tmpTransY]) - np.array(fPoints1[matchedPoints[j][0]]) )**2).sum())
            if d <= 3:
                dist += d
                c += 1
        
        if c > cMax:
            transX = tmpTransX
            transY = tmpTransY
            cMax = c
            dMin = dist
        elif c == cMax and dist < dMin:
            transX = tmpTransX
            transY = tmpTransY
            dMin = dist

    print(transX, transY, cMax)
        
    return (transX, transY)

def readPanoCSV():

    fileTable = []
    # csv format: filename(str),focal(float)
    with open('test/parrington/pano.csv') as csvfile:
        rows = csv.reader(csvfile)
        flag = False
        for row in rows:
            if flag:
                row[1] = float(row[1])
                fileTable.append(row)
            else:
                flag = True
    return fileTable


if __name__ == '__main__':


    fileTable = readPanoCSV() 
    
    images = {}
    fPoints = {}
    descriptors = {}
#    kdtrees = {}
    for i in range(nImages):
        images[i] = cv2.imread(os.path.join(testpath, fileTable[i][0]))
        sys.stdout.write("\r ({}/{}) Projecting to cylinder...".format(i+1, nImages))
        sys.stdout.flush()
        images[i] = cylindricalProjection(images[i], fileTable[i][1], showResult=False)
        
        sys.stdout.write("\r ({}/{}) Harris corner detecting...".format(i+1, nImages))
        sys.stdout.flush()
        fPoints[i], descriptors[i] = harrisCornerDetection(images[i], showResult=False)
#        kdtrees[i] = kdtree.cKDTree(descriptors[i])
    sys.stdout.write("\n")
    sys.stdout.flush()

    matchedPoints = {}
    trans = {}
    for i in range(nImages-1):

        sys.stdout.write("\r ({}/{}) Featuring matching...".format(i+1, nImages))
        sys.stdout.flush()
        matchedPoints[i] = featureMatching(descriptors[i], descriptors[i+1])


        sys.stdout.write("\r ({}/{}) Running RANSAC...".format(i+1, nImages))
        sys.stdout.flush()
        printFeatureMatchPoints(matchedPoints[i], images[i], images[i+1], fPoints[i], fPoints[i+1])
        trans[i] = ransac(matchedPoints[i], fPoints[i], fPoints[i+1])

    sys.stdout.write("\n")
    sys.stdout.flush()
    
    result = imageStitch(trans[0], images[0], images[1])
    for i in range(1, nImages-1):
        result = imageStitch(trans[i], result, images[i+1])




