#! /usr/bin/env python
import argparse
import os
import csv
import numpy as np
import cv2
import scipy.spatial.ckdtree as KDTree
import sys
import math
import random
import copy


############################################
#                                          #
# Utility Functions                        #
#                                          #
############################################

def createParser():
    parser = argparse.ArgumentParser(description='Assemble separate images to a panorama')
    parser.add_argument('--path', type=str, default='test/parrington', metavar='PATH', help='path to the directory in which the images to be stitched, as well as a pano.csv are inside (default="test/parrington")')
    parser.add_argument('-o', type=str, default='', metavar='PATH', help='path to output image; if not specified, a "out.jpg" will be outputted in the directory specified in "--path".')
    parser.add_argument('--poisson', action='store_true', default=False, help='perform Poisson blending instead of linear blending, overrides "--constant"')
    parser.add_argument('--constant', type=int, default=-1, metavar='c', help='perform constant-value linear blending with given constant value, assigning a negative number will have it perform normal linear blending (defaultL -1)')
    parser.add_argument('--show-warping', action='store_true', default=False, help='show results right after cylindrical warping')
    parser.add_argument('--show-feature', action='store_true', default=False, help='show feature on images after feature detection')
    parser.add_argument('--show-matching', action='store_true', default=False, help='show matching features after feture matching')
    parser.add_argument('--show-stitch', action='store_true', default=False, help='show intermediate results of image stitching')

    return parser

def getImageFilename(i, prefix='', suffix='', digit=0):
    if digit == 0:
        return prefix + str(i) + suffix
    
    n = digit - len(str(i))
    if n < 0:
        n = 0
    return prefix + n * '0' + str(i) + suffix

def isBlank(image, i, horizontal=False):
    """
    Check if a column/row of an image is all-zero.
    
    Parameters
    __________
    image : cv2 image
        input image

    i : int
        row or column index

    horizontal : bool
        if true, then i will be regarded as index of dimension 0

    Returns
    -------
    cImage : bool

    Implementation
    -------------
    1. brute-force
    """

    flag = True
    if horizontal:
        return np.alltrue(image[i,:,:] == 0)
    else:
        return np.alltrue(image[:,i,:] == 0)

def readPanoCSV(path):

    fileTable = []
    # csv format: filename(str),focal(float)
    with open(os.path.join(path, 'pano.csv')) as csvfile:
        rows = csv.reader(csvfile)
        flag = False
        for row in rows:
            if flag:
                row[1] = float(row[1])
                fileTable.append(row)
            else:
                flag = True
    return fileTable

def saveImage(result, path):
    cv2.imwrite(path, result)
############################################
#                                          #
# Parsing arguments                        #
#                                          #
############################################

parser = createParser()
args = parser.parse_args()
    

############################################
#                                          #
# 1. cylindrical projection                #
# 2. feature detection (Harris)            #
# 3. feature matching                      #
# 4. image matching (RANSAC)               #
# 5. image stitching (contains blending)   #
#                                          #
############################################
    
def cylindricalProjection(image, f, showResult=False):
    """
    Perform cylindrical warping given focal lengths.

    Parameters
    __________
    image : cv2 image
        input image

    f : float
        focal point

    Returns
    -------
    cImage : ndarray of dtype=np.uint8
        warped image

    Implementation
    -------------
    1. define center points as (0, 0)
    2. fill the empty cImage with formula of warping
    3. crop blankings
    4. (optional) print results
    """
    # 1. define center points
    xCen = int(image.shape[1] / 2)
    yCen = int(image.shape[0] / 2)
    cImage = np.zeros(image.shape, dtype=np.uint8)
    
    # 2. fill warped cImage
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
                cImage[yc,xc,:] = image[j,i,:]


    # 3. crop left/right/up/down blankings
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
    
    # 4. show results
    if showResult:
        cv2.imshow('image', cImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
            
    
    return cImage


"""
    the descriptor vector will be (2 * size - 1)**2
"""
def harrisDescriptor(image, fPoint, size, useVerticalCoord=False):
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
        if useVerticalCoord:
            vec.append(point[0])

        des.append(vec)

    return des



def harrisCornerDetection(image, showResult=False):
    """
    Perform Harris Corner Detection.

    Parameters
    __________
    image : cv2 image
        input image

    showResult : bool
        show features with cv2 viewer

    Returns
    -------
    fPoint : list of 2-tuples
        feature points
    des : 2-d list
        descriptors w.r.t. fPoint

    Implementation
    -------------
    0. declare constants
    1. Harris corner detection algorithm
    1a. perform Sobel operators
    1b. products of derivatives
    1c. pass with Gaussian blur kernels
    1d. derive response map R
    1e. find the highest n percents of points as candidate
    2. remove redundant features
    2a. remove edge features
    2b. remove features due to warping
    2c. max pooling
    3. generate descriptors
    4. (optional) print results
    """
    # 0. define const.
    # k: empricially 0.04 - 0.06
    k = 0.05
    bestPercent = 10
    distortBufferValue = 5
    
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   
    # 1a. x, y derivatives
    # Sobel operators
    I_x = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, ksize=-1)
    I_y = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, ksize=-1)

    # 1b. products of derivatives
    I_xx = I_x * I_x
    I_yy = I_y * I_y
    I_xy = I_x * I_y

    # 1c. sums of products at each pixel
    S_xx = cv2.GaussianBlur(I_xx, (5, 5), 1.)
    S_yy = cv2.GaussianBlur(I_yy, (5, 5), 1.)
    S_xy = cv2.GaussianBlur(I_xy, (5, 5), 1.)


    # 1d. response map R

    R = np.zeros(S_xx.shape)
    """
    # Intuitive codes:
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            Mij = np.array([[S_xx[i,j], S_xy[i,j]], [S_xy[i,j], S_yy[i,j]]])
            R[i,j] = np.linalg.det(Mij) - k * (np.trace(Mij) ** 2)
    """
    # (1d.) faster codes:
    R = (S_xx * S_yy) - (S_xy * S_xy) - k * (S_xx + S_yy) * (S_xx + S_yy)


    # 1e.
    # find the N largest points as features

    N = int(R.shape[0] * R.shape[1] * 0.01 * bestPercent) # highest i%
    
    arr = abs(R.reshape([R.shape[0] * R.shape[1]]))
    largest = arr.argsort()[-N:][::-1] # argsort: sort corresponding index from lowest to largest; [-N:]: the last N (i.e. largest) elements; [::-1]: reverse ordering
    x = np.floor_divide(largest, R.shape[1])
    y = largest % R.shape[1]
    fPoint = list(zip(x.tolist(), y.tolist()))

    # 2.
    # 2a. remove edge features
    # 2b. remove features due to cylindrical projection
    maxDistort = 0 # find the largest index that exists blank pixels due to warping
    while np.alltrue(image[maxDistort,0,:] == 0): 
        maxDistort += 1
    
    tmp = []
    for p in fPoint:
        if p[0] < maxDistort + distortBufferValue or p[1] < 2 or p[0] >= R.shape[0] - maxDistort - distortBufferValue - 1 or p[1] >= R.shape[1] - 2:
            tmp.append(p)
    for p in tmp:
        fPoint.remove(p)
    del tmp


    # 2c. 2*2 max pooling
    maxPool = set({})
    for i in range(2,R.shape[0]-1, 2):
        for j in range(2,R.shape[1]-1, 2):
            tmp = [(R[i,j], (i,j)), (R[i+1,j], (i+1,j)), (R[i,j+1], (i,j+1)), (R[i+1, j+1], (i+1, j+1))]
            maxPool.add(max(tmp)[1])
    
    # 4. show results
    if showResult:
        fPointOld = fPoint
    fPoint = list(maxPool.intersection(set(fPoint)))

    # 3. find descriptors
    des = harrisDescriptor(grayscale, fPoint, 4, useVerticalCoord=True)
    

    # 4. show results
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

        cv2.imshow('image', fImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    
    return fPoint, des

def featureMatching(des1, des2, k=4, printAll=False):
    """
    Perform feature matching using kd-trees.

    Parameters
    __________
    des1 : 2d list
        descriptors of 1st image

    des2 : 2d list
        descriptors of 2nd image
    k : int
        k near neighbors queried by kd-tree

    Returns
    -------
    matchedPoints : list of 2-tuples  ("[(ind11, ind12), (ind21, ind22), ...]")
        matched points

    Implementation
    -------------
    1. build kd tree for des2
    2. query distances and indices of k nearest neighbors for every element of des1 
    3. see if the match meets the criteria
    4. (optional) print sorted list
    """
    # 1. kd tree 
    kdtree2 = KDTree.cKDTree(des2)

    # 2-3.
    matchedPoints = []
    # n = index of des1 = index of fPoint1
    # d = k nearest distance
    # i = indices of k nearest distance = indices for des2
    for n in range(len(des1)):
        d, i = kdtree2.query(des1[n], k=k)
        if d[0] < 256. and d[0] <= 0.8 * d[1]:
            matchedPoints.append( (n, i[0]) )
    
    # 4. 
    if printAll:
        nn = []
        for n in range(len(des1)):
            d, i = kdtree2.query(des1[n], k=k)
            nn.append((n, d, i))
        
        nn = sorted(nn, key=lambda a: a[1][0])
    
        for x in nn:
            print(x)
    
    # return a list of matched points of format 2-tuple: ([ind of 1st fPoint], [ind of 2nd fPoint])
    return matchedPoints

   
def printFeatureMatchPoints(matchedPoints, image1, image2, fPoint1, fPoint2):

    a = np.zeros([max(image1.shape[0], image2.shape[0]), image1.shape[1] + image2.shape[1], image1.shape[2]], dtype=np.uint8)
    grayscale1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    grayscale2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    a[0:image1.shape[0],0:image1.shape[1], :] = np.tile(np.expand_dims(grayscale1, axis=2), [1,1,3])
    a[0:image2.shape[0],image1.shape[1]:image1.shape[1]+image2.shape[1], :] = np.tile(np.expand_dims(grayscale2, axis=2), [1,1,3])
   
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

def ransac(matchedPoints, fPoints1, fPoints2, n=2, p=0.6, P=0.99, kMin=128):
    """
    Perform image matching using RANSAC.

    Parameters
    __________
    matchedPoints : list of 2d tuples
        the index tuples that denotes the matches between some points in fPoints1 and fPoints2

    fPoints1 : list of 2d tuples
        feature points of image 1
    
    fPoints2: list of 2d tuples
        feature points of image 2

    n : int
        # of points for sampling in every RANSAC iteration
    
    p : float 
        prob. of inliers (must be between 0 and 1)

    P : float
        prob. that the estimated model is correct 

    kMin: int
        minimum numbers of iteration 
    
    Returns
    -------
    (transX, transY) : 2-tuple 
        denotes that how image 2 should translate to fit with image 1

    Implementation
    -------------
    1. compute iterations needed
    2. iterate k times
    2a. sample n pairs
    2b. check inliers
    2c. update optimal results
    """

    # 1. compute k
    k = max(math.ceil(math.log(1-P)/math.log(1-p**n)), kMin)

    # 2. iterate
    transX = 0
    transY = 0
    
    cMax = 0
    dMin = math.inf
    print(len(matchedPoints))
    for i in range(k):
        # 2a. sample
        random.shuffle(matchedPoints)
        tmpTransX = 0
        tmpTransY = 0
        for j in range(n):
            tmpTransX += fPoints1[matchedPoints[j][0]][0] - fPoints2[matchedPoints[j][1]][0] 
            tmpTransY += fPoints1[matchedPoints[j][0]][1] - fPoints2[matchedPoints[j][1]][1]

        tmpTransX = int(tmpTransX / n)
        tmpTransY = int(tmpTransY / n)

        # 2b. check inliers
        c = 0
        dist = 0.
        for j in range(n, len(matchedPoints)):
            d = math.sqrt( ( (np.array(fPoints2[matchedPoints[j][1]]) + np.array([tmpTransX, tmpTransY]) - np.array(fPoints1[matchedPoints[j][0]]) )**2).sum())
            if d <= 3:
                dist += d
                c += 1
        # 2c. update
        if c > cMax:
            transX = tmpTransX
            transY = tmpTransY
            cMax = c
            dMin = dist
        elif c == cMax and dist < dMin:
            transX = tmpTransX
            transY = tmpTransY
            dMin = dist

    sys.stdout.write("({}, {}, {})\n".format(transX, transY, cMax))
    sys.stdout.flush()
        
    return (transX, transY)



def imageStitch(trans, image1, image2, showResult=False, poisson=False, constant=-1):

    """
    Perform image stitching with blending.

    Parameters
    __________
    trans : 2-tuple
        denotes x-y translation

    image1 : ndarray
        input image 1
    
    image2: ndarray
        input image 2

    showResult : bool
        show stitched result right after the process
    
    poisson : bool 
        perform Poisson blending

    constant : int 
        width of constant-width blending  

    
    Returns
    -------
    result : ndarray
        the result stitched image 
    
    (startX2, startY2): 2-tuple 
        the displacement of the 2nd image that will be used in the next stitching iteration 

    Implementation
    -------------
    1. compute starting points 
    2. perform stitching and blending
    2a. poisson 
    2b. linear 
    2c. linear with constant width
    """
    # 1. starting points
    startX1 = abs(min(trans[0], 0))
    startY1 = abs(min(trans[1], 0))

    startX2 = abs(max(trans[0], 0))
    startY2 = abs(max(trans[1], 0))

    result = np.zeros([max(image1.shape[0] + startX1, image2.shape[0] + startX2), max(image1.shape[1] + startY1, image2.shape[1] + startY2), image1.shape[2]], dtype=np.uint8)

    # 2a. poisson
    if poisson:

        mask = np.zeros([result.shape[0], result.shape[1]])
        background = np.zeros(result.shape)
        target = np.zeros(result.shape)
        
        background[startX2:startX2+image2.shape[0],startY2:startY2+image2.shape[1],:] = image2[:,:,:]
        mask1 = np.tile( np.expand_dims(((cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)) == 0).astype(int), axis=2) , (1,1,3))
        background[startX1:startX1+image1.shape[0],startY1:startY1+image1.shape[1],:] *= mask1
        background[startX1:startX1+image1.shape[0],startY1:startY1+image1.shape[1],:] += image1[:,:,:]
        

        target[startX2:startX2+image2.shape[0],startY2:startY2+image2.shape[1],:] = image2[:,:,:]
        

        thr = 2
        mask[startX2-int(trans[0]>0)*min(thr, startX2):startX2+image2.shape[0]+int(trans[0]<0)*min(thr, mask.shape[0]-startX2-image2.shape[0]),startY2-int(trans[1]>0)*min(thr, startY2):startY2+image2.shape[1]+int(trans[1]<0)*min(thr, mask.shape[1]-startY2-image2.shape[1])] = 1
        
        
        grayB = cv2.cvtColor(background.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        grayT = cv2.cvtColor(target.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        # make exact black area masked
        mask = mask * ((grayB > 0).astype(int) + (grayT > 0).astype(int) > 0).astype(int)
        mask = np.tile(np.expand_dims(mask, axis=2), (1,1,3) )


        tBoundMask = (grayT > 0).astype(int)
        tBoundMask = np.tile(np.expand_dims(tBoundMask, axis=2), (1,1,3) )

        bBoundMask = ((grayB > 0).astype(int) + (grayT > 0).astype(int) > 0).astype(int)
        #bBoundMask = (grayB > 0).astype(int)
        bBoundMask = np.tile(np.expand_dims(bBoundMask, axis=2), (1,1,3) )

        
        # fill fixed matrices: coef, coefInv, b0_w, b0_s, b0_e, b0_n, t0_n, t0_w, t0_s, t0_e, t0_n 
        coef = np.zeros(result.shape)
        coef[:,1:,:] += tBoundMask[:,:-1,:]
        coef[:,:-1,:] += tBoundMask[:,1:,:]
        coef[1:,:,:] += tBoundMask[:-1,:,:]
        coef[:-1,:,:] += tBoundMask[1:,:,:]
        coef = coef * tBoundMask

        coefB = np.zeros(result.shape)
        coefB[:,1:,:] += bBoundMask[:,:-1,:]
        coefB[:,:-1,:] += bBoundMask[:,1:,:]
        coefB[1:,:,:] += bBoundMask[:-1,:,:]
        coefB[:-1,:,:] += bBoundMask[1:,:,:]
        coefB = coefB * bBoundMask
        coefB += (coefB == 0).astype(np.uint8)
        coefInv = 1/coefB


        #coef += 2
        #coef[1:-1,:,:] += 1
        #coef[:,1:-1,:] += 1
        #coefInv = 1/coef

        # initial background (shifted, masked)
        b0_all = np.zeros(result.shape)
        b0_all[:,1:,:] += (1. - mask[:,:-1,:])* background[:,:-1,:]
        b0_all[:-1,:,:] += (1. - mask[1:,:,:])* background[1:,:,:]
        b0_all[:,:-1,:] += (1. - mask[:,1:,:])* background[:,1:,:]
        b0_all[1:,:,:] += (1. - mask[:-1,:,:])* background[:-1,:,:]
        

        # target (shifted)
        t_all = np.zeros(result.shape)
        t_all[:,1:,:] += target[:,:-1,:]
        t_all[:-1,:,:] += target[1:,:,:]
        t_all[:,:-1,:] += target[:,1:,:]
        t_all[1:,:,:] += target[:-1,:,:]
        t_all *= tBoundMask
        

        bi = np.zeros(result.shape)
        bi[:,:,:] = background
        

        iteration = 200
        for i in range(iteration):
            sys.stdout.write('\r ({}/{}) perform Poisson blending on image...'.format(i+1, iteration))


            b_w = np.zeros(result.shape)
            b_w[:,1:,:] = mask[:,:-1,:] * bi[:,:-1,:]
            b_s = np.zeros(result.shape)
            b_s[:-1,:,:] = mask[1:,:,:] * bi[1:,:,:]
            b_e = np.zeros(result.shape)
            b_e[:,:-1,:] = mask[:,1:,:] * bi[:,1:,:]
            b_n = np.zeros(result.shape)
            b_n[1:,:,:] = mask[:-1,:,:] * bi[:-1,:,:]
            
            bi = mask * coefInv * (coef * target - (t_all) + (b0_all) + (b_n + b_s + b_e + b_w) ) + (1-mask) * background
            bi = np.clip(bi, 0, 255)
            

        bi = bi.astype(np.uint8)
        
        if showResult:
            cv2.imshow('image', bi)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


        return bi, (startX2, startY2)

    
    # 2b-c. two linear blendings
    result[startX1:startX1+image1.shape[0],startY1:startY1+image1.shape[1],:] = image1[:,:,:]

    yOverlap = image1.shape[1] + image2.shape[1] - result.shape[1]
    yMid = int((yOverlap + abs(trans[1]))/2)
    threshold = constant
    for i in range(image2.shape[0]):
        for j in range(image2.shape[1]):
            if result[i+startX2,j+startY2,0] == 0 and result[i+startX2,j+startY2,1] == 0 and result[i+startX2,j+startY2,2] == 0:
                result[i+startX2,j+startY2,:] = image2[i,j,:]
            else:
                # blending
                if threshold >= 0:
                    if np.sign(trans[1]) * (j+startY2 - yMid) >= threshold:
                        coef = 1.
                    elif np.sign(trans[1]) * (j+startY2 - yMid) <= -threshold:
                        coef = 0. 
                    else:
                        coef = float(trans[1] > 0) + (float(trans[1] <= 0) - float(trans[1] > 0))* ((j+startY2) - (yMid-threshold)) / (2* threshold)
 

                else:
                    coef = float(trans[1] > 0) + (float(trans[1] <= 0) - float(trans[1] > 0))* ((j+startY2) - abs(trans[1])) / yOverlap 
                result[i+startX2,j+startY2,:] = (coef * image2[i,j,:] + (1. - coef) * result[i+startX2,j+startY2,:]).astype(np.uint8)

    if showResult:
        cv2.imshow('image', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    
    return result, (startX2, startY2)





if __name__ == '__main__':


    fileTable = readPanoCSV(args.path) 
    nImages = len(fileTable)
   
    # cylinder, harris
    images = {}
    fPoints = {}
    descriptors = {}
    for i in range(nImages):
        images[i] = cv2.imread(os.path.join(args.path, fileTable[i][0]))
        sys.stdout.write("\r ({}/{}) Projecting to cylinder...".format(i+1, nImages))
        sys.stdout.flush()
        images[i] = cylindricalProjection(images[i], fileTable[i][1], showResult=args.show_warping)
        
        sys.stdout.write("\r ({}/{}) Harris corner detecting...".format(i+1, nImages))
        sys.stdout.flush()
        fPoints[i], descriptors[i] = harrisCornerDetection(images[i], showResult=args.show_feature)
    sys.stdout.write("\n")
    sys.stdout.flush()

    # feature matching, ransac
    matchedPoints = {}
    trans = {}
    printMatchResult = args.show_matching
    for i in range(nImages-1):

        sys.stdout.write("\r ({}/{}) Featuring matching...".format(i+1, nImages - 1))
        sys.stdout.flush()
        matchedPoints[i] = featureMatching(descriptors[i], descriptors[i+1])


        sys.stdout.write("\r ({}/{}) Running RANSAC...".format(i+1, nImages - 1))
        sys.stdout.flush()
        if printMatchResult:
            printFeatureMatchPoints(matchedPoints[i], images[i], images[i+1], fPoints[i], fPoints[i+1])
        trans[i] = ransac(matchedPoints[i], fPoints[i], fPoints[i+1])

    sys.stdout.write("\n")
    sys.stdout.write("\r ({}/{}) Image Stitching...".format(1, nImages-1))
    sys.stdout.flush()
   
    # stitch
    result, shift = imageStitch(trans[0], images[0], images[1], showResult=args.show_stitch, poisson=args.poisson, constant=args.constant)
    for i in range(1, nImages-1):

        sys.stdout.write("\r ({}/{}) Image Stitching...".format(i+1, nImages-1))
        sys.stdout.flush()
        result, shift = imageStitch( (trans[i][0] + shift[0], trans[i][1] + shift[1]), result, images[i+1], showResult=args.show_stitch, poisson=args.poisson, constant=args.constant)

    if args.o == '':
        saveImage(result, os.path.join(args.path, 'out.jpg'))
    else:
        saveImage(result, os.path.join(args.path))


