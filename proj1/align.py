#! /usr/bin/env python

import os
import math
import cv2
import numpy as np

sampleInputPath =  "./ldr_sample"
sampleOutputPath = "./ldr_aligned"

def displace(img, d):
    [x,y] = d
    h = np.size(img, 0)
    w = np.size(img, 1)
    if x < 0:
        dImg = img[:, -x:w]
    elif x > 0:
        dImg = img[:, 0:w-x]
    else:
        dImg = img
    if y < 0:
        dImg = dImg[-y:h, :]
    elif y > 0:
        dImg = dImg[0:h-y, :]
    else:
        pass
    return dImg


class MTBImg():
    def __init__(self, rawImg):
        self.origin = rawImg
        self.grayscale = cv2.cvtColor(rawImg, cv2.COLOR_BGR2GRAY)
        self.thresImg = None
        self.excludeMap = None
        self.d = np.array([0, 0])      # displacement (x, y)

    def medianThresh(self, level, d, tol):
        # scale image to current level
        scalar = 1/(2**level)
        lvlImg = cv2.resize(self.grayscale, None, fx=scalar, fy=scalar, interpolation=cv2.INTER_AREA)

        # displace
        lvlImg = displace(lvlImg, d)

        # threshold the image
        median = np.median(lvlImg)
        ret, self.thresImg = cv2.threshold(lvlImg, median, 255, cv2.THRESH_BINARY)

        # find exclude bitmap of avoid noise close to median
        ret, thresImgHi = cv2.threshold(lvlImg, median+tol, 255, cv2.THRESH_BINARY)
        ret, thresImgLo = cv2.threshold(lvlImg, median-tol, 255, cv2.THRESH_BINARY_INV)
        self.excludeMap = cv2.bitwise_or(thresImgHi, thresImgLo)

    def displaceThresh(self, x, y):
        d = np.array([x,y])
        self.curThresh =  displace(self.thresImg, d)
        self.curExclude = displace(self.excludeMap, d)

    def addDisplacement(self, displace):
        self.d = np.add(2*self.d, displace)



class LDRImageSet():
    # setSize: number of images of the same scene
    def __init__(self, setSize=7):
        self.rawSet = []        # A Set of LDR images converts to one HDR image
        self.alignedSet = []    # Aligned Set of LDR images ready to go
        self.fileName = []
        self.setSize = setSize  # Number of images in the Set


    # prints properties of this class
    def __str__(self):
        return "Input images with ({}, {}, {}, {}) (setSize, height, width, channels)"\
            .format(self.setSize, self.imgHeight, self.imgWidth, self.imgChannels)

    # reads a set of image from specified path
    def readImages(self, path):
        for fileName in os.listdir(path):
            self.rawSet.append(cv2.imread(os.path.join(path, fileName)))
            self.fileName.append(fileName)
        self.imgHeight = np.size(self.rawSet[0], 0)     # image height
        self.imgWidth = np.size(self.rawSet[0], 1)      # image width
        self.imgChannels = np.size(self.rawSet[0], 2)   # image channels

    def writeImages(self, path):
        i = 0
        for img in self.alignedSet:
            cv2.imwrite(os.path.join(path, self.fileName[i]), img)
            i += 1


    # pivot:    Index of the image every else is aligned to
    # shiftMax: Maximum search distance while doing alignment
    # tol:      tolerance, avoid noise near median
    def MTBAlign(self, pivot=0, shiftMax=0.05, tol=10):
        # calculate the level to search
        shiftMaxPixel = shiftMax * min(self.imgWidth, self.imgHeight)
        maxlvl = math.ceil(math.log2(shiftMaxPixel))
        print('shiftMaxPixel:', shiftMaxPixel)
        print('maxlvl:', maxlvl)

        # set up data
        MTBSet = []
        for rawImg in self.rawSet:
            mtbImg = MTBImg(rawImg)
            MTBSet.append(mtbImg)

        # find displacement for each image to match the pivot image
        pivImg = MTBSet[pivot]
        for mtbImg in MTBSet:
            # find displacement for each level
            for lvl in reversed(range(maxlvl+1)):
                minDiff = None

                # scale down, displace then threshold
                d = mtbImg.d
                mtbImg.medianThresh(lvl, d, tol)
                pivImg.medianThresh(lvl, -d, tol)

                # find the smallest difference in 9 displacements
                for x in range(-1,2):
                    for y in range(-1,2):
                        mtbImg.displaceThresh(x, y)
                        pivImg.displaceThresh(-x, -y)
                        diffThresh = cv2.bitwise_xor(mtbImg.curThresh, pivImg.curThresh)
                        diffThresh = cv2.bitwise_and(diffThresh, mtbImg.curExclude)
                        diff = diffThresh.sum()
                        if minDiff is None or diff < minDiff:
                            minDiff = diff
                            d = np.array([x, y])    # displacement
                mtbImg.addDisplacement(d)

        # crop images
        # find furthest displacement
        dmax = np.array([0,0])
        dmin = np.array([0,0])
        for mtbImg in MTBSet:
            print('d:', mtbImg.d)
            for i in range(2):
                if mtbImg.d[i] > dmax[i]:
                    dmax[i] = mtbImg.d[i]
                if mtbImg.d[i] < dmin[i]:
                    dmin[i] = mtbImg.d[i]

        print('dmax:', dmax)
        print('dmin:', dmin)

        # displace(crop) the image
        for mtbImg in MTBSet:
            resImg = displace(mtbImg.origin, -dmax+mtbImg.d)
            resImg = displace(resImg, -dmin-mtbImg.d)
            self.alignedSet.append(resImg)




if __name__ == '__main__':
    LDRSet = LDRImageSet()
    LDRSet.readImages(sampleInputPath)
    print(LDRSet)
    LDRSet.MTBAlign()
    LDRSet.writeImages(sampleOutputPath)
