#! /usr/bin/env python

import os
import math
import cv2
import numpy as np

examplePath = "./memorial"
samplePoints = 64
samplePointsh = 8
zMin = 0
zMax = 255

class InputImages():
    def __init__(self):
        self.img = []
        self.logShutterSpeed = []
        self.nImages = 0
        self.height = 0
        self.width = 0
        self.channels = 0

    def __str__(self):
        return "Input images with ({}, {}, {}, {}) (images, height, width, channels)".format(self.nImages, self.height, self.width, self.channels)

    def readImages(self, path, imgList='image_list.txt'):
        """
        readImages: read images and its shutter speeds
        """
        filename2speed = {}
        with open(os.path.join(path, imgList)) as listFile:
            lines = [line for line in listFile]
            for line in lines:
                line = line.split(' ')
                filename2speed[line[0]] = math.log(float(line[1]))

        for filename in filename2speed:
            self.img.append(cv2.imread(os.path.join(path, filename)))
            self.logShutterSpeed.append(filename2speed[filename])

        self.nImages = len(self.img) # in example "memorial": 16 images
        self.height = np.size(self.img[0], 0) # in example "memorial": 768
        self.width = np.size(self.img[0], 1) # in example "memorial": 512
        self.channels = np.size(self.img[0], 2) # in example "memorial": 3 (RGB)


class DebevecHDR():
    def __init__(self, P, lam=0.2):
        self.P = P
        self.N = samplePoints
        self.Nh = samplePointsh
        self.lam = lam

        self.hdrImage = None
        self.zMatR = np.zeros([self.N, P], dtype=np.uint8)
        self.zMatG = np.zeros([self.N, P], dtype=np.uint8)
        self.zMatB = np.zeros([self.N, P], dtype=np.uint8)
        self.coord = []
        self.logSSMat = np.zeros(P)


    def sampleUniformly(self, rawImages):


        # fill self.coord first
        for nh in range(self.Nh):
            for nw in range(int(self.N/self.Nh)):
                self.coord.append(( int((2*nh+1)/(2*self.Nh) * rawImages.height), int((2*nw+1)/(2*self.N/self.Nh) * rawImages.width )))
        print(self.coord)

        for p in range(self.P):
            self.logSSMat[p] = rawImages.logShutterSpeed[p]
            for n in range(self.N):
                self.zMatR[n,p] = rawImages.img[p][self.coord[n][0],self.coord[n][1],0]
                self.zMatG[n,p] = rawImages.img[p][self.coord[n][0],self.coord[n][1],1]
                self.zMatB[n,p] = rawImages.img[p][self.coord[n][0],self.coord[n][1],2]

    def solveDevebec(self):
        n = 256
        A = np.zeros([self.N * self.P + n + 1, n + self.N])
        b = np.zeros([self.N * self.P + n + 1, 1])

        k = 1
        for i in range(self.N):
            for j in range(self.P):
                wij = self.w(self.zMatR[i,j]+1)
                A[k,self.zMatR[i,j]+1] = wij
                A[k,n+i] = -wij
                b[k,0] = wij * self.logSSMat[j]
                k += 1

        A[k, 129] = 1
        k += 1

        for i in range(n-2):
            A[k,i] = self.lam * self.w(i+1)
            A[k,i+1] = -2 * self.lam * self.w(i+1)
            A[k, i+2] = self.lam * self.w(i+1)
            k += 1

        x, _, _, _ = np.linalg.lstsq(A, b)

        import matplotlib.pyplot as plt
        z = np.linspace(0, 255, 256)
        y = np.reshape(x, (320))[:256]
        print(y)
        plt.plot(z, y, label='result')
        plt.legend()
        plt.show()

    def w(self, z):
        if z <= (zMax + zMin)/2:
            return (z - zMin)
        return (zMax - z)


if __name__ == '__main__':
    raw = InputImages()
    raw.readImages(examplePath)
    print(raw)
    hdr = DebevecHDR(raw.nImages)

    hdr.sampleUniformly(raw)
    hdr.solveDevebec()
