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
                filename2speed[line[0]] = math.log(1.0 /float(line[1]))

        for filename in filename2speed:
            self.img.append(cv2.imread(os.path.join(path, filename)))
            self.logShutterSpeed.append(filename2speed[filename])

        self.nImages = len(self.img) # in example "memorial": 16 images
        self.height = np.size(self.img[0], 0) # in example "memorial": 768
        self.width = np.size(self.img[0], 1) # in example "memorial": 512
        self.channels = np.size(self.img[0], 2) # in example "memorial": 3 (RGB)


class DebevecHDR():
    def __init__(self, P, lam=10):
        self.P = P
        self.N = samplePoints
        self.Nh = samplePointsh
        self.lam = lam

        self.hdrImage = None
        self.zMat = np.zeros([self.N, P, 3], dtype=np.uint8) # N sample points x P images x (R, G, B) 3 channels (where self.zMat[;,;,0] represents R channel and so on)
        self.coord = []
        self.logSSMat = np.zeros(P)

        # predeclare result matrix
        self.xr = None
        self.xg = None
        self.xb = None


    def sampleUniformly(self, rawImages):


        # fill self.coord first
        for nh in range(self.Nh):
            for nw in range(int(self.N/self.Nh)):
                self.coord.append(( int((2*nh+1)/(2*self.Nh) * rawImages.height), int((2*nw+1)/(2*self.N/self.Nh) * rawImages.width )))

        for p in range(self.P):
            self.logSSMat[p] = rawImages.logShutterSpeed[p]
            for n in range(self.N):
                self.zMat[n,p,0] = rawImages.img[p][self.coord[n][0],self.coord[n][1],0]
                self.zMat[n,p,1] = rawImages.img[p][self.coord[n][0],self.coord[n][1],1]
                self.zMat[n,p,2] = rawImages.img[p][self.coord[n][0],self.coord[n][1],2]


    def solveDevebec(self):
        n = 256
        A = np.zeros([self.N * self.P + n - 1, n + self.N, 3])
        b = np.zeros([self.N * self.P + n - 1, 1, 3])

        k = 0
        for i in range(self.N):
            for j in range(self.P):
                for chan in range(3):
                    wijc = self.w(self.zMat[i, j, chan])
                    A[k,self.zMat[i, j, chan], chan ] = wijc
                    A[k, n+i, chan] = -wijc
                    b[k, 0, chan] = wijc * self.logSSMat[j]
                k += 1

        A[k, 128, 0] = 1
        A[k, 128, 1] = 1
        A[k, 128, 2] = 1
        k += 1

        for i in range(n-2):
            for chan in range(3):
                A[k, i, chan] = self.lam * self.w(i+1)
                A[k, i+1, chan] = -2 * self.lam * self.w(i+1)
                A[k, i+2, chan] = self.lam * self.w(i+1)
            k += 1

        self.xr, _, _, _ = np.linalg.lstsq(A[:,:,0], b[:,:,0], rcond=None)
        self.xg, _, _, _ = np.linalg.lstsq(A[:,:,1], b[:,:,1], rcond=None)
        self.xb, _, _, _ = np.linalg.lstsq(A[:,:,2], b[:,:,2], rcond=None)


    def plotCurve(self):
        if self.xr is None or self.xg is None or self.xb is None:
            return
        import matplotlib.pyplot as plt
        n = 256
        z = np.linspace(0, n-1, n)
        yr = np.reshape(self.xr, (n + self.N))[:n]
        yg = np.reshape(self.xg, (n + self.N))[:n]
        yb = np.reshape(self.xb, (n + self.N))[:n]
        plt.plot(yr, z, label='red', color='r')
        plt.plot(yg, z, label='green', color='g')
        plt.plot(yb, z, label='blue', color='b')
        plt.title('Recovered response functions')
        plt.xlabel('log exposure X')
        plt.ylabel('pixel value Z')
        plt.legend()
        plt.show()
        del plt

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
    hdr.plotCurve()
