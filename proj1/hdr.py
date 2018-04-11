#! /usr/bin/env python

import os
import math
import imageio
import cv2
import numpy as np
import argparse

def createParser():
    parser = argparse.ArgumentParser(description='HDR radiance map generator')
    parser.add_argument('--path', type=str, default='./ldr_aligned', metavar='PATH', help='path to the directory of the aligned images')
    parser.add_argument('-l', type=float, default=10, metavar='l', help='Lambda value for recovered curve smoothness')
    parser.add_argument('--sample-points', type=int, default=64, metavar='N', help='# of points to sample')
    parser.add_argument('--sample-points-h', type=int, default=8, metavar='Nh', help='# of points to sample along the hight axis when using uniform sample')
    parser.add_argument('--output', type=str, default='out.hdr', metavar='PATH', help='output map path')
    parser.add_argument('--nofilter', action='store_true', default=False, help='do not use (0, 0, 255) to filter empty spaces due to alignment')
    parser.add_argument('--plot-curve', action='store_true', default=False, help='plot recovered curve with matplotlib')
    return parser


parser = createParser()
args = parser.parse_args()

inputPath = args.path
samplePoints = args.sample_points
samplePointsh = args.sample_points_h
outputPath = args.output

zMin = 0-0.001
zMax = 255+0.001

def w_func(z):
    if z <= (zMax + zMin)/2:
        return (z - zMin)
    return (zMax - z)



class InputImages():
    def __init__(self):
        self.img = []
        self.logShutterTime = []
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
            self.img.append(imageio.imread(os.path.join(path, filename)))
            self.logShutterTime.append(filename2speed[filename])

        self.nImages = len(self.img) # in example "memorial": 16 images
        self.height = np.size(self.img[0], 0) # in example "memorial": 768
        self.width = np.size(self.img[0], 1) # in example "memorial": 512
        self.channels = np.size(self.img[0], 2) # in example "memorial": 3 (RGB)

    def isAlignedRegion(self, h, w, p):
        if self.img[p][h,w,0] == 0 and self.img[p][h,w,1] == 0 and self.img[p][h,w,2] == 255:
            return True
        return False

    def alignMask(self, p, nofilter=False):
        if nofilter:
            mask = np.ones([self.height, self.width, self.channels]).astype(np.float32)
            return mask
        mask = np.all(self.img[p] != (0, 0, 255), axis=-1).astype(np.float32)
        mask = np.tile(np.expand_dims(mask, axis=2), [1,1,3])
        return mask

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
            self.logSSMat[p] = rawImages.logShutterTime[p]
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
                    wijc = w_func(self.zMat[i, j, chan])
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
                A[k, i, chan] = self.lam * w_func(i+1)
                A[k, i+1, chan] = -2 * self.lam * w_func(i+1)
                A[k, i+2, chan] = self.lam * w_func(i+1)
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

    def arrangeHDRImage(self, rawImages, nofilter=False):
        self.hdrImage = np.zeros([rawImages.height, rawImages.width, rawImages.channels], dtype=np.float32)
        denoMat = np.zeros([rawImages.height, rawImages.width, rawImages.channels], dtype=np.float32)
        numeMat = np.zeros([rawImages.height, rawImages.width, rawImages.channels], dtype=np.float32)

        w_vec = np.vectorize(w_func)

        for p in range(rawImages.nImages):
            alignMask = rawImages.alignMask(p, nofilter=nofilter)
            denoMat = denoMat + alignMask * w_vec(rawImages.img[p][:,:,:])

            takeMat = np.zeros([rawImages.height, rawImages.width, rawImages.channels], dtype=np.float32)
            takeMat[:,:,0] = np.take(self.xr[0:256,0], rawImages.img[p][:,:,0])
            takeMat[:,:,1] = np.take(self.xg[0:256,0], rawImages.img[p][:,:,1])
            takeMat[:,:,2] = np.take(self.xb[0:256,0], rawImages.img[p][:,:,2])


            numeMat = numeMat + alignMask * w_vec(rawImages.img[p][:,:,:]) * (takeMat - self.logSSMat[p] * np.ones([rawImages.height, rawImages.width, rawImages.channels], dtype=np.float32))

        self.hdrImage = numeMat / denoMat
        self.hdrImage = np.exp(self.hdrImage)


        self.hdrImage = self.hdrImage.astype(np.float32)
        print(self.hdrImage)

    def outputHDR(self, path='out.hdr'):
        if self.hdrImage is None:
            return
        imageio.imwrite(path, self.hdrImage, format='HDR-FI')


if __name__ == '__main__':
    raw = InputImages()
    raw.readImages(inputPath)
    print(raw)
    hdr = DebevecHDR(raw.nImages, lam=args.l)

    hdr.sampleUniformly(raw)
    hdr.solveDevebec()
    if args.plot_curve:
        hdr.plotCurve()
    hdr.arrangeHDRImage(raw, nofilter=args.nofilter)
    hdr.outputHDR(outputPath)
