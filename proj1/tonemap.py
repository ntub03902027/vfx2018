#! /usr/bin/env python

import os
import math
import imageio
import cv2
import numpy as np



ldmax = 230.
b = 0.65
Zmin = 0.
Zmax = 255.

#ld = (ldmax * 0.01 / math.log10(np.max(hdr) + 1)) * (np.log(hdr + 1.) / np.log(2. + 8. * (hdr / np.max(hdr))**(math.log(b)/math.log(0.5)) ))

#print(ld)

#ld = (Zmax - Zmin) / (np.max(ld)-np.min(ld)) * ld
#ld = ld.astype(np.uint8)
#print(ld)

#imageio.imwrite('out.png', ld)


def tonemapReinhardSimple(hdr, a=0.6, output='out.png'):

    delta = 0.000001
    lwhite = 1000000.

    lw = 0.2126 * hdr[:,:,0] + 0.7152 * hdr[:,:,1] + 0.0722 * hdr[:,:,2]
    height = hdr.shape[0]
    width = hdr.shape[1]


    lw_avg = math.exp(np.sum(np.log(delta + lw)) / (height * width))

    lum = a / lw_avg * lw
    ld = (lum) * (1. + lum / lwhite**2) / (1. + lum)

    hsv = cv2.cvtColor(hdr, cv2.COLOR_RGB2HSV)

    hsv[:,:,2] = ld
    hdr_mod = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    hdr_mod = np.clip(hdr_mod * 255., 0, 255).astype(np.uint8)


    imageio.imwrite(output, hdr_mod)

    return

def tonemapReinhardLocal(hdr, a=0.6, phi=15., output='out.png'):
    # aka. dodging and burning

    kmax = 8
    alpha1 = 0.35
    alpha2 = 0.35 * 1.6
    epsilon = 10e-6

    lw = 0.2126 * hdr[:,:,0] + 0.7152 * hdr[:,:,1] + 0.0722 * hdr[:,:,2]
    height = hdr.shape[0]
    width = hdr.shape[1]

    r1 = np.zeros([height, width, kmax])
    r2 = np.zeros([height, width, kmax])

    hmap = np.tile(np.expand_dims(np.arange(1, height+1), axis=1), [1, width])
    wmap = np.transpose(np.tile(np.expand_dims(np.arange(1, width+1), axis=1), [1, height]))

    for i in range(1, kmax+1):
        s = 1.6 ** i
        r1[:,:,i-1] = np.exp(-(hmap**2 + wmap**2) / (alpha1**2) ) / (math.pi * (alpha1 * s)**2)
        r2[:,:,i-1] = np.exp(-(hmap**2 + wmap**2) / (alpha2**2) ) / (math.pi * (alpha2 * s)**2)


    v1 = np.zeros([height, width, kmax])
    v2 = np.zeros([height, width, kmax])

    for i in range(1, kmax+1):
        v1[:,:,i-1] = cv2.filter2D(lw, -1, r1)
        v2[:,:,i-1] = cv2.filter2D(lw, -1, r2)


    v = np.zeros([height, width, kmax])
    for i in range(1, kmax+1):
        v[:,:,i-1] = (v1[:,:,i-1] - v2[:,:,i-1]) / ((2**phi * a) / i**2 + v1[:,:,i-1] )

    sm = np.zeros([height, width])
    for i in range(height):
        for j in range(width):
            for k in range(kmax):
                if math.fabs(v[i,j,k]) < epsilon:
                    sm[i,j] = v[i,j,k]

    ld = lw / (1. + sm)

    hsv = cv2.cvtColor(hdr, cv2.COLOR_RGB2HSV)

    hsv[:,:,2] = ld
    hdr_mod = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    hdr_mod = np.clip(hdr_mod * 255., 0, 255).astype(np.uint8)
    imageio.imwrite(output, hdr_mod)

if __name__ == '__main__':

    hdr = imageio.imread('memorial/info/memorial.hdr', format='HDR-FI')

    tonemapReinhardSimple(hdr, a=0.6)
    tonemapReinhardLocal(hdr, a=0.6, phi=10)
