#! /usr/bin/env python

import os
import math
import imageio
import cv2
import numpy as np
import argparse


def createParser():
    parser = argparse.ArgumentParser(description='Tone mapper')
    parser.add_argument('--path', type=str, default='out.hdr', metavar='PATH', help='path to the HDR file')
    parser.add_argument('--output', type=str, default='out.png', metavar='PATH', help='output image path')
    parser.add_argument('--method', type=str, default='ReinhardSimple', metavar='M', help='choose methods (available: ReinhardSimple, ReinhardLocal, Drago)')
    parser.add_argument('-b', type=float, default=0.85, metavar='b', help='Bias value for the Drago mathod (default: 0.85)')
    parser.add_argument('-a', type=float, default=0.6, metavar='a', help='"a" value for the Reinhard methods (default: 0.6)')
    parser.add_argument('-phi', type=float, default=15, metavar='phi', help='phi value for the ReinhardLocal methods (default: 15)')
    return parser


def tonemapDrago(hdr, b=0.85, output='out.png'):
    ldmax = 100.
    lwmax = 230.


    lw = 0.2126 * hdr[:,:,0] + 0.7152 * hdr[:,:,1] + 0.0722 * hdr[:,:,2]

    ld = (ldmax * 0.01 / math.log10(lwmax + 1.)) * (np.log(lw + 1.) / np.log(2.+((lw / lwmax)**(math.log(b)/math.log(0.5))) * 8.))

    hsv = cv2.cvtColor(hdr, cv2.COLOR_RGB2HSV)

    hsv[:,:,2] = ld
    hdr_mod = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    hdr_mod = np.clip(hdr_mod * 255., 0, 255).astype(np.uint8)


    imageio.imwrite(output, hdr_mod)


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

    parser = createParser()
    args = parser.parse_args()
    imageio.plugins.freeimage.download()

    hdr = imageio.imread(args.path, format='HDR-FI')

    if args.method == 'ReinhardSimple':
        tonemapReinhardSimple(hdr, a=args.a, output=args.output)
    elif args.method == 'ReinhardLocal':
        tonemapReinhardLocal(hdr, a=args.a, phi=args.phi, output=args.output)
    elif args.method == 'Drago':
        tonemapDrago(hdr, b=args.b, output=args.output)
    else:
        print("Unknown method! Exit.")
