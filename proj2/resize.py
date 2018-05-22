#! /usr/bin/env python3
import argparse
import os
import csv
import cv2
import sys 

def createParser():
    parser = argparse.ArgumentParser(description='Resize input images')
    parser.add_argument('--path', type=str, default='.', metavar='PATH', help='path to the directory where pano.csv locates')
    parser.add_argument('--scale', type=float, default='1.', metavar='K', help='scale factor (1/K)')
    parser.add_argument('--out', type=str, default='.', metavar='PATH', help='path to output images')
    return parser

def readPanoCSV(path):

    fileList = []
    with open(os.path.join(path, 'pano.csv')) as csvfile:
        rows = csv.reader(csvfile)
        flag = False
        for row in rows:
            if flag:
                fileList.append(row[0])
            else:
                flag = True 

    return fileList 

if __name__ == '__main__':

    parser = createParser()
    args = parser.parse_args()

    
    fileList = readPanoCSV(args.path)
    if not os.path.isdir(args.out):
        os.makedirs(args.out)
    
    h = 0
    w = 0
    for filename in fileList:
        sys.stdout.write('processing: {}\n'.format(filename))
        sys.stdout.flush()

        img = cv2.imread(os.path.join(args.path, filename))

        if h == 0:
            h = int(img.shape[0] / args.scale)
            w = int(img.shape[1] / args.scale)

        res = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        
        cv2.imwrite(os.path.join(args.out, filename), res)
    

    sys.stdout.write('all images of size ({}, {}) sucessfully saved to {}\n'.format(h, w, os.path.abspath(args.out)))
    sys.stdout.flush()
