#! /usr/bin/env python

import os
import cv2
import numpy


examplePath = "./memorial"


class InputImages():
    def __init__(self):
        self.img = []
        self.images = 0
        self.height = 0
        self.width = 0
        self.channels = 0

    def readImages(self, path):
        for rawFile in os.listdir(path):
            if ".png" in rawFile:
                self.img.append(cv2.imread(os.path.join(path, rawFile)))
        self.images = len(self.img) # in example "memorial": 16 images
        self.height = numpy.size(self.img[0], 0) # in example "memorial": 768
        self.width = numpy.size(self.img[0], 1) # in example "memorial": 512
        self.channels = numpy.size(self.img[0], 2) # in example "memorial": 3 (RGB)


        print(self.images, self.height, self.width, self.channels)

class DebevecHDR():
    def __init__(self):
        pass




if __name__ == '__main__':
    raw = InputImages()
    raw.readImages(examplePath)

