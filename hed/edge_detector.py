import cv2 as cv
import numpy as np


class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = int((inputShape[2] - targetShape[2]) / 2)
        self.xstart = int((inputShape[3] - targetShape[3]) / 2)
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.ystart:self.yend, self.xstart:self.xend]]


class EdgeDetector(object):
    def __init__(self, prototxt='./deploy.prototxt', caffemodel='./hed_pretrained_bsds.caffemodel'):
        self.net = cv.dnn.readNetFromCaffe(prototxt, caffemodel)
        cv.dnn_registerLayer('Crop', CropLayer)

    def run(self, input):
        height, width, channels = input.shape

        img_blob = cv.dnn.blobFromImage(input, scalefactor=1.0, size=(width, height),
                                        mean=(104.00698793, 116.66876762,
                                              122.67891434),
                                        swapRB=False, crop=False)
        self.net.setInput(img_blob)

        out = self.net.forward()
        out = out[0, 0]
        out = cv.resize(out, (width, height))

        out = cv.cvtColor(out, cv.COLOR_GRAY2BGR)
        out = 255 * out
        out = out.astype(np.uint8)
        out = cv.bitwise_not(out)

        con = np.concatenate((input, out), axis=1)

        return con
