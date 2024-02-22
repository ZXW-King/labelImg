import argparse
import os
import sys
import os.path as osp
import cv2
import torch
import numpy as np
import onnxruntime as ort
from math import exp


CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]

meshgrid = []

class_num = len(CLASSES)
headNum = 3
strides = [8, 16, 32]
mapSize = [[80, 80], [40, 40], [20, 20]]
nmsThresh = 0.45
objectThresh = 0.35

input_imgH = 640
input_imgW = 640


class DetectBox:
    def __init__(self, classId, score, xmin, ymin, xmax, ymax):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax


def GenerateMeshgrid():
    for index in range(headNum):
        for i in range(mapSize[index][0]):
            for j in range(mapSize[index][1]):
                meshgrid.append(j + 0.5)
                meshgrid.append(i + 0.5)


def IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2):
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    innerWidth = xmax - xmin
    innerHeight = ymax - ymin

    innerWidth = innerWidth if innerWidth > 0 else 0
    innerHeight = innerHeight if innerHeight > 0 else 0

    innerArea = innerWidth * innerHeight

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    total = area1 + area2 - innerArea

    return innerArea / total


def NMS(detectResult):
    predBoxs = []

    sort_detectboxs = sorted(detectResult, key=lambda x: x.score, reverse=True)

    for i in range(len(sort_detectboxs)):
        xmin1 = sort_detectboxs[i].xmin
        ymin1 = sort_detectboxs[i].ymin
        xmax1 = sort_detectboxs[i].xmax
        ymax1 = sort_detectboxs[i].ymax
        classId = sort_detectboxs[i].classId

        if sort_detectboxs[i].classId != -1:
            predBoxs.append(sort_detectboxs[i])
            for j in range(i + 1, len(sort_detectboxs), 1):
                if classId == sort_detectboxs[j].classId:
                    xmin2 = sort_detectboxs[j].xmin
                    ymin2 = sort_detectboxs[j].ymin
                    xmax2 = sort_detectboxs[j].xmax
                    ymax2 = sort_detectboxs[j].ymax
                    iou = IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2)
                    if iou > nmsThresh:
                        sort_detectboxs[j].classId = -1
    return predBoxs


def sigmoid(x):
    return 1 / (1 + exp(-x))


def PostProcessor_YOLOV8(out, img_h, img_w):
    print('postprocess ... ')

    detectResult = []
    output = []
    for i in range(len(out)):
        print(out[i].shape)
        output.append(out[i].reshape((-1)))

    scale_h = img_h / input_imgH
    scale_w = img_w / input_imgW

    gridIndex = -2

    for index in range(headNum):
        reg = output[index * 2 + 0]
        cls = output[index * 2 + 1]

        for h in range(mapSize[index][0]):
            for w in range(mapSize[index][1]):
                gridIndex += 2

                for cl in range(class_num):
                    cls_val = sigmoid(cls[cl * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w])

                    if cls_val > objectThresh:
                        x1 = (meshgrid[gridIndex + 0] - reg[0 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]) * strides[index]
                        y1 = (meshgrid[gridIndex + 1] - reg[1 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]) * strides[index]
                        x2 = (meshgrid[gridIndex + 0] + reg[2 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]) * strides[index]
                        y2 = (meshgrid[gridIndex + 1] + reg[3 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]) * strides[index]

                        xmin = x1 * scale_w
                        ymin = y1 * scale_h
                        xmax = x2 * scale_w
                        ymax = y2 * scale_h

                        xmin = xmin if xmin > 0 else 0
                        ymin = ymin if ymin > 0 else 0
                        xmax = xmax if xmax < img_w else img_w
                        ymax = ymax if ymax < img_h else img_h

                        box = DetectBox(cl, cls_val, xmin, ymin, xmax, ymax)
                        detectResult.append(box)
    # NMS
    print('detectResult:', len(detectResult))
    predBox = NMS(detectResult)

    return predBox
