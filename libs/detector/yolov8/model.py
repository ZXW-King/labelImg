#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: 焦子傲
@contact: jiao1943@qq.com
@file: model.py
@time: 2021/4/14 10:56
@desc:
'''
import os
from libs.detector.ssd.onnxmodel import ONNXModel
from libs.detector.yolov3.postprocess.postprocess import load_class_names
from libs.detector.yolov8.preprocess import preProcessPadding
from libs.detector.yolov8.postprocess.postprocess import PostProcessor_YOLOV8, GenerateMeshgrid
import cv2

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__)).split('libs')[0]


class YOLOv8(object):
    def __init__(self, file='./config/human/yolov8.onnx', class_sel=[]):
        class_path = os.path.split(file)[0]
        self.classes = load_class_names(class_path + "/classes.names")
        self.class_sel = class_sel

        if os.path.isfile(file):
            self.net = ONNXModel(file)
        else:
            raise IOError("no such file {}".format(file))

    def forward(self, image):
        oriY = image.shape[0]
        oriX = image.shape[1]
        image = preProcessPadding(image)
        pred_results = self.net.forward(image)
        out = []
        for i in range(len(pred_results)):
            out.append(pred_results[i])
        GenerateMeshgrid()
        predbox = PostProcessor_YOLOV8(out, oriY,oriX)

        # TODO : get rect
        shapes = []
        results_box = []
        for i in range(len(predbox)):
            xmin = int(predbox[i].xmin)
            ymin = int(predbox[i].ymin)
            xmax = int(predbox[i].xmax)
            ymax = int(predbox[i].ymax)
            label = predbox[i].classId
            score = predbox[i].score
            x, y, x2, y2, score, label = int(xmin), int(ymin), int(xmax), int(ymax), float(score), int(label)
            shapes.append((self.classes[label], [(x, y), (x2, y), (x2, y2), (x, y2)], None, None, False, 0))
            results_box.append([x, y, x2, y2, score, self.classes[label]])
        return shapes, results_box
