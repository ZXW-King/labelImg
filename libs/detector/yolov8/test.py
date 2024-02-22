#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: 焦子傲
@contact: jiao1943@qq.com
@file: test.py
@time: 2021/4/9 14:05
@desc:
'''
import sys, os

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../../../'))

import argparse
import numpy as np
from libs.detector.ssd.onnxmodel import ONNXModel
import time
import cv2
from libs.detector.utils.timer import Timer
from libs.detector.utils.file import Walk
from libs.detector.yolov8.postprocess.postprocess import PostProcessor_YOLOV8, input_imgH, input_imgW, GenerateMeshgrid
from libs.detector.yolov8.preprocess import precess_image


CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]
SCORE_ID = 4

def GetArgs():
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--onnx", type=str, help="onnx model file")
    parser.add_argument("--image", type=str, help="image directory")

    args = parser.parse_args()
    return args


def main():
    args = GetArgs()
    if os.path.isfile(args.image):
        image_paths = [args.image, ]
    else:
        image_paths = Walk(args.image, ["jpg", "png", "jpeg"])
    GenerateMeshgrid()
    time_start1 = time.time()
    net = ONNXModel(args.onnx)
    time_end2 = time.time()
    print('load model cost', time_end2 - time_start1)

    for i, file in enumerate(sorted(image_paths)):
        timer = Timer()
        orig = cv2.imread(file)
        img_h, img_w = orig.shape[:2]
        timer.Timing("read image")
        image = precess_image(orig, 640, 640)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)
        timer.Timing("preprocess")

        pred_results = net.forward(image)
        timer.Timing("inference")
        out = []
        for i in range(len(pred_results)):
            out.append(pred_results[i])
        predbox = PostProcessor_YOLOV8(out, img_h, img_w)

        print('obj num is :', len(predbox))

        for i in range(len(predbox)):
            xmin = int(predbox[i].xmin)
            ymin = int(predbox[i].ymin)
            xmax = int(predbox[i].xmax)
            ymax = int(predbox[i].ymax)
            classId = predbox[i].classId
            score = predbox[i].score
            cv2.rectangle(orig, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            ptext = (xmin, ymin)
            title = CLASSES[classId] + "%.2f" % score
            cv2.putText(orig, title, ptext, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        # cv2.imwrite('./test_rknn_result.jpg', orig)
        cv2.imshow("draw", orig)
        cv2.waitKey()


if __name__ == '__main__':
    main()