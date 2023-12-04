from object_detection_pb2 import BBoxes
import random
from utilities.constants import IMG_SIZE
import pickle
import time

import torch


class ObjectDetector:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    def detect(self, frame) -> BBoxes:
        output = []
        result = self.model(frame)

        if len(result.pandas().xyxy[0]) == 0:
            return BBoxes(data=pickle.dumps(output)), 0

        score = max(result.pandas().xyxy[0].confidence)
        for _, row in result.pandas().xyxy[0].iterrows():
            output.append(
                [row['name'], row['xmin'], row['ymin'], row['xmax'],  row['ymax'], row['confidence']]
            )
        res = BBoxes(data=pickle.dumps(output))
        return res, score
