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

        # for _ in range(random.randint(1, 5)):
        #     dummy_output.append(
        #         ['rectangle', random.randint(0, IMG_SIZE), random.randint(0, IMG_SIZE), random.randint(10, 50), random.randint(10, 50), random.random()]
        #     )
        result = self.model(frame)
        score = max(result.pandas().xyxy[0].confidence) if result.pandas().xyxy[0].confidence else 100
        for _, row in result.pandas().xyxy[0].iterrows():
            output.append(
                ['rectangle', row['xmin'], row['xmax'], row['ymin'], row['ymax'], row['confidence']]
            )
        res = BBoxes(data=pickle.dumps(output))
        # if random.random() < 0.8:
        #     score = random.uniform(85, 100)
        # else:
        #     score = random.uniform(0, 85)
        # time.sleep(0.05)
        return res, score
