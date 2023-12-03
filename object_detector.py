from object_detection_pb2 import BBoxes
import random
from utils.constants import IMG_SIZE
import pickle
import time


class ObjectDetector:
    def detect(self, frame) -> BBoxes:
        dummy_output = []

        for _ in range(random.randint(1, 5)):
            dummy_output.append(
                ['rectangle', random.randint(0, IMG_SIZE), random.randint(0, IMG_SIZE), random.randint(10, 50), random.randint(10, 50), random.random()]
            )
        res = BBoxes(data=pickle.dumps(dummy_output))
        if random.random() < 0.8:
            score = random.uniform(85, 100)
        else:
            score = random.uniform(0, 85)
        time.sleep(0.05)
        return res, score
