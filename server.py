import grpc

import object_detection_pb2_grpc

from object_detection_pb2 import Request, Response, BBoxes

from concurrent import futures
import time
import cv2
import pickle

from utils.constants import MAX_CAMERAS
from time import sleep

import random

IMG_SIZE = 224

class ObjectDetector:
    def detect(self, frame) -> BBoxes:
        dummy_output = []

        for _ in range(random.randint(1, 5)):
            dummy_output.append(
                ['rectangle', random.randint(0, 224), random.randint(0, 224), random.randint(10, 50), random.randint(10, 50), random.random()]
            )
        res = BBoxes(data=pickle.dumps(dummy_output))
        return res

class Detector(object_detection_pb2_grpc.DetectorServicer):
    def __init__(self, detector=None) -> None:
        super(Detector, self).__init__()
        self.detector = detector

    def detect(self, request:Request, context):
        frame = pickle.loads(request.frame_data)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

        bboxes = self.detector.detect(frame)

        res = Response(
            bboxes=bboxes,
            signal=0
        )
        return res


def serve(detector):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_CAMERAS))
    object_detection_pb2_grpc.add_DetectorServicer_to_server(detector, server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("Server started at port 50051")
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    serve(Detector(detector=ObjectDetector()))
