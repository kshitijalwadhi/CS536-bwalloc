import grpc

import object_detection_pb2_grpc

from object_detection_pb2 import Request, Response, BBoxes, InitRequest, InitResponse, CloseRequest, CloseResponse

from concurrent import futures
import time
import cv2
import pickle

from utils.constants import MAX_CAMERAS, OD_THRESH, IMG_SIZE, BW
from time import sleep

import random

from threading import Lock


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
        else:  # 20% probability
            score = random.uniform(0, 85)
        return res, score


class Detector(object_detection_pb2_grpc.DetectorServicer):
    def __init__(self, detector=None) -> None:
        super(Detector, self).__init__()
        self.detector = detector
        self.current_load = 0
        self.current_num_clients = 0
        self.lock = Lock()
        self.connected_clients = {}

    def init_client(self, request: InitRequest, context):
        with self.lock:
            client_id = random.randint(1, MAX_CAMERAS)
            while client_id in self.connected_clients:
                client_id = random.randint(1, MAX_CAMERAS)
            self.connected_clients[client_id] = {
                "fps": 0,
                "size_each_frame": 0,
            }
        return InitResponse(
            client_id=client_id,
        )

    def close_connection(self, request: CloseRequest, context):
        with self.lock:
            if request.client_id in self.connected_clients:
                del self.connected_clients[request.client_id]
        return CloseResponse()

    def detect(self, request: Request, context):
        new_fps = request.fps
        change_fps = False
        with self.lock:
            self.current_load += len(request.frame_data)
            self.current_num_clients += 1
            self.connected_clients[request.client_id]["fps"] = request.fps
            self.connected_clients[request.client_id]["size_each_frame"] = len(request.frame_data)
            if score < OD_THRESH:
                print("Object Detection Score below threshold")
                change_fps = True

            if self.current_load > BW:
                print("Server is overloaded")
                change_fps = True
            
            if change_fps:
                new_fps = self.calculate_adjusted_fps(request.client_id)

        frame = pickle.loads(request.frame_data)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

        bboxes, score = self.detector.detect(frame)

        res = Response(
            bboxes=bboxes,
            signal=new_fps
        )
        with self.lock:
            self.current_load -= len(request.frame_data)
            self.current_num_clients -= 1

        return res
    
    def calculate_adjusted_fps(self, requesting_client_id):
        total_fps = 0
        max_fps = 0
        max_fps_client_id = None

        for client_id, info in self.connected_clients.items():
            total_fps += info["fps"]
            if info["fps"] > max_fps:
                max_fps = info["fps"]
                max_fps_client_id = client_id

        if max_fps_client_id == requesting_client_id:
            return 0

        average_fps = total_fps / len(self.connected_clients)
        return average_fps if max_fps_client_id is not None else 0


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
