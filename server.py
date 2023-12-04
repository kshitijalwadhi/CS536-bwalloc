import grpc

import object_detection_pb2_grpc

from object_detection_pb2 import Request, Response, BBoxes, InitRequest, InitResponse, CloseRequest, CloseResponse

from concurrent import futures
import time
import cv2
import pickle

from utilities.constants import MAX_CAMERAS, OD_THRESH, IMG_SIZE, BW

from object_detector import ObjectDetector

import random

from threading import Lock


class Detector(object_detection_pb2_grpc.DetectorServicer):
    def __init__(self, detector=None) -> None:
        super(Detector, self).__init__()
        self.detector = detector
        self.current_load = 0
        self.current_num_clients = 0
        self.lock = Lock()
        self.connected_clients = {}

        self.pending_client_updates = {}  # client_id -> fps

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
        with self.lock:
            self.current_load += len(request.frame_data)
            self.current_num_clients += 1
            self.connected_clients[request.client_id]["fps"] = request.fps
            self.connected_clients[request.client_id]["size_each_frame"] = len(request.frame_data)

        frame = pickle.loads(request.frame_data)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

        bboxes, score = self.detector.detect(frame)

        increase_quality_flag = False
        decrease_quality_flag = False

        if int(score) < OD_THRESH:
            print(f"Object Detection Score below threshold: {score}")
            increase_quality_flag = True

        with self.lock:
            if self.current_load > BW:
                print("Max Bandwidth Exceeded")
                self.calculate_adjusted_fps_bw_exceed()

        if request.client_id in self.pending_client_updates:
            new_fps = self.pending_client_updates[request.client_id]
            del self.pending_client_updates[request.client_id]
            print("new fps from pending_client_updates: ", new_fps)
            decrease_quality_flag = True

        if increase_quality_flag and decrease_quality_flag:
            increase_quality_flag = False

        res = Response(
            bboxes=bboxes,
            fps=int(new_fps),
            increase_quality=increase_quality_flag,
            decrease_quality=decrease_quality_flag,
        )

        with self.lock:
            self.current_load -= len(request.frame_data)
            self.current_num_clients -= 1

        return res

    def calculate_adjusted_fps_bw_exceed(self):
        total_fps = 0
        max_fps = 0
        max_fps_client_id = None

        for client_id, info in self.connected_clients.items():
            total_fps += info["fps"]
            if info["fps"] > max_fps:
                max_fps = info["fps"]
                max_fps_client_id = client_id

        average_fps = total_fps / (2 * len(self.connected_clients))
        self.pending_client_updates[max_fps_client_id] = average_fps


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
