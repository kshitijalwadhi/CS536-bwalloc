import grpc

import object_detection_pb2_grpc

from object_detection_pb2 import Request, Response, BBoxes, InitRequest, InitResponse, CloseRequest, CloseResponse

from concurrent import futures
import time
import cv2
import pickle

from utils.constants import MAX_CAMERAS
from time import sleep

import random

from threading import Lock

import torch
import torchvision
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn

IMG_SIZE = 224
BW = 8000
DUMMY_DETECTOR = 0
DETECTOR_THRESH = 0.5


class ObjectDetector:
    def __init__(self):
        self.object_detection_model = fasterrcnn_resnet50_fpn(pretrained=True, progress=False)
        self.object_detection_model.eval();

    def detect(self, frame) -> BBoxes:
        output = []

        if(not DUMMY_DETECTOR):
            color_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image=Image.fromarray(color_converted)

            pil_image = pil_to_tensor(pil_image)
            pil_image = pil_image.unsqueeze(dim=0)
            pil_image = pil_image / 255.0
            pil_image = self.object_detection_model(pil_image)
            pil_image[0]["boxes"] = pil_image[0]["boxes"][pil_image[0]["labels"] ==1]
            pil_image[0]["scores"] = pil_image[0]["scores"][pil_image[0]["labels"] ==1]
            pil_image[0]["labels"] = pil_image[0]["labels"][pil_image[0]["labels"] ==1]
            pil_image[0]["boxes"] = pil_image[0]["boxes"][pil_image[0]["scores"] > DETECTOR_THRESH].detach().numpy()
            pil_image[0]["labels"] = pil_image[0]["labels"][pil_image[0]["scores"] > DETECTOR_THRESH].detach().numpy()
            pil_image[0]["scores"] = pil_image[0]["scores"][pil_image[0]["scores"] > DETECTOR_THRESH].detach().numpy()
            for i in range(len(pil_image[0]["boxes"])):
                output.append(['rectangle', pil_image[0]["boxes"][i][0],
                 pil_image[0]["boxes"][i][1], pil_image[0]["boxes"][i][2],
                 pil_image[0]["boxes"][i][3], pil_image[0]["scores"][i]])
        print(len(output))
        if(DUMMY_DETECTOR or len(output)==0):
            for _ in range(random.randint(1, 5)):
                output.append(
                    ['rectangle', random.randint(0, IMG_SIZE), random.randint(0, IMG_SIZE), random.randint(10, 50), random.randint(10, 50), random.random()]
                )
        res = BBoxes(data=pickle.dumps(output))
        return res


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
        with self.lock:
            self.current_load += len(request.frame_data)
            self.current_num_clients += 1
            self.connected_clients[request.client_id]["fps"] = request.fps
            self.connected_clients[request.client_id]["size_each_frame"] = len(request.frame_data)

            if self.current_load > BW:
                print("Server is overloaded")

        frame = pickle.loads(request.frame_data)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

        bboxes = self.detector.detect(frame)

        res = Response(
            bboxes=bboxes,
            signal=0
        )
        with self.lock:
            self.current_load -= len(request.frame_data)
            self.current_num_clients -= 1

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
