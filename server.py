import grpc

import object_detection_pb2
import object_detection_pb2_grpc

from concurrent import futures
import time
import cv2
import pickle

MAX_WORKERS = 10


class Detector(object_detection_pb2_grpc.DetectorServicer):
    def __init__(self, detector=None) -> None:
        super(Detector, self).__init__()
        self.detector = detector

    def detect(self, request, context):
        frame = pickle.loads(request.frame)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

        res = self.detector.detect(frame)
        return object_detection_pb2.BBoxes(data=pickle.dumps(res))

    pass


def serve(detector):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_WORKERS))
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
    serve(Detector())
