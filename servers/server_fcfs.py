from concurrent import futures
import time
import cv2
import pickle
import grpc
import object_detection_pb2
import object_detection_pb2_grpc
from queue import Queue
import threading

from ..server import Detector

from ..utils.constants import MAX_CAMERAS

class FCFSQueue:
    def __init__(self):
        self.queue = Queue()
        self.thread = threading.Thread(target=self.process_queue)
        self.thread.daemon = True
        self.thread.start()
    
    def enqueue(self, item):
        self.queue.put(item)

    def process_queue(self):
        while True:
            detector, request, context = self.queue.get()
            self.handle_request(detector, request, context)
            self.queue.task_done()

    def handle_request(self, detector, request, context):
        jpg = pickle.loads(request.jpeg_data)
        img = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
        result = detector.detect(img)
        return object_detection_pb2.BBoxes(data=pickle.dumps(result))

class DetectorServicer(object_detection_pb2_grpc.DetectorServicer):
    def __init__(self, detector=None):
        self.detector = detector
        self.fcfs_queue = FCFSQueue()
    
    def detect(self, request, context):
        self.fcfs_queue.enqueue((self.detector, request, context))
        return object_detection_pb2.BBoxes(data=pickle.dumps('Request enqueued'))

def serve(detector):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_CAMERAS))
    object_detection_pb2_grpc.add_DetectorServicer_to_server(DetectorServicer(detector), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(60 * 60 * 24)
    except KeyboardInterrupt:
        server.stop(0)
    pass

if __name__ == '__main__':
    serve(Detector())
