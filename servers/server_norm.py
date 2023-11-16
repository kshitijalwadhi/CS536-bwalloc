from concurrent import futures
import time
import cv2
import pickle
import grpc
import object_detection_pb2
import object_detection_pb2_grpc


from ..server import Detector
from .server_fcfs import FCFSQueue

from ..utils.constants import MAX_CAMERAS, MAX_POSSIBLE_ALLOC

from queue import PriorityQueue
import threading

class PriorityStreamQueue:
    def __init__(self):
        self.queue = PriorityQueue()
        self.thread = threading.Thread(target=self.process_queue)
        self.thread.daemon = True
        self.thread.start()

    def enqueue(self, item, priority):
        self.queue.put((-priority, item))

    def process_queue(self):
        while True:
            _, (detector, request, context, stream_info) = self.queue.get()
            self.handle_request(detector, request, context, stream_info)
            self.queue.task_done()

    def handle_request(self, detector, request, context):
        jpg = pickle.loads(request.jpeg_data)
        img = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
        result = detector.detect(img)
        return object_detection_pb2.BBoxes(data=pickle.dumps(result))

class StreamInfo:
    def __init__(self, id, incoming_fps, max_fps):
        self.id = id
        self.norm_factor = incoming_fps / max_fps
        self.resource_allocation = self.calculate_resource_allocation()

    def calculate_resource_allocation(self):
        # TODO: Need to change this to actual resource allocation
        return MAX_POSSIBLE_ALLOC * self.norm_factor

class DetectorServicer(object_detection_pb2_grpc.DetectorServicer):
    def __init__(self, detector=None, max_fps=30):
        self.detector = detector
        self.priority_queue = PriorityStreamQueue()
        self.streams = {}
        self.max_fps = max_fps

    def detect(self, request, context):
        client_id = request.client_id
        incoming_fps = request.incoming_fps
        if client_id not in self.streams:
            self.streams[client_id] = StreamInfo(client_id, incoming_fps, self.max_fps)
        
        stream_info = self.streams[client_id]
        self.priority_queue.enqueue((self.detector, request, context, stream_info), stream_info.resource_allocation)
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
