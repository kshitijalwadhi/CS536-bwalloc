from concurrent import futures
import time
import pickle
import grpc
import object_detection_pb2
import object_detection_pb2_grpc


from ..server import Detector
from .server_fcfs import FCFSQueue

from ..utils.constants import MAX_CAMERAS

class StreamInfo:
    def __init__(self, id, incoming_fps, max_fps):
        self.id = id
        self.norm_factor = incoming_fps / max_fps
        # Calculate the resource allocation based on NORM_FACTOR
        self.resource_allocation = self.calculate_resource_allocation()

    def calculate_resource_allocation(self):
        MAX_POSSIBLE_ALLOC = 100  # Placeholder value, replace with your logic
        return MAX_POSSIBLE_ALLOC * self.norm_factor

class DetectorServicer(object_detection_pb2_grpc.DetectorServicer):
    def __init__(self, detector=None, max_fps=30):  # max_fps can be a predefined value
        self.detector = detector
        self.fcfs_queue = FCFSQueue()
        self.streams = {}  # Store StreamInfo objects
        self.max_fps = max_fps

    def detect(self, request, context):
        stream_id = request.stream_id  # Assuming the client sends a unique stream ID
        incoming_fps = request.incoming_fps  # Assuming the client sends its incoming FPS
        if stream_id not in self.streams:
            self.streams[stream_id] = StreamInfo(stream_id, incoming_fps, self.max_fps)
        
        # Enqueue with stream information
        self.fcfs_queue.enqueue((self.detector, request, context, self.streams[stream_id]))
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
