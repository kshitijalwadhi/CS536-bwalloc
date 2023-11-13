from concurrent import futures
import time
import pickle
import grpc
import object_detection_pb2
import object_detection_pb2_grpc


from ..server import Detector
from .server_fcfs import FCFSQueue

from ..utils.constants import MAX_CAMERAS, MAX_POSSIBLE_ALLOC

class StreamInfo:
    def __init__(self, id, incoming_fps, max_fps):
        self.id = id
        self.norm_factor = incoming_fps / max_fps
        self.resource_allocation = self.calculate_resource_allocation()

    def calculate_resource_allocation(self):
        return MAX_POSSIBLE_ALLOC * self.norm_factor

class DetectorServicer(object_detection_pb2_grpc.DetectorServicer):
    def __init__(self, detector=None, max_fps=30):
        self.detector = detector
        self.fcfs_queue = FCFSQueue()
        self.streams = {}
        self.max_fps = max_fps

    def detect(self, request, context):
        stream_id = request.stream_id 
        incoming_fps = request.incoming_fps
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
