from concurrent import futures
import time
import pickle
import grpc
import object_detection_pb2
import object_detection_pb2_grpc


from ..server import Detector
from .server_norm import PriorityStreamQueue

from ..utils.constants import MAX_CAMERAS


class StreamInfo:
    def __init__(self, id, client_fps, max_fps):
        self.id = id
        self.client_fps = client_fps

class DetectorServicer(object_detection_pb2_grpc.DetectorServicer):
    def __init__(self, detector=None, max_fps=30):
        self.detector = detector
        self.priority_queue = PriorityStreamQueue()
        self.streams = {}
        self.max_fps = max_fps

    def detect(self, request, context):
        client_id = request.client_id
        client_fps = request.client_fps
        if client_id not in self.streams:
            self.streams[client_id] = StreamInfo(client_id, client_fps, self.max_fps)
        
        stream_info = self.streams[client_id]
        self.priority_queue.enqueue((self.detector, request, context, stream_info), -1*stream_info.client_fps)
        return object_detection_pb2.BBoxes(data=pickle.dumps('Request enqueued'))


def serve(detector):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_CAMERAS))
    object_detection_pb2_grpc.add_DetectorServicer_to_server(DetectorServicer(detector), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)
    pass

if __name__ == '__main__':
    serve(Detector())
