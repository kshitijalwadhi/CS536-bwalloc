import argparse

from imutils.video import FPS
import time
import grpc

import object_detection_pb2
import object_detection_pb2_grpc

def send_video(server_address, client_fps, client_packet_drop_rate):
    print("Sending video")
    channel = grpc.insecure_channel(args.server)
    stub = object_detection_pb2_grpc.DetectorStub(channel)
    #vs = VideoStream(src=0).start()
    time.sleep(1.0)
    fps = FPS().start()
    pass

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bandwidth Allocation for Multi-Stream Video")
    parser.add_argument('--server', 
        default='some-server:50051', 
        help='Server url:port'
    )
    parser.add_argument(
        "fps",
        type=int,
        required=False,
        help="Frame Rate for Client",
        metavar="",
        default=30
    )
    parser.add_argument(
        "packet_drop_rate",
        type=int,
        required=False,
        help="Packet Drop Rate for Client",
        metavar="",
        default=0
    )

if __name__ == "__main__":
    args = get_args()
    server_address = args.server
    client_fps = args.fps
    client_packet_drop_rate = args.packet_drop_rate
    send_video(server_address, client_fps, client_packet_drop_rate)