import argparse

from imutils.video import FPS
import time
import grpc

import cv2

import object_detection_pb2
import object_detection_pb2_grpc

def send_video(server_address, client_fps, client_packet_drop_rate):
    print("Sending video")
    channel = grpc.insecure_channel(server_address)
    stub = object_detection_pb2_grpc.DetectorStub(channel)
    #vs = VideoStream(src=0).start()
    time.sleep(1.0)
    fps = FPS().start()
    cap = cv2.VideoCapture('some_video.mp4') #TODO: CHANGE THIS
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    while(cap.isOpened()):

        ret, frame = cap.read()
        if ret == True:

            cv2.imshow('Video Frame', frame)
            fps.update()
            wait_time = int(1000/fps)
            if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                break

        else: 
            break
    cap.release()
    fps.stop()

    pass

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bandwidth Allocation for Multi-Stream Video")
    parser.add_argument('--server', 
        default='some-server:50051', 
        required=True,
        help='Server url:port'
    )
    parser.add_argument(
        "fps",
        type=int,
        required=False,
        help="Frame Rate for Client",
        default=30
    )
    parser.add_argument(
        "packet_drop_rate",
        type=int,
        required=False,
        help="Packet Drop Rate for Client",
        default=0
    )

if __name__ == "__main__":
    args = get_args()
    server_address = args.server
    client_fps = args.fps
    client_packet_drop_rate = args.packet_drop_rate
    send_video(server_address, client_fps, client_packet_drop_rate)