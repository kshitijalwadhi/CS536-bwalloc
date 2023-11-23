import argparse
from imutils.video import FPS
from imutils.video import FileVideoStream
import time
import grpc
import pickle

import cv2

import object_detection_pb2
import object_detection_pb2_grpc

from utils.utils import draw_result

def webcam(vs, mirror=False):
    while True:
        img = vs.read()
        if mirror: 
            img = cv2.flip(img, 1)
        # crop image to square as YOLO input
        if img.shape[0] < img.shape[1]:
            pad = (img.shape[1]-img.shape[0])//2
            img = img[:, pad: pad+img.shape[0]]
        else:
            pad = (img.shape[0]-img.shape[1])//2
            img = img[pad: pad+img.shape[1], :]
        yield img

def send_video(server_address, client_fps, client_packet_drop_rate, client_id):
    print("Sending video")
    channel = grpc.insecure_channel(server_address)
    stub = object_detection_pb2_grpc.DetectorStub(channel)
    
    time.sleep(1.0)
    fps = FPS().start()
    vs = FileVideoStream('some_video.mp4').start() #TODO: CHANGE THIS
    size = 224
    try:
        for img in webcam(vs, mirror=True):
            # compress frame
            resized_img = cv2.resize(img, (size, size))
            jpg = cv2.imencode('.jpg', resized_img)[1]
            # send to server for object detection
            response = stub.detect(object_detection_pb2.Image(
                jpeg_data=pickle.dumps(jpg)), 
                client_id=client_id, 
                client_fps=client_fps
            )
            # parse detection result and draw on the frame
            result = pickle.loads(response.data)
            display = draw_result(img, result, scale=float(img.shape[0])/size)
            cv2.imshow('Video Frame', display)
            wait_time = int(1000/client_fps)
            cv2.waitKey(wait_time)
            fps.update()
    except grpc._channel._Rendezvous as err:
        print(err)
    except KeyboardInterrupt:
        fps.stop()
        print('[INFO] elapsed time: {:.2f}'.format(fps.elapsed()))
        print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))
        cv2.destroyAllWindows()
        vs.stop()

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bandwidth Allocation for Multi-Stream Video")
    parser.add_argument('--server', 
        default='some-server:50051', 
        required=True,
        help='Server url:port'
    )
    parser.add_argument(
        "--id",
        type=int,
        required=True,
        help="Unique Client ID",
        default=30
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
    client_id = args.id
    client_fps = args.fps
    client_packet_drop_rate = args.packet_drop_rate
    send_video(server_address, client_fps, client_packet_drop_rate, client_id)
