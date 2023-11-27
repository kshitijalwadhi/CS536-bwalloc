import argparse
from imutils.video import FPS
from imutils.video import FileVideoStream
import time
import grpc
import pickle

import cv2

import object_detection_pb2_grpc

from object_detection_pb2 import Request, Response, BBoxes

from utils.utils import draw_result


def yield_frames_from_video(vs, mirror=False):
    while True:
        img = vs.read()
        if img is None:
            break
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
    print("Initializing client")
    channel = grpc.insecure_channel(server_address)
    stub = object_detection_pb2_grpc.DetectorStub(channel)

    time.sleep(1.0)
    fps = FPS().start()
    vs = FileVideoStream('sample.mp4').start()
    size = 224
    scaling_factor = 4
    try:
        for img in yield_frames_from_video(vs, mirror=True):
            # compress frame
            resized_img = cv2.resize(img, (size//scaling_factor, size//scaling_factor))
            jpg = cv2.imencode('.jpg', resized_img)[1]
            # send to server for object detection
            frame_data = pickle.dumps(jpg)
            req = Request(
                frame_data=frame_data,
                fps=client_fps,
                client_id=client_id,
            )

            resp = stub.detect(req)

            # parse detection result and draw on the frame
            result = pickle.loads(resp.bboxes.data)
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
                        default='localhost:50051',
                        help='Server url:port'
                        )
    parser.add_argument(
        "--id",
        type=int,
        help="Unique Client ID",
        default=30
    )
    parser.add_argument(
        "--fps",
        type=int,
        help="Frame Rate for Client",
        default=30
    )
    parser.add_argument(
        "--packet_drop_rate",
        type=int,
        help="Packet Drop Rate for Client",
        default=0
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    server_address = args.server
    client_id = args.id
    client_fps = args.fps
    client_packet_drop_rate = args.packet_drop_rate
    send_video(server_address, client_fps, client_packet_drop_rate, client_id)
