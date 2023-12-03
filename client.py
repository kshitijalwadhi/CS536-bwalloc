import argparse
from imutils.video import FPS
from imutils.video import FileVideoStream
import time
import grpc
import pickle

import cv2

import object_detection_pb2_grpc

from object_detection_pb2 import Request, Response, BBoxes, InitRequest, InitResponse, CloseRequest, CloseResponse

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


def send_init_request(server_address):
    channel = grpc.insecure_channel(server_address)
    stub = object_detection_pb2_grpc.DetectorStub(channel)
    req = InitRequest()
    resp = stub.init_client(req)
    return resp.client_id


def send_video(server_address, client_fps, client_id):
    print("Sending video to server")
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

            resp_fps = resp.fps

            client_fps = resp_fps if resp_fps < 100 else client_fps
            if resp_fps < 95:
                client_fps += 5
            print(client_fps)
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


def close_connection(server_address, client_id):
    channel = grpc.insecure_channel(server_address)
    stub = object_detection_pb2_grpc.DetectorStub(channel)
    req = CloseRequest(
        client_id=client_id,
    )
    resp = stub.close_connection(req)
    return resp


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bandwidth Allocation for Multi-Stream Video")
    parser.add_argument('--server',
                        default='localhost:50051',
                        help='Server url:port'
                        )
    parser.add_argument(
        "--fps",
        type=int,
        help="Frame Rate for Client",
        default=30
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    server_address = args.server
    client_fps = args.fps

    client_id = send_init_request(server_address)
    print("Client ID: {}".format(client_id))

    send_video(server_address, client_fps, client_id)

    print("Closing connection")
    close_connection(server_address, client_id)
