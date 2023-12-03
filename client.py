import argparse
from imutils.video import FileVideoStream
import time
import grpc
import pickle

import cv2

import object_detection_pb2_grpc

from object_detection_pb2 import Request, InitRequest, CloseRequest

from utils.utils import draw_result, yield_frames_from_video


class Client:
    def __init__(self, server_address, client_fps):
        self.channel = grpc.insecure_channel(server_address)
        self.stub = object_detection_pb2_grpc.DetectorStub(self.channel)

        req = InitRequest()
        resp = self.stub.init_client(req)
        self.client_id = resp.client_id

        print("Client ID: {}".format(self.client_id))

        self.fps = client_fps

    def send_video(self):
        print("Sending video to server")
        time.sleep(1.0)

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
                    fps=self.fps,
                    client_id=self.client_id,
                )

                resp = self.stub.detect(req)

                # parse detection result and draw on the frame
                result = pickle.loads(resp.bboxes.data)
                display = draw_result(img, result, scale=float(img.shape[0])/size)
                cv2.imshow('Video Frame', display)

                resp_fps = resp.fps

                self.fps = resp_fps + 3
                print("New FPS: ", self.fps)
                wait_time = int(1000/self.fps)
                cv2.waitKey(wait_time)
        except grpc._channel._Rendezvous as err:
            print(err)
        except KeyboardInterrupt:
            cv2.destroyAllWindows()
            vs.stop()

    def close_connection(self):
        req = CloseRequest(
            client_id=self.client_id,
        )
        resp = self.stub.close_connection(req)
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

    client = Client(server_address, client_fps)
    client.send_video()
    print("Closing connection")
    client.close_connection()
