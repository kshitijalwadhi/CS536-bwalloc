import argparse
from imutils.video import FileVideoStream
import time
import grpc
import pickle

import cv2

import object_detection_pb2_grpc

from object_detection_pb2 import Request, InitRequest, CloseRequest

from utilities.helpers import draw_result, yield_frames_from_video
from utilities.constants import MAX_FPS, MAX_SCALING_FACTOR, MIN_SCALING_FACTOR

import time


class Client:
    def __init__(self, server_address, client_fps):
        self.channel = grpc.insecure_channel(server_address)
        self.stub = object_detection_pb2_grpc.DetectorStub(self.channel)

        req = InitRequest(fps=client_fps)
        resp = self.stub.init_client(req)
        self.client_id = resp.client_id

        print("Client ID: {}".format(self.client_id))

        self.fps = client_fps
        self.size = 224
        self.scaling_factor = 4

    def send_video(self):
        print("Sending video to server")
        time.sleep(1.0)

        vs = FileVideoStream('sample.mp4').start()
        roi = None
        use_roi = False
        try:
            for img in yield_frames_from_video(vs, mirror=True):

                t1 = time.time()

                if roi is not None and use_roi:
                    # Process only within the ROI if defined
                    x, y, w, h = roi
                    img_roi = img[y:y+h, x:x+w]
                else:
                    img_roi = img

                # compress frame
                resized_img = cv2.resize(img_roi, (self.size//self.scaling_factor, self.size//self.scaling_factor))
                jpg = cv2.imencode('.jpg', resized_img)[1]
                # send to server for object detection
                frame_data = pickle.dumps(jpg)
                req = Request(
                    frame_data=frame_data,
                    fps=self.fps,
                    client_id=self.client_id,
                )

                resp = self.stub.detect(req)

                if resp.increase_quality == True:
                    self.scaling_factor = max(MIN_SCALING_FACTOR, self.scaling_factor - 2)

                if resp.decrease_quality == True:
                    self.scaling_factor = min(MAX_SCALING_FACTOR, self.scaling_factor + 2)

                # parse detection result and draw on the frame
                result = pickle.loads(resp.bboxes.data)

                # calculate ROI
                if result and use_roi and len(result) > 4:
                    x_coords = [bbox[1] for bbox in result]
                    y_coords = [bbox[2] for bbox in result]
                    heights = [bbox[3] for bbox in result]
                    widths = [bbox[4] for bbox in result]
                    x_min, x_max = min(x_coords), max(x + w for x, w in zip(x_coords, widths))
                    y_min, y_max = min(y_coords), max(y + h for y, h in zip(y_coords, heights))
                    roi = (x_min, y_min, x_max - x_min, y_max - y_min)

                if result:
                    display = draw_result(img, result, scale=float(img.shape[0])/self.size)
                    cv2.imshow('Video Frame', display)
                else:
                    cv2.imshow('Video Frame', img)
                resp_fps = resp.fps

                self.fps = resp_fps + int(0.1*MAX_FPS) if resp_fps < MAX_FPS else MAX_FPS

                t2 = time.time()

                time_elapsed = (t2 - t1) * 1000

                if (time_elapsed > 1000/self.fps):
                    #self.fps = int(1000/time_elapsed)
                    cv2.waitKey(1)
                else:
                    cv2.waitKey(int(1000/self.fps - time_elapsed))

                print("New FPS: ", self.fps)
                print("Current scaling factor: ", self.scaling_factor)
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
