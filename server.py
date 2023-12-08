import grpc

import object_detection_pb2_grpc

from object_detection_pb2 import Request, Response, BBoxes, InitRequest, InitResponse, CloseRequest, CloseResponse

from concurrent import futures
import time
import cv2
import pickle

from utilities.constants import MAX_CAMERAS, OD_THRESH, IMG_SIZE, MAX_BW

from object_detector import ObjectDetector

import random

from threading import Lock

import threading

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import numpy as np


class Detector(object_detection_pb2_grpc.DetectorServicer):
    def __init__(self, detector=None) -> None:
        super(Detector, self).__init__()
        self.detector = detector
        self.current_load = 0
        self.current_num_clients = 0
        self.lock = Lock()
        self.connected_clients = {}
        self.connected_clients_plotting = {}
        self.accuracies = {}
        self.od_scores = {}
        self.verbose = 1
        self.total_bandwidth = []
        self.bandwidths = {}
        self.server_start_time = time.time()
        self.start_times = {}
        self.client_times = {}
        self.server_times = []
        self.client_fps = {}
        self.client_res = {}

        self.pending_client_updates = {}  # client_id -> fps

    def init_client(self, request: InitRequest, context):
        with self.lock:
            client_id = random.randint(1, MAX_CAMERAS)
            while client_id in self.connected_clients:
                client_id = random.randint(1, MAX_CAMERAS)
            self.connected_clients[client_id] = {
                "requested_fps": request.fps,
                "fps": 0,
                "size_each_frame": 0,
            }
            self.connected_clients_plotting[client_id] = {
                "requested_fps": request.fps,
                "fps": 1,
                "size_each_frame": 1024,
            }

            self.od_scores[client_id] = []
            self.od_scores[client_id].append(1)
            self.accuracies[client_id] = []
            self.accuracies[client_id].append(1)
            self.bandwidths[client_id] = []
            self.bandwidths[client_id].append(0)
            self.client_fps[client_id] = []
            self.client_fps[client_id].append(request.fps)
            self.client_res[client_id] = []
            self.client_res[client_id].append(request.res)
            self.client_times[client_id] = []
            self.client_times[client_id].append(time.time() - self.server_start_time)
            self.start_times[client_id] = time.time() - self.server_start_time
            self.server_times.append(time.time() - self.server_start_time)
            self.total_bandwidth.append(0)

        return InitResponse(
            client_id=client_id,
        )

    def plot_metrics(self):
        # Create a figure with two subplots
        fig = plt.figure(figsize=(12, 6))

        # Plotting Total Bandwidth
        plt.subplot(3, 2, 1)
        total_bandwidth = self.total_bandwidth
        print(len(total_bandwidth))
        print(len(self.server_times))
        averaged_bw = [np.mean(total_bandwidth[i:i+5]) for i in range(0, len(total_bandwidth), 5)]
        times = [np.mean(self.server_times[i:i+5]) for i in range(0, len(self.server_times), 5)]

        plt.plot(times, averaged_bw[:len(times)])
        plt.title('Total Bandwidth over Time')
        plt.xlabel('Time')
        plt.ylabel('Total Bandwidth')

        # Plotting Bandwidth of Each Client
        plt.subplot(3, 2, 2)
        for client_id, bw in self.bandwidths.items():
            averaged_bw = [np.mean(bw[i:i+5]) for i in range(0, len(bw), 5)]
            times = [np.mean(self.client_times[client_id][i:i+5]) for i in range(0, len(self.client_times[client_id]), 5)]
            fps = self.connected_clients_plotting[client_id]['requested_fps']
            plt.plot(times, averaged_bw[:len(times)], label=f'Client FPS: {fps}')

        plt.title('Bandwidth of Each Client')
        plt.xlabel('Time')
        plt.ylabel('Bandwidth')

        # Plotting Object Detection Score of Each Client
        plt.subplot(3, 2, 3)
        for client_id, score in self.od_scores.items():
            averaged_scores = [np.mean(score[i:i+5]) for i in range(0, len(score), 5)]
            times = [np.mean(self.client_times[client_id][i:i+5]) for i in range(0, len(self.client_times[client_id]), 5)]
            fps = self.connected_clients_plotting[client_id]['requested_fps']
            plt.plot(times, averaged_scores[:len(times)], label=f'Client FPS: {fps}')

        plt.title('Object Detection Scores of Each Client')
        plt.xlabel('Time')
        plt.ylabel('Score')

        plt.subplot(3, 2, 4)
        for client_id, fps in self.client_fps.items():
            
            times = [np.mean(self.client_times[client_id][i:i+5]) for i in range(0, len(self.client_times[client_id]), 5)]
            averaged_fps = [np.mean(fps[i:i+5]) for i in range(0, len(fps), 5)]
            fps_label = self.connected_clients_plotting[client_id]['requested_fps']
            plt.plot(times, averaged_fps[:len(times)], label=f'Client FPS: {fps_label}')

        plt.title('Client FPS over Time')
        plt.xlabel('Time')
        plt.ylabel('FPS')

        plt.subplot(3, 2, 5)
        for client_id, res in self.client_res.items():
            
            times = [np.mean(self.client_times[client_id][i:i+5]) for i in range(0, len(self.client_times[client_id]), 5)]
            averaged_res = [np.mean(res[i:i+5]) for i in range(0, len(res), 5)]
            fps = self.connected_clients_plotting[client_id]['requested_fps']
            plt.plot(times, averaged_res[:len(times)], label=f'Client FPS: {fps}')

        plt.title('Client Scaling Factor over Time')
        plt.xlabel('Time')
        plt.ylabel('FPS')

        handles, labels = plt.gca().get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower right')

        plt.tight_layout()

        # Save the plot to a file
        plt.savefig('./server_metrics.png')
        print("Plot saved to 'server_metrics.png'")
        plt.close()

    def close_connection(self, request: CloseRequest, context):
        with self.lock:
            if request.client_id in self.connected_clients:
                del self.connected_clients[request.client_id]
        return CloseResponse()

    def detect(self, request: Request, context):
        new_fps = request.fps
        with self.lock:
            self.current_num_clients += 1
            self.connected_clients[request.client_id]["fps"] = request.fps
            self.connected_clients[request.client_id]["size_each_frame"] = len(request.frame_data)

            self.connected_clients_plotting[request.client_id]["fps"] = request.fps
            self.connected_clients_plotting[request.client_id]["size_each_frame"] = len(request.frame_data)
        
        self.current_load = 0

        for client_id in self.connected_clients:
            load = self.connected_clients[client_id]["size_each_frame"]
            self.current_load += load

        load = len(request.frame_data)
        self.bandwidths[request.client_id].append(load)

        self.client_fps[request.client_id].append(request.fps)
        self.client_res[request.client_id].append(request.res)

        self.total_bandwidth.append(self.current_load)
        self.server_times.append(time.time() - self.server_start_time)

        frame = pickle.loads(request.frame_data)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

        bboxes, score = self.detector.detect(frame)

        self.od_scores[request.client_id].append(score)
        self.client_times[request.client_id].append(time.time() - self.server_start_time)

        increase_quality_flag = False
        decrease_quality_flag = False

        target_fps = self.connected_clients[request.client_id]["requested_fps"]

        fps_factor = request.fps / target_fps
        print(f"fps factor: {fps_factor}")
        print(f"target_fps: {target_fps}")
        print(f"request.fps: {request.fps}")

        if int(score) < OD_THRESH:
            print(f"Object Detection Score below threshold: {score}")
            increase_quality_flag = True

        with self.lock:
            if self.current_load > MAX_BW:
                print("Max Bandwidth Exceeded")
                print(f"Current Load: {self.current_load}")
                self.calculate_adjusted_fps_bw_exceed()

                if fps_factor > 1.5:
                    self.pending_client_updates[request.client_id] = request.fps - fps_factor*10
            
            if fps_factor < 0.5:
                self.pending_client_updates[request.client_id] = request.fps + 0.1*target_fps

        if request.client_id in self.pending_client_updates:
            new_fps = self.pending_client_updates[request.client_id]
            del self.pending_client_updates[request.client_id]
            print("new fps from pending_client_updates: ", new_fps)
            decrease_quality_flag = True

        if increase_quality_flag and decrease_quality_flag:
            increase_quality_flag = False

        res = Response(
            bboxes=bboxes,
            fps=int(new_fps),
            increase_quality=increase_quality_flag,
            decrease_quality=decrease_quality_flag,
        )

        with self.lock:
            self.current_load -= len(request.frame_data)
            self.current_num_clients -= 1

        return res

    def calculate_adjusted_fps_bw_exceed(self):
        total_fps = 0
        max_fps = 0
        max_fps_client_id = None

        for client_id, info in self.connected_clients.items():
            total_fps += info["fps"]
            if info["fps"] > max_fps:
                max_fps = info["fps"]
                max_fps_client_id = client_id

        average_fps = total_fps / (2 * len(self.connected_clients))
        self.pending_client_updates[max_fps_client_id] = average_fps


def start_plotting_thread(server_instance, interval=5):
    def plot():
        while not stop_plotting_thread.is_set():
            time.sleep(interval)
            server_instance.plot_metrics()

    stop_plotting_thread = threading.Event()
    plotting_thread = threading.Thread(target=plot)
    plotting_thread.start()
    return stop_plotting_thread, plotting_thread


if __name__ == '__main__':
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_CAMERAS))
    od_server = Detector(detector=ObjectDetector())
    object_detection_pb2_grpc.add_DetectorServicer_to_server(od_server, server)
    server.add_insecure_port('[::]:50051')
    server.start()
    stop_event, plotting_thread = start_plotting_thread(od_server)
    print("Server started at port 50051")
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)
        stop_event.set()  # Signal the plotting thread to stop
        plotting_thread.join()  # Wait for the plotting thread to finish
        server.stop(0)
