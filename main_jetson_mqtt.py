import sys
import signal
import os

sys.path.insert(0, os.path.abspath("supervision"))
""" The Supervision library is used for object detection and tracking. And this demo contains an edited version of the library"""
import supervision as sv
from ultralytics import YOLO

import cv2
import numpy as np
from datetime import datetime
import torch
import argparse

from mqttLogHandler import MqttLogHandler as MqttHandler
import logging

class VehicleCounter:
    """Class to handle vehicle detection and counting using YOLO model.
    This class initializes the YOLO model, sets up the video capture, and processes frames
    to detect vehicles, count them based on their direction, and log the results to a file.
    """


    def __init__(self, model_path, device_name, video_number, logger):
        """Initializes the VehicleCounter with the model, device, output file, and video parameters.
        Args:
            model_path (str): Path to the YOLO model file.
            device_name (str): Device to run the model on ('cuda' or 'cpu').
            output_txt (str): Name of the output text file for logging counts.
            video_number (int): Number of the video to process (1 or 2).
        """
        self.model = YOLO(model_path)
        self.device = self.get_device(device_name)
        self.model.to(device=self.device)

        # Initialize the ByteTrack tracker
        self.tracker = sv.ByteTrack()

        # Define class names and queried IDs for vehicles
        self.class_names_dict = self.model.model.names
        self.class_ids = [2, 3, 5, 7]  # car, motorcycle, bus, truck

        # Set video parameters based on the video number
        self.video_number = video_number
        self.start_point, self.end_point, self.cap = self.get_video_parameters(video_number)

        # The line zone is the area where vehicles will be counted if they cross it
        # The start and end points define the line's position in the video frame
        self.count_line = sv.LineZone(start=self.start_point, end=self.end_point)
        self.line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=0.5)

        # Initialize counts for inbound and outbound vehicles
        self.curr_in_count = 0
        self.curr_out_count = 0

        # Handle graceful shutdown on SIGINT (Ctrl+C)
        self.keep_running = True

        # Set up Mqtt logging
        self.logger = logger
        self.logger.info("VehicleCounter initialized with model: %s, device: %s", model_path, device_name)

    def stop(self, signum, frame):
        print(f"\n[INFO] Caught signal {signum}. Exiting gracefully...")
        self.keep_running = False



    def get_device(self, device_name):
        """Determines the device to run the model on.
        Args:
            device_name (str): Name of the device ('cuda' or 'cpu').
        Returns:
            str: The device to use for model inference.
        Raises:
            ValueError: If the device name is not 'cuda' or 'cpu'.
        """
        if device_name not in ["cuda", "cpu"]:
            raise ValueError("Invalid device. Choose 'cuda' or 'cpu'.")
        if device_name == "cuda" and torch.cuda.is_available():
            print("Using CUDA.")
            return "cuda"
        print("Using CPU.")
        return "cpu"

    def get_video_parameters(self, video_number):
        """Sets the start and end points for the counting line and initializes video capture.
        Args:
            video_number (int): Number of the video to process (1 or 2).
        Returns:
            tuple: Start and end points for the counting line, and the video capture object.
        Raises:
            ValueError: If the video number is not 1 or 2.
        """
        if video_number == 1:
            start, end = sv.Point(x=150, y=300), sv.Point(x=1190, y=530)
            cap = cv2.VideoCapture('videos/1.mp4')
        elif video_number == 2:
            start, end = sv.Point(x=150, y=450), sv.Point(x=1200, y=450)
            cap = cv2.VideoCapture('videos/2.avi')
        elif video_number == 3:
            start, end = sv.Point(x=40, y=120), sv.Point(x=450, y=100)
            stream_url = "https://s58.nysdot.skyvdn.com/rtplive/TA_208/chunklist_w1095005895.m3u8"
            cap = cv2.VideoCapture(stream_url)
            if not cap.isOpened():
                print("Failed to open stream.")
                exit()
        elif video_number == 4:
            start, end = sv.Point(x=170, y=160), sv.Point(x=500, y=200)
            stream_url = "http://80.151.142.110:8080/?action=stream"
            cap = cv2.VideoCapture(stream_url)
            if not cap.isOpened():
                print("Failed to open stream.")
                exit()
        else:
            raise ValueError("Invalid video number.")
        return start, end, cap

    def log_counts(self):
        """ This method formats the counts into a log entry and publishes it to the MQTT topic."""
        self.logger.info("Total vehicles: %s, Vehicles inbound: %s, Vehicles outbound: %s", self.curr_in_count + self.curr_out_count, self.curr_in_count, self.curr_out_count)


    def process_detections(self, frame):
        """Processes the frame to detect vehicles and update counts.
        Args:
            frame (numpy.ndarray): The current video frame to process.
        Returns:
            sv.Detections: Detections of vehicles in the frame after filtering.
        """
        # Run the model on the current frame
        results = self.model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[np.isin(detections.class_id, self.class_ids)]
        detections = detections[np.greater(detections.confidence, 0.6)]
        detections = self.tracker.update_with_detections(detections)
        self.count_line.trigger(detections)
        return detections

    def run(self):
        """Runs the vehicle detection and counting process.
        This method captures video frames, processes them for vehicle detection,
        annotates the frames, and displays the results in a window.
        It also logs the counts of inbound and outbound vehicles to the output file.
        """
        ret, frame = self.cap.read()

        while ret and self.keep_running:
            detections = self.process_detections(frame)

            if (self.curr_in_count < self.count_line.in_count or
                self.curr_out_count < self.count_line.out_count):
                self.curr_in_count = self.count_line.in_count
                self.curr_out_count = self.count_line.out_count
                self.log_counts()


            #For execution on Jetson we ommit displaying of frames

            ret, frame = self.cap.read()

        self.cleanup()

    def cleanup(self):
        """Cleans up resources after processing is complete.
        This method releases the video capture object, closes the output file,
        and destroys all OpenCV windows.
        """
        self.cap.release()

def parse_args():
    parser = argparse.ArgumentParser(description="Vehicle Detection and Counting")
    parser.add_argument("--vid", type=int, choices=[1, 2,3, 4], default=1, help="Video number to use for detection")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run the model on")
    parser.add_argument("--model", type=str, default="yolov8m.pt", choices=["yolov8m.pt", "yolov8s.pt", "yolov8n.pt", "yolov5su.pt"], help="Path to the YOLO model file")
    parser.add_argument("--mqtt_broker", type=str, default="mosquitto_test", help="MQTT broker host name or IP address")
    parser.add_argument("--mqtt_port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--mqtt_topic", type=str, default="counting/logs", help="MQTT topic to publish logs to")
    return parser.parse_args()

def main():
    args = parse_args()

    # MQTT log setup
    logger = logging.getLogger("vehicle_logger")
    logger.setLevel(logging.INFO)

    mqtt_handler = MqttHandler(args.mqtt_broker, args.mqtt_port, args.mqtt_topic)
    mqtt_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(mqtt_handler)

    logger.info("Jetson Vehicle Counting container started")

    counter = VehicleCounter(
        model_path=args.model,
        device_name=args.device,
        video_number=args.vid,
        logger = logger
    )

    signal.signal(signal.SIGINT, counter.stop)   # Ctrl+C
    signal.signal(signal.SIGTERM, counter.stop)  # docker stop

    counter.run()
    logger.info("Stream finished.")
    mqtt_handler.close()

if __name__ == "__main__":
    main()
