import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np
from datetime import datetime
import torch
import argparse

class VehicleCounter:
    """Class to handle vehicle detection and counting using YOLO model.
    This class initializes the YOLO model, sets up the video capture, and processes frames
    to detect vehicles, count them based on their direction, and log the results to a file.
    """


    def __init__(self, model_path, device_name, output_txt, video_number):
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
        self.output_file = open(output_txt, "w")
        self.output_file.write("************Vehicle Counting Log*************\n")

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
        """Logs the current counts of inbound and outbound vehicles to the output file.
        This method writes the current counts to the output file along with a timestamp.
        """
        now = datetime.now().strftime("%H:%M:%S")
        self.output_file.write(f"\n************Update at {now}*************\n")
        self.output_file.write(f"Stats:\n{{\n"
                               f"Total vehicles: {self.curr_in_count + self.curr_out_count}\n"
                               f"Vehicles inbound: {self.curr_in_count}\n"
                               f"Vehicles outbound: {self.curr_out_count}\n"
                               f"}}\n")
        self.output_file.flush()

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

    def annotate_frame(self, frame, detections):
        """Annotates the frame with bounding boxes, labels, and traces for detected vehicles.
        Args:
            frame (numpy.ndarray): The current video frame to annotate.
            detections (sv.Detections): Detections of vehicles in the frame.
        Returns:
            numpy.ndarray: The annotated frame with bounding boxes, labels, and traces.
        """
        box_annotator = sv.BoundingBoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        trace_annotator = sv.TraceAnnotator()

        labels = [
            f"#{tracker_id} {self.class_names_dict[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, tracker_id, _ in detections
        ]

        frame = self.line_annotator.annotate(frame, self.count_line)
        annotated = box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
        annotated = trace_annotator.annotate(scene=annotated, detections=detections)
        return annotated

    def run(self):
        """Runs the vehicle detection and counting process.
        This method captures video frames, processes them for vehicle detection,
        annotates the frames, and displays the results in a window.
        It also logs the counts of inbound and outbound vehicles to the output file.
        """
        ret, frame = self.cap.read()

        while ret:
            detections = self.process_detections(frame)
            annotated_frame = self.annotate_frame(frame, detections)

            if (self.curr_in_count < self.count_line.in_count or
                self.curr_out_count < self.count_line.out_count):
                self.curr_in_count = self.count_line.in_count
                self.curr_out_count = self.count_line.out_count
                self.log_counts()

            # Show the current frame with annotations
            display = cv2.resize(annotated_frame, (1280, 960))
            cv2.imshow("Vehicle Detection", display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            ret, frame = self.cap.read()

        self.cleanup()

    def cleanup(self):
        """Cleans up resources after processing is complete.
        This method releases the video capture object, closes the output file,
        and destroys all OpenCV windows.
        """
        self.cap.release()
        self.output_file.close()
        cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description="Vehicle Detection and Counting")
    parser.add_argument("--vid", type=int, choices=[1, 2,3, 4], default=1, help="Video number to use for detection")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run the model on")
    parser.add_argument("--model", type=str, default="yolov8m.pt", choices=["yolov8m.pt", "yolov8s.pt", "yolov8n.pt", "yolov5su.pt"], help="Path to the YOLO model file")
    parser.add_argument("--output_txt", type=str, default="output.txt", help="Output text file name for logging")
    return parser.parse_args()

def main():
    args = parse_args()
    print("Starting vehicle counting...")
    counter = VehicleCounter(
        model_path=args.model,
        device_name=args.device,
        output_txt=args.output_txt,
        video_number=args.vid
    )
    counter.run()
    print("Stream finished.")
    print(f"Output saved to {args.output_txt}")

if __name__ == "__main__":
    main()
