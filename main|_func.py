import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np
from datetime import datetime
import time
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Vehicle Detection and Counting")
    parser.add_argument("--vid", type=int, choices=[1, 2], default=1, help="Video number to use for detection")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda","cpu"] , help="Device to run the model on (cpu or cuda)")
    parser.add_argument("--model", type=str, default="yolov8m.pt", choices=["yolov8m.pt", "yolov8s.pt", "yolov8n.pt", "yolov5su.pt"], help="Path to the YOLO model file")
    parser.add_argument("--output_txt", type=str, default="output.txt", help="Output text file name for logging")
    return parser.parse_args()

def printout(incount, outcount, f):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    f.write(("\n************Update at " + current_time + "*************\n"))
    f.write("Stats:" +
            "{\n" +
            "Total vehicles: "+ str(incount+outcount) + "\n" +
            "Vehicles inbound: "+ str(incount) + "\n" +
            "Vehicles outbound: "+ str(outcount) + "\n" +
            #"Vehicles with unknown direction: "+ str(unknown_vehicles) + "\n" +
            "}\n")
    f.flush()

def calculate_center(detection):
    x_center = (detection[0] + detection[2]) / 2.0
    y_center = (detection[1] + detection[3]) / 2.0
    #print(x_center, y_center)
    return [x_center, y_center]


def detections_process(model, frame, tracker, count_line, class_ids):
    confidence_threshold = 0.6

    results = model(frame)[0]

    detections = sv.Detections.from_ultralytics(results)

    detections = detections[np.isin(detections.class_id, class_ids)]
    detections = detections[np.greater(detections.confidence, confidence_threshold)]
    detections = tracker.update_with_detections(detections)

    count_line.trigger(detections)

    return detections

def frame_annotations(detections, frame, class_names_dict, line_annotator, count_line):

    box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    trace_annotator = sv.TraceAnnotator()

    # format custom labels
    labels = [
        f"#{tracker_id} {class_names_dict[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, tracker_id, _
        in detections
    ]

    frame = line_annotator.annotate(frame, count_line)

    annotated_frame = box_annotator.annotate(
        scene=frame.copy(), detections=detections
    )


    annotated_labeled_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections, labels=labels
    )


    annotated_labeled_frame = trace_annotator.annotate(
        scene=annotated_labeled_frame,
        detections=detections
    )

    return annotated_labeled_frame


def get_vid_parameters(video_number):
    if video_number == 1:
        # Define the start and end points for the line counting zone
        # These points should be adjusted based on the area where vehicles need to be counted
        start, end = sv.Point(x=150, y=300), sv.Point(x=1190, y=530)
        cap = cv2.VideoCapture(f'videos/1.mp4')
    elif video_number == 2:
        start, end = sv.Point(x=150, y=450), sv.Point(x=1200, y=450)
        cap = cv2.VideoCapture(f'videos/2.avi')
    else:
        raise ValueError("Invalid video number provided.")
    return start, end, cap

def get_device(device_name):
    if device_name not in ["cuda", "cpu"]:
        raise ValueError("Invalid device name. Choose 'cuda' or 'cpu'.")
    if device_name == "cuda" and torch.cuda.is_available():
        print("CUDA is available, using GPU for processing.")
        return "cuda"
    elif device_name == "cpu":
        print("CUDA is not available or chosen CPU instead.")
        return "cpu"

def main(args):

    video_number = args.vid
    # Load the YOLO model
    model_path = args.model
    device_name = args.device
    output_txt = args.output_txt

    model = YOLO(model_path)
    model.to(device=get_device(device_name))

    tracker = sv.ByteTrack()

    f = open(output_txt, "w")
    f.write("************Vehicle Counting Log*************\n")


    # dict maping class_id to class_name
    class_names_dict = model.model.names
    # class_ids of interest - car, motorcycle, bus and truck
    class_ids = [2, 3, 5, 7]

    video_number = args.vid

    start, end, cap = get_vid_parameters(video_number)


    count_line = sv.LineZone(start=start, end=end)

    line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=0.5)

    curr_in_count = 0
    curr_out_count = 0

    ret, frame = cap.read()

    # # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can use other codecs like 'MJPG' or 'MP4V'
    # out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1280, 960))



    while ret:
        detections = detections_process(model, frame, tracker, count_line, class_ids)

        annotated_frame = frame_annotations(detections, frame, class_names_dict, line_annotator, count_line)

        if (curr_in_count < count_line.in_count or curr_out_count < count_line.out_count):
            curr_in_count = count_line.in_count
            curr_out_count = count_line.out_count
            printout(curr_in_count, curr_out_count, f)

        display = annotated_frame
        #out.write(display)
        display = cv2.resize(display, (1280, 960))
        cv2.imshow("Vehicle Detection", display)
        if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        ret, frame = cap.read()

    
    cv2.destroyAllWindows()
    cap.release()
    f.close()

    # out.release()

if __name__ == "__main__":
    args = parse_args()
    print("Starting vehicle detection...")
    main(args)
    print("Done!")
    print("Output saved to output.avi and output.txt")
