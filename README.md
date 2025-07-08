# Vehicle Counting Demo

This project performs **real-time vehicle detection, tracking, and counting** using **YOLO models and ByteTrack** via a self modified and improved  **Supervision** library. It supports both local video files and live video streams.

The Supervision Line Zone logic was specifically improved for suboptimal video feed conditions — such as low frame rate, occlusions, and vehicles appearing too close or too far from the camera — where consistent per-frame detection cannot be guaranteed. The goal is to ensure robust vehicle counting even when object detections are intermittently lost or delayed.

Counts of vehicles crossing a predefined line are displayed on screen and written to a log file (output.txt by default).

## Contents

   - main.py — main script for detection, tracking, and counting.
  
   - Modified Supervision package (supervision/supervision/detection/line_zone.py)

   - Uses YOLOv8 or YOLOv5 for object detection.

   - Uses ByteTrack (via Supervision) for multi-object tracking.

   - Annotates frames with bounding boxes, labels, tracks, and line-crossing stats.

## Dependencies

Install all required packages in a virtual environment with:

```sh
pip install -r requirements.txt

```
## Usage

run the script using

```sh
python main.py --vid 1 --device cuda --model yolov8m.pt --output_txt output.txt
```

### Command line arguments
| Argument       | Description                          | Options/Default                   |
| -------------- | ------------------------------------ | --------------------------------- |
| `--vid`        | Video source to use (see list below) | `1`, `2`, `3`, `4` (default: `1`) |
| `--device`     | Device to run YOLO model on          | `cuda`, `cpu` (default: `cuda`)   |
| `--model`      | Path to YOLO v8 or v5 model                   | `yolov8m.pt`, `yolov5su.pt`, etc.  |
| `--output_txt` | File to write vehicle count logs     | `output.txt`                      |

## Video Sources

You can choose between four video inputs with the --vid argument:

| ID | Source         | Description                                                        |
| -- | -------------- | ------------------------------------------------------------------ |
| 1  | `videos/1.mp4` | Local test high quality video                                 |
| 2  | `videos/2.avi` | Local high quality test video                              |
| 3  | Traffic live stream    | Publicly available [`NYSDOT` RTSP stream](https://s58.nysdot.skyvdn.com)              |
| 4  | Traffic live stream    | Publicly available [`MJPEG stream`](http://80.151.142.110:8080/?action=stream) |

 ## Features

- Line-based counting: Vehicles are counted once they cross a predefined virtual line.

- YOLO object detection: Fast and accurate bounding box predictions.

- ByteTrack tracking: Maintains consistent tracking of objects across frames.

- Output logging: Vehicle counts are logged periodically to a file.

- Visual feedback: Bounding boxes, tracks, and counts are rendered on the video.

## Log Output Example

```txt

************Vehicle Counting Log*************

************Update at 14:32:16*************
Stats:
{
Total vehicles: 21
Vehicles inbound: 10
Vehicles outbound: 11
}

```

Logs are updated only when new vehicles are detected crossing the counting line.

## Key design objectives in modifying line counting in Supervision package

1) Partial exit handling: A vehicle is counted even if its detection disappears before fully crossing the line, as long as the majority of its bounding box area has clearly crossed from one side of the line to the other. This accounts for cases where tracking or detection fails during the exit.

2) Late detection handling: A vehicle is counted even if its first detection appears already straddling or partially over the line. This accommodates scenarios where detection starts late — i.e., after the vehicle has already begun crossing — so the system doesn't miss entries due to delayed recognition.

3) Skipped frame resilience: The system is designed to tolerate skipped or missing frames, ensuring that vehicle transitions across the line are inferred from tracked positions before and after the missing frames. This helps maintain count accuracy even under severe frame drops or intermittent detection losses.

This behavior is implemented by extending and modifying classes from the supervision package, which is integrated via a Git reference in the `requirements.txt`:

```
-e git+https://github.com/roboflow/supervision.git@8599345a3c00c26e7a05d7c81e23bf1d5cf8b8e2#egg=supervision

```


## Notes

- If you restart the script, the log file (output.txt by default) will be overwritten.

- For accurate counting, adjust the line coordinates in get_video_parameters() per your input video or stream.

- Confidence threshold for detections is set to 0.6.

## Credits

Test video files were taken from the https://www.kaggle.com/datasets/shawon10/road-traffic-video-monitoring dataset on Kaggle.