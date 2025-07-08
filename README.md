# Vehicle Counting Demo

This project performs **real-time vehicle detection, tracking, and counting** using **YOLO models and ByteTrack** via the **Supervision** library. It supports both local video files and live video streams.

Counts of vehicles crossing a predefined line are displayed on screen and written to a log file (output.txt by default).

## Contents

   - main.py — main script for detection, tracking, and counting.

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
| `--model`      | Path to YOLO model                   | `yolov8m.pt`, `yolov8s.pt`, etc.  |
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

## Notes

- If you restart the script, the log file (output.txt by default) will be overwritten.

- For accurate counting, adjust the line coordinates in get_video_parameters() per your input video or stream.

- Confidence threshold for detections is set to 0.6.

## Credits

Test video files were taken from the https://www.kaggle.com/datasets/shawon10/road-traffic-video-monitoring dataset on Kaggle.