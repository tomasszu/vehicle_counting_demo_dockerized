# Vehicle Counting Demo

This is a repository for the dockerization of the original vehicle detection and counting project.

The project original description and usage is available at: [Vehicle Counting Demo Repo](https://github.com/tomasszu/vehicle_counting_demo?tab=readme-ov-file#vehicle-counting-demo)

## Contents

This repo includes:

- The original project code
- Dockerfile for the ubuntu image
- Docker image to launch on Ubuntu [vehiclecountingdemodockerized:ubuntu_latest](https://github.com/tomasszu/vehicle_counting_demo_dockerized/pkgs/container/vehiclecountingdemodockerized)

## Dependencies

All the dependencies exist within the Docker image, however, support for NVIDIA GPU usage must be present on the device to use CUDA device GPU functionality.

## Usage

0. Download the Docker image ( available on the right under the section "Packages")

1. Allow Docker to run GPU apps

```sh
xhost +local:docker  # allow Docker GUI apps
```

2. Run the Docker container

```sh
docker run --rm -it \
    --gpus all \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    vehicle-counter

```

(Remove the `--gpus` option if you plan to run it on CPU)

3. Additional parameters can be passed like this:

The additional parameters are passed on the last line after container name. The example contains the default values that are picked when ommiting additional parameter execution.

```sh 
docker run --rm -it \
    --gpus all \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    vehicle-counter \
    python3 main.py --vid 1 --device cuda
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


## Credits

Test video files were taken from the https://www.kaggle.com/datasets/shawon10/road-traffic-video-monitoring dataset on Kaggle.