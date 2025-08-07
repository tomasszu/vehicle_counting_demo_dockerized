# Vehicle Counting Demo Dockerized

This is a repository for the dockerization of the original vehicle detection and counting project. Dockerization is available both for running the container on Ubuntu and Nvidia Jetson.

The project original description and usage is available at: [Vehicle Counting Demo Repo](https://github.com/tomasszu/vehicle_counting_demo?tab=readme-ov-file#vehicle-counting-demo)

The Jetsion container version slightly differs from the original project in that it has no GUI video display of the counting results. The counting results can been acquired from the logs which will be passed from the container to the host via volume mounting. The resulting logs will be available in the output.txt (by default) file in the folder of your choosing e.g. ~/counting/logs. More on this in [Usage](#usage)

> [!IMPORTANT]
>  A Jetson docker container version that sends the vehicle counting results over MQTT is available on the [jetson_mqtt](https://github.com/tomasszu/vehicle_counting_demo_dockerized/tree/jetson_mqtt#vehicle-counting-demo-jetson-mqtt-branch) branch of this project.</font>

## Contents

This repo includes:

- The original project code
- Dockerization for **Ubuntu** launch-case
  - Dockerfile
  - Docker image [vehiclecountingdemodockerized:ubuntu_latest](https://github.com/tomasszu/vehicle_counting_demo_dockerized/pkgs/container/vehiclecountingdemodockerized)
- Dockerization for Nvidia **Jetson** launch-case
  - Dockerfile
  - Docker image [vehiclecountingdemodockerized:jetson_latest](https://github.com/tomasszu/vehicle_counting_demo_dockerized/pkgs/container/vehiclecountingdemodockerized)


## Dependencies

All the dependencies exist within the Docker image, however, support for NVIDIA GPU usage must be present on the device to use CUDA device GPU functionality. The Jetson images have been tested on Jetson Orin exclusively

## Usage

### Ubuntu version

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
    --name vehicle_counting \
    ghcr.io/tomasszu/vehiclecountingdemodockerized:jetson_ubuntu

```

(Remove the `--gpus` option if you plan to run it on CPU)

3. Additional parameters can be passed like this:

The additional parameters are passed on the last line after container name. The example contains the default values that are picked when ommiting additional parameter execution.

```sh 
docker run --rm -it \
    --gpus all \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --name vehicle_counting \
    ghcr.io/tomasszu/vehiclecountingdemodockerized:jetson_ubuntu \
    python3 main.py --vid 1 --device cuda
```

### Jetson version

0. Download the Docker image ( available on the right under the section "Packages")

1. Create the directory you'd like the logs to be saved in e.g. `~/counting/logs`

```sh

mkdir ~/counting/logs

```

2. Run the Docker container by passing the mounted volume folder of your choice

```sh

sudo docker run -it --net=host --runtime nvidia \
    -v ~/counting/logs:/app/logs \
    --name vehicle_counting \
    ghcr.io/tomasszu/vehiclecountingdemodockerized:jetson_latest

```

3. Additional parameters can be passed like this:

The additional parameters are passed on the last line after container name. The example contains the default values that are picked when ommiting additional parameter execution.

```sh 
sudo docker run -it --net=host --runtime nvidia \
    -v ~/counting/logs:/app/logs \
    --name vehicle_counting \
    ghcr.io/tomasszu/vehiclecountingdemodockerized:jetson_latest \
    python3 main_jetson.py --vid 1 --device cuda
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