# Vehicle Counting Demo Jetson MQTT branch

This is a repository for the dockerization of the original vehicle detection and counting project. On this branch dockerization is available running the container on Nvidia Jetson and outputting the counting logs via mqtt.

The project original description and usage is available at: [Vehicle Counting Demo Repo](https://github.com/tomasszu/vehicle_counting_demo?tab=readme-ov-file#vehicle-counting-demo)

The Jetsion container version slightly differs from the original project in that it has no GUI video display of the counting results. The counting results will be forwarded via MQTT messages to the selected broker and topic. More on this in [Usage](#usage).

## Contents

This repo includes:

- The original project code
- Dockerization for Nvidia **Jetson** device
  - Dockerfile
  - Docker image [vehiclecountingdemodockerized:jetson_mqtt_latest](https://github.com/tomasszu/vehicle_counting_demo_dockerized/pkgs/container/vehiclecountingdemodockerized)


## Dependencies

All the dependencies exist within the Docker image, however, support for NVIDIA GPU usage must be present on the device to use CUDA device GPU functionality. The Jetson images have been tested on Jetson Orin exclusively. An MQTT broker needs to be accessable for receiving the messages.

## Usage


1. Download the Docker image ( available on the right under the section "Packages")

2. Run the Docker container by selecting your mqtt broker details

```sh

sudo docker run -it --runtime nvidia \
  --name vehicle_counting  \
  ghcr.io/tomasszu/vehiclecountingdemodockerized:jetson_mqtt_latest \
  python3 main_jetson_mqtt.py --mqtt_broker=test.mosquitto.org --mqtt_port=1883 --mqtt_topic="counting/logs"

```

3. Additional parameters can be passed like this:

The additional parameters are passed on the last line after container name. The example contains the default values for parameters like `--vid` and `--device`.

```sh 
sudo docker run -it --net=host --runtime nvidia \
    --name vehicle_counting \
    ghcr.io/tomasszu/vehiclecountingdemodockerized:jetson_mqtt_latest \
    python3 main_jetson_mqtt.py --mqtt_broker=test.mosquitto.org --mqtt_port=1883 --mqtt_topic="counting/logs" --vid 1 --device cuda
```

### Default test usage

1. Download the Docker image ( available on the right under the section "Packages")
   
2. Create a docker network

```sh

docker network create mqtt_test_net

```

3. Run a test mosquitto broker (with a .conf file of your liking - in this case allowing annonimous connections and exposing port 1883)

```sh

docker run -d   --name mosquitto_test \
   --network mqtt_test_net   \
   -p 1885:1883   \
   -v "$PWD/mosquitto.conf":/mosquitto/config/mosquitto.conf    \
   eclipse-mosquitto

```

4. Run a temporary SUB listening container with counting/logs topic

```sh

docker run -it --rm --network mqtt_test_net eclipse-mosquitto  \
 mosquitto_sub --verbose -h mosquitto_test -p 1883 -t counting/logs


```

5. Run the vehilce counting container with the default options

```sh

sudo docker run -it --net=mqtt_test_net --runtime nvidia \
  --name vehicle_counting  \
  ghcr.io/tomasszu/vehiclecountingdemodockerized:jetson_mqtt_latest


```



### Other command line arguments
| Argument       | Description                          | Options/Default                   |
| -------------- | ------------------------------------ | --------------------------------- |
| `--vid`        | Video source to use (see list below) | `1`, `2`, `3`, `4` (default: `1`) |
| `--device`     | Device to run YOLO model on          | `cuda`, `cpu` (default: `cuda`)   |
| `--model`      | Path to YOLO v8 or v5 model                   | `yolov8m.pt`, `yolov5su.pt`, etc.  |

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