# yolov5_pytorch_ros

## Ros Requirement
ROS Noetic Ninjemys (Recommanded)

or

ROS Melodic Morenia (Need to build cv_bridge(python3 version))



## Install requirement
```bash
# upgrade pip
python3 -m pip install --upgrade pip

cd yolov5_pytorch_ros/src/yolov5/
pip3 install -r requirements.txt
```

## Run in ROS Noetic Ninjemys

### Build and source
```bash
mkdir -p catkin_ws_pytorch_yolov5/src
cd catkin_ws_pytorch_yolov5/src
git clone https://github.com/Shua-Kang/ros_pytorch_yolov5.git
cd ..
catkin init
catkin build
source devel/setup.bash
```

### Launch yolov5_torch node in one terminal
```bash
roslaunch yolov5_torch detector.launch
```

### Rosbag play in another terminal
```bash
cd bag_dataset
rosbag play test_yolo.bag
```
## Run in ROS Melodic Morenia

### Build cv_bridge (Ros Melodic require) 
The default cv_bridge in ros melodic is python2 version.
Follow the instruction to build python3 version.
```bash
mkdir -p catkin_ws_cv_bridge/src
cd catkin_ws_cv_bridge/src
git clone https://github.com/ros-perception/vision_opencv.git
cd vision_opencv
# the newer version may require python>3.7
git checkout 1.13.0
cd ..
catkin init
# Instruct catkin to set cmake variables
catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
# Instruct catkin to install built packages into install place. It is $CATKIN_WORKSPACE/install folder
catkin config --install
catkin build
```
### Build and source
```bash
cd yolov5_pytorch_ros
catkin init
catkin build
source devel/setup.bash
cd catkin_ws_cv_bridge
# using --extend to avoid replace previous source
source install/setup.bash --extend
```

### Launch yolov5_torch node
```bash
roslaunch yolov5_torch detector.launch device:=cpu
#or launch with gpu device 0
roslaunch yolov5_torch detector.launch device:=0
# multi-gpu: roslaunch yolov5_torch detector.launch device:=0,1
```

### Rosbag play
```bash
cd bag_dataset
rosbag play test_yolo.bag
```
