## Ros Requirement
ROS Noetic Ninjemys (Recommanded)

or

ROS Melodic Morenia (Need to build cv_bridge(python3 version))



## Clone and Install requirement
```bash
mkdir -p catkin_ws_pytorch_yolov5/src
cd catkin_ws_pytorch_yolov5/src
git clone https://github.com/Shua-Kang/ros_pytorch_yolov5.git

# If pytorch gpu is needed, install pytorch with gpu firstly. See https://pytorch.org/get-started/locally/
# Upgrade pip and install requirement
cd ros_pytorch_yolov5/yolov5/
python3 -m pip install --upgrade pip
pip3 install -r requirements.txt
```

## Build in ROS Noetic Ninjemys

### Build yolov5 pytorch
```bash
cd catkin_ws_pytorch_yolov5
catkin init
catkin build
```


## Build in ROS Melodic Morenia

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
# If it has opencv version error, replace opencv4 with opencv3 by pip. See [issue3](https://github.com/Shua-Kang/ros_pytorch_yolov5/issues/3) for more help
```
### Build yolov5 pytorch
```bash
cd ros_pytorch_yolov5
catkin init
catkin build
```

## Start

### (For ROS Noetic) Launch yolov5_torch Node in One Terminal
```bash
cd catkin_ws_pytorch_yolov5
source devel/setup.bash
roslaunch ros_pytorch_yolov5 detector.launch device:=cpu view_img:=True
#or launch with gpu device 0
roslaunch yolov5_torch detector.launch device:=0 view_img:=True
# multi-gpu: roslaunch yolov5_torch detector.launch device:=0,1
```
### (For ROS Melodic) Launch yolov5_torch Node in One Terminal
```bash
cd catkin_ws_pytorch_yolov5
source devel/setup.bash
cd catkin_ws_cv_bridge
# using --extend to avoid replace previous source
source install/setup.bash --extend

roslaunch ros_pytorch_yolov5 detector.launch device:=cpu view_img:=True
#or launch with gpu device 0
roslaunch ros_pytorch_yolov5 detector.launch device:=0 view_img:=True
# multi-gpu: roslaunch yolov5_torch detector.launch device:=0,1
```


### Rosbag Play in Another Terminal
```bash
cd catkin_ws_pytorch_yolov5
source devel/setup.bash
wget https://star-center.shanghaitech.edu.cn/seafhttp/files/54f59ac6-1607-4f17-9626-9f3fca01c95a/Normal.bag -O test.bag
rosbag play test.bag
```
