#!/usr/bin/env python3

import argparse
import time
from pathlib import Path
import rospy
import std_msgs.msg
from rospkg import RosPack
from std_msgs.msg import UInt8
from sensor_msgs.msg import Image
from geometry_msgs.msg import Polygon, Point32
from ros_pytorch_yolov5.msg import BoundingBox, BoundingBoxes
from cv_bridge import CvBridge, CvBridgeError
from skimage.transform import resize

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_sync
from utils.datasets import letterbox

# Deep learning imports
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import os
package = RosPack()
package_path = package.get_path('ros_pytorch_yolov5')

class detectManager:
    def __init__(self):
        # print("\ndetect init\n")
        self.weights = rospy.get_param('~weights')
        self.source = rospy.get_param('~source')
        self.view_img = rospy.get_param('~view_img')
        self.save_txt = rospy.get_param('~save_txt')
        self.save_txt = False
        self.img_size = rospy.get_param('~img_size')
        self.name = rospy.get_param('~name')
        self.exist_ok = rospy.get_param('~exist_ok')
        self.project = rospy.get_param('~project')
        self.device = str(rospy.get_param('~device'))
        self.device = select_device(self.device)
        self.augment = rospy.get_param('~augment')
        self.iou_thres = rospy.get_param('~iou_thres')
        if(self.device.type!="cpu"):
            self.half = True
        else:
            self.half = False
        if(rospy.get_param('~classes') == 'None'):
            self.classes = None
        else:
            self.classes = rospy.get_param('~classes')
        self.agnostic_nms = rospy.get_param('~agnostic_nms')
        self.conf_thres = rospy.get_param('~conf_thres')
        self.save_conf = rospy.get_param('~save_conf')

        # Initialize width and height
        self.h = 0
        self.w = 0
        
        self.model = ''

        # Load other parameters
        self.gpu_id = rospy.get_param('~gpu_id', 0)
        self.network_img_size = rospy.get_param('~img_size', 416)
        self.publish_image = rospy.get_param('~publish_image')

        self.image_topic = rospy.get_param('~image_topic')

        # Load CvBridge
        self.bridge = CvBridge()
        # Load publisher topics
        self.detected_objects_topic = rospy.get_param('~detected_objects_topic')
        self.published_image_topic = rospy.get_param('~detections_image_topic')

        # Define subscribers
        self.image_sub = rospy.Subscriber(
            self.image_topic, Image, self.imageCb, queue_size=1, buff_size=2**24)

        # Define publishers
        self.pub_ = rospy.Publisher(
            self.detected_objects_topic, BoundingBoxes, queue_size=10)
        self.pub_viz_ = rospy.Publisher(
            self.published_image_topic, Image, queue_size=10)
        rospy.loginfo("Launched node for object detection")
        self.path = package_path
        # Spin
        self.warmup()
        rospy.spin()

    def imageCb(self, data):
        # Convert the image to OpenCV
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")
        except CvBridgeError as e:
            print(e) 
    
        #a = input()
        # Initialize detection results
        detection_results = BoundingBoxes()
        detection_results.header = data.header
        detection_results.image_header = data.header

        # Configure input
        input_img = self.imagePreProcessing(self.cv_image)
        input_img = Variable(input_img.type(torch.FloatTensor))

        # Get detections from network
        with torch.no_grad():
            detections = self.detect(self.cv_image, data)
        return 0


    def imagePreProcessing(self, img):
        # Extract image and shape
        img = np.copy(img)
        img = img.astype(float)
        height, width, channels = img.shape

        if (height != self.h) or (width != self.w):
            self.h = height
            self.w = width

            # Determine image to be used
            self.padded_image = np.zeros(
                (max(self.h, self.w), max(self.h, self.w), channels)).astype(float)

        # Add padding
        if (self.w > self.h):
            self.padded_image[(self.w-self.h)//2: self.h +
                              (self.w-self.h)//2, :, :] = img
        else:
            self.padded_image[:, (self.h-self.w) //
                              2: self.w + (self.h-self.w)//2, :] = img

        # Resize and normalize
        input_img = resize(
            self.padded_image, (self.network_img_size, self.network_img_size, 3))/255.

        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))

        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()
        input_img = input_img[None]

        return input_img

    def warmup(self):
        self.weights = os.path.join(package_path, 'yolov5/weights', self.weights)
        print("Load model: " + self.weights)
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        if self.half:
            self.model.half()  # to FP16


    def detect(self, opencv_img, data, save_img=False):
        
        self.source = os.path.join(package_path,'yolov5', self.source)
        # print(self.weights)
        source, weights, view_img, save_txt, imgsz = self.source, self.weights, self.view_img, self.save_txt, self.img_size
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://'))
        self.project = os.path.join(package_path,'yolov5', self.project)
        # Directories
        save_dir = Path(increment_path(Path(self.project) / self.name,
                        exist_ok=self.exist_ok))  # increment run
        
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True,
                                                            exist_ok=True)  # make dir

        # Initialize
        set_logging()
        device = self.device
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # print(os.getcwd())
        # Load model
        model = self.model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size


        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load(
                'weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        else:
            #save_img = True
            save_img = False
            dataset = LoadImages(source, img_size=imgsz, stride=stride)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
                next(model.parameters())))  # run once
        t0 = time.time()

        # path = r"/workspace/yolov5/data/images/bus.jpg"
        vid_cap = None
        #im0s = cv2.imread(path)
        im0s = opencv_img
        img = letterbox(im0s, 640, stride=stride)[0]
        # Convert
        img = img.transpose(2, 0, 1) # to NCHW
        img = np.ascontiguousarray(img)
        # img = cv2.imread("")

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t1 = time_sync()

        pred = model(img, augment=False)[0]
        # Apply NMS
        pred = non_max_suppression(
            pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
        t2 = time_sync()
        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        # Process detections
        detection_results = BoundingBoxes()
        detection_results.header = data.header
        detection_results.image_header = data.header

        for i, det in enumerate(pred):  # detections per image

            im0 = im0s
            p = self.path
            s = ''
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
                xmin, ymin, xmax, ymax, conf, det_class = det[0]
                detection_msg = BoundingBox()
                detection_msg.xmin = int(xmin.item())
                detection_msg.xmax = int(xmax.item())
                detection_msg.ymin = int(ymin.item())
                detection_msg.ymax = int(ymax.item())
                detection_msg.probability = conf.item()
                detection_msg.Class = names[int(det_class)]
                detection_results.bounding_boxes.append(detection_msg)

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label,
                                color=colors[int(cls)], line_thickness=3)
            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
            # Stream results
            if view_img:
                im_bgr = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)
                cv2.imshow(str(p), im_bgr)
                cv2.waitKey(1)  # 1 millisecond
            img_msg = self.bridge.cv2_to_imgmsg(im0, encoding="rgb8")
            self.pub_viz_.publish(img_msg)
        self.pub_.publish(detection_results)


        print(f'Done. ({time.time() - t0:.3f}s)')




if __name__ == '__main__':
    rospy.init_node("detector_manager_node")
    rospy.loginfo("start detect node")
    dm = detectManager()
