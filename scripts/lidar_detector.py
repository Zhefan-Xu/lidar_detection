#!/usr/bin/env python

import rospy
import numpy as np
import torch
import glob
from time import time
from pathlib import Path
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
import os
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import time
import ros_numpy


height_offset = -1.2 # move the origin 1m up

path_curr = os.path.dirname(__file__)
pointcloud_topic_name = "/ouster/points"
cfg_path = "cfg/pv_rcnn.yaml"
model_filename = "pv_rcnn_8369.pth"
# cfg_path = "cfg/pointpillar.yaml"
# model_filename = "pointpillar_7728.pth"
cfg_from_yaml_file(os.path.join(path_curr, cfg_path), cfg)


class DetectorDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.npy'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.length = 0

    def load_data(self, points):
        self.points = points
        self.length = 1

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        input_dict = {
            'points': self.points,
            'frame_id': 0,
        }
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

class lidar_detector:
    def __init__(self):
        self.pointcloud_received = False
        self.pointcloud_detected = False


        # load dataset
        self.detector_dataset = DetectorDataset(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False, root_path=Path(path_curr))

        # load model
        logger = common_utils.create_logger()
        self.model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=self.detector_dataset)
        self.model.load_params_from_file(filename=os.path.join(path_curr, model_filename), logger=logger, to_cpu=False)
        self.model.cuda()
        self.model.eval()
            
        # subscriber
        self.pc_sub = rospy.Subscriber(pointcloud_topic_name, PointCloud2, self.pointcloud_callback)
        
        # publisher
        self.bbox_pub = rospy.Publisher("lidar_detector/detected_bounding_boxes", MarkerArray, queue_size=10)



        # timer
        rospy.Timer(rospy.Duration(0.05), self.detect_callback)
        rospy.Timer(rospy.Duration(0.033), self.vis_callback)

    def pointcloud_callback(self, pointcloud):
        start_time = time.time()
        pc = ros_numpy.numpify(pointcloud)
        points_xyzi=np.zeros((pc.shape[0] * pc.shape[1],4))
        points_xyzi[:,0]= np.resize(pc['x'], pc.shape[0] * pc.shape[1])
        points_xyzi[:,1]= np.resize(pc['y'], pc.shape[0] * pc.shape[1])
        points_xyzi[:,2]= np.resize(pc['z']+height_offset, pc.shape[0] * pc.shape[1])
        points_xyzi[:,3]= 0
        # points_xyzi = np.array([[0, 0, 0, 0]])
        # gen = pc2.read_points(pointcloud, skip_nans=True)
        # int_data = list(gen)
        # start_time = time.time()

        # for x in int_data:
        #     points_xyzi = np.append(points_xyzi, [[x[0], x[1], x[2]+height_offset, x[3]/256.]], axis=0)

        self.curr_input_data = self.prepare_data(points_xyzi)
        self.pointcloud_received = True
        end_time = time.time()
        # print("data prepare time: ", end_time-start_time)
    def detect_callback(self, event):
        if (self.pointcloud_received):
            # print("start inference.")
            start_time = time.time()
            self.detection_results = self.inference(self.curr_input_data)
            self.pointcloud_detected = True 
            end_time = time.time()
            # print("Detection time: ", end_time-start_time)
            # print("finish inference")

    def vis_callback(self, event):
        if (self.pointcloud_detected):
            boxes_msg = self.get_bbox_msg(self.detection_results)
            self.bbox_pub.publish(boxes_msg)

    def get_bbox_msg(self, results):
        boxes_msg = MarkerArray()
        pred_bboxes = results[0]["pred_boxes"].cpu()
        marker_id = 0
        i = 0
        for box in pred_bboxes:
            if (results[0]["pred_labels"][i] == 2):
                cx = box[0]
                cy = box[1]
                cz = box[2] - height_offset
                lx = box[3]
                ly = box[4]
                lz = box[5]
                angle = box[6]
                # angle = 0

                # get eight points (unrotated)
                p1 = np.array([cx+lx/2., cy+ly/2., cz+lz/2.])
                p2 = np.array([cx+lx/2., cy-ly/2., cz+lz/2.])
                p3 = np.array([cx-lx/2., cy+ly/2., cz+lz/2.])
                p4 = np.array([cx-lx/2., cy-ly/2., cz+lz/2.])
                p5 = np.array([cx+lx/2., cy+ly/2., cz-lz/2.])
                p6 = np.array([cx+lx/2., cy-ly/2., cz-lz/2.])
                p7 = np.array([cx-lx/2., cy+ly/2., cz-lz/2.])
                p8 = np.array([cx-lx/2., cy-ly/2., cz-lz/2.])

                # rotation matrix
                R = np.array([[np.cos(angle), -np.sin(angle), 0],
                              [np.sin(angle), np.cos(angle), 0],
                              [0, 0, 1]
                            ])

                center = np.array([cx, cy, cz])
                p1r = R @ (p1 -  center) + center
                p2r = R @ (p2 -  center) + center
                p3r = R @ (p3 -  center) + center
                p4r = R @ (p4 -  center) + center
                p5r = R @ (p5 -  center) + center
                p6r = R @ (p6 -  center) + center
                p7r = R @ (p7 -  center) + center
                p8r = R @ (p8 -  center) + center

                l1 = self.make_line_msg(p1r, p2r, marker_id)
                marker_id += 1
                l2 = self.make_line_msg(p1r, p3r, marker_id)
                marker_id += 1
                l3 = self.make_line_msg(p2r, p4r, marker_id)
                marker_id += 1
                l4 = self.make_line_msg(p3r, p4r, marker_id)
                marker_id += 1
                l5 = self.make_line_msg(p1r, p5r, marker_id)
                marker_id += 1
                l6 = self.make_line_msg(p2r, p6r, marker_id)
                marker_id += 1
                l7 = self.make_line_msg(p3r, p7r, marker_id)
                marker_id += 1
                l8 = self.make_line_msg(p4r, p8r, marker_id)
                marker_id += 1
                l9 = self.make_line_msg(p5r, p6r, marker_id)
                marker_id += 1
                l10 = self.make_line_msg(p5r, p7r, marker_id)
                marker_id += 1
                l11 = self.make_line_msg(p6r, p8r, marker_id)
                marker_id += 1
                l12 = self.make_line_msg(p7r, p8r, marker_id)
                marker_id += 1
                boxes_msg.markers.append(l1)
                boxes_msg.markers.append(l2)
                boxes_msg.markers.append(l3)
                boxes_msg.markers.append(l4)
                boxes_msg.markers.append(l5)
                boxes_msg.markers.append(l6)
                boxes_msg.markers.append(l7)
                boxes_msg.markers.append(l8)
                boxes_msg.markers.append(l9)
                boxes_msg.markers.append(l10)
                boxes_msg.markers.append(l11)
                boxes_msg.markers.append(l12)
                i+=1
        return boxes_msg

    def make_line_msg(self, p1, p2, marker_id):
        line_msg = Marker()
        x1 = p1[0]
        y1 = p1[1]
        z1 = p1[2]
        
        x2 = p2[0]
        y2 = p2[1]
        z2 = p2[2]

        p1p = Point()
        p1p.x = x1
        p1p.y = y1
        p1p.z = z1
        p2p = Point()
        p2p.x = x2
        p2p.y = y2
        p2p.z = z2      
        
        line_msg.points.append(p1p)
        line_msg.points.append(p2p)

        line_msg.header.frame_id = "os_sensor"
        line_msg.id = marker_id
        line_msg.type = line_msg.LINE_LIST
        line_msg.action = line_msg.MODIFY
        line_msg.scale.x = 0.2
        line_msg.scale.y = 0.2
        line_msg.scale.z = 0.2
        line_msg.color.a = 1.0
        line_msg.color.r = 0.0
        line_msg.color.g = 1.0
        line_msg.color.b = 1.0
        line_msg.pose.orientation.w = 1.0
        line_msg.lifetime = rospy.Time(0.1)
        return line_msg


    # prepare model inputs
    def prepare_data(self, points):
        self.detector_dataset.load_data(points)
        curr_input_data = self.detector_dataset[0]
        curr_input_data = self.detector_dataset.collate_batch([curr_input_data])
        return curr_input_data

    def inference(self, inputs):
        with torch.no_grad():
            load_data_to_gpu(inputs)
            detection_results, _ = self.model.forward(inputs)
        return detection_results