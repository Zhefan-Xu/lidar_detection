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
from visualization_msg.msg import MarkerArray
from visualization_msg.msg import Marker
from geometry_msg.msg import Point

height_offset = -1.0 # move the origin 1m up

path_curr = os.path.dirname(__file__)
pointcloud_topic_name = "/ouster/points"
# cfg_path = "cfg/pv_rcnn.yaml"
# model_filename = "pv_rcnn_8369.pth"
cfg_path = "cfg/pointpillar.yaml"
model_filename = "pointpillar_7728.pth"
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
	 	self.model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
	    self.model.load_params_from_file(filename=os.path.join(path_curr, model_filename), logger=logger, to_cpu=False)
	    self.model.cuda()
	    self.model.eval()
			
		# subscriber
		self.pc_sub = rospy.Subscriber(pointcloud_topic_name, PointCloud2, self.pointcloud_callback)
		
		# publisher
		self.bbox_pub = rospy.Publisher("lidar_detector/detected_bounding_boxes", MarkerArray, queue_size=10)



		# timer
		rospy.Timer(rospy.Duration(0.1). self.detect_callback)
		rospy.Timer(rospy.Duration(0.033), self.vis_callback)

	def pointcloud_callback(self, pointcloud):
		points_xyzi = np.array([[0, 0, 0, 0]])
		gen = pc2.read_points(pointcloud, skip_nans=True)
		int_data = list(pointcloud)
		for x in int_data:
			points_xyzi = np.append(xyzi, [[x[0]+5, x[1], x[2]-1.0, x[3]/256.]], axis=0)

		self.curr_input_data = self.prepare_data(points_xyzi)
		self.pointcloud_received = True

	def detect_callback(self, event):
		if (self.pointcloud_received):
			self.detection_results = self.inference(self.curr_input_data)
			self.pointcloud_detected = True	

	def vis_callback(self, event):
		if (self.pointcloud_detected):
			boxes_msg = self.get_bbox_msg(self.detection_results)
			self.bbox_pub.publish(boxes_msg)

	def get_bbox_msg(self, results):
		boxes_msg = MarkerArray()
		pred_bboxes = results["pred_boxes"]
		for box in pred_bboxes:
			cx = box[0]
			cy = box[1]
			cz = box[2]
			lx = box[3]
			ly = box[4]
			lz = box[5]
			angle = box[6]

			# get eight points (unrotated)
			p1 = np.array([cx+lx/2., cy+ly/2., cz+lz/2.])
			p2 = np.array([cx+lx/2., cy-ly/2., cz+lz/2.])
			p3 = np.array([cx-lx/2., cy+ly/2., cz+lz/2.])
			p4 = np.array([cx-lx/2., cy-ly/2., cz+lz/2.])
			p5 = np.array([cx+lx/2., cy+ly/2., cz-lz/2.])
			p6 = np.array([cx+lx/2., cy-ly/2., cz-lz/2.])
			p7 = np.array([cx-lx/2., cy+ly/2., cz-lz/2.])
			p5 = np.array([cx-lx/2., cy+ly/2., cz-lz/2.])

			# rotation matrix
			R = np.array([[cos(angle), -sin(angle), 0],
				          [sin(angle), cos(angle), 0],
				          [0, 0, 1]
				        ])

			p1r = R @ p1
			p2r = R @ p2
			p3r = R @ p3
			p4r = R @ p4
			p5r = R @ p5
			p6r = R @ p6
			p7r = R @ p7
			p8r = R @ p8

			l1 = self.make_line_msg(p1r, p2r)
			l2 = self.make_line_msg(p1r, p3r)
			l3 = self.make_line_msg(p2r, p4r)
			l4 = self.make_line_msg(p3r, p4r)
			l5 = self.make_line_msg(p1r, p5r)
			l6 = self.make_line_msg(p2r, p6r)
			l7 = self.make_line_msg(p3r, p7r)
			l8 = self.make_line_msg(p4r, p8r)
			l9 = self.make_line_msg(p5r, p6r)
			l10 = self.make_line_msg(p5r, p7r)
			l11 = self.make_line_msg(p6r, p8r)
			l12 = self.make_line_msg(p7r, p8r)
			boxes_msg.markers.push_back(l1)
			boxes_msg.markers.push_back(l2)
			boxes_msg.markers.push_back(l3)
			boxes_msg.markers.push_back(l4)
			boxes_msg.markers.push_back(l5)
			boxes_msg.markers.push_back(l6)
			boxes_msg.markers.push_back(l7)
			boxes_msg.markers.push_back(l8)
			boxes_msg.markers.push_back(l9)
			boxes_msg.markers.push_back(l10)
			boxes_msg.markers.push_back(l11)
			boxes_msg.markers.push_back(l12)
		return boxes_msg

	def make_line_msg(self, p1, p2):
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
		
		line_msg.points.push_back(p1p)
		line_msg.points.push_back(p2p)

		marker.header.frame_id = "/lidar_detection_frame"
		marker.type = marker.LINE_LIST
		marker.action = marker.ADD
		marker.scale.x = 0.2
		marker.scale.y = 0.2
		marker.scale.z = 0.2
		marker.color.a = 1.0
		marker.color.r = 1.0
		marker.color.g = 0.0
		marker.color.b = 1.0
		marker.pose.orientation.w = 1.0
		return line_msg


	# prepare model inputs
	def prepare_data(self, points):
		self.detector_dataset.load_data(points)
		curr_input_data = self.detector_dataset[0]
		curr_input_data = self.detector_dataset.collate_batch([curr_input_data])
		return curr_input_data

	def inference(self, inputs):
		detection_results, _ = self.model.forward(load_data_to_gpu(inputs))
		return detection_results