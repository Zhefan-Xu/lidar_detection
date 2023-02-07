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
from visualization_msg import MarkerArray

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
			self.inference(self.curr_input_data)
			self.pointcloud_detected = True	

	def vis_callback(self, event):
		pass

	# prepare model inputs
	def prepare_data(self, points):
		self.detector_dataset.load_data(points)
		curr_input_data = self.detector_dataset[0]
		curr_input_data = self.detector_dataset.collate_batch([curr_input_data])
		return curr_input_data

	def inference(self, inputs):
		self.detection_results, _ = self.model.forward(load_data_to_gpu(inputs))
