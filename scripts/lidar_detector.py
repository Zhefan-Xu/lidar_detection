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
from geometry_msgs.msg import Point, Pose, Vector3, PoseStamped, Quaternion
from vision_msgs.msg import BoundingBox3D, BoundingBox3DArray
from nav_msgs.msg import Odometry
import time
import ros_numpy
from tf.transformations import quaternion_matrix


height_offset = -1.2 # move the origin 1m up
lidar_height = 0.8

path_curr = os.path.dirname(__file__)
pointcloud_topic_name = "/theia/os_cloud_node/points"
# pointcloud_topic_name = "/ouster/points"
pose_topic_name = "/mavros/local_position/pose"
cfg_path = "cfg/pv_rcnn.yaml"
model_filename = "pv_rcnn_8369.pth"
#cfg_path = "cfg/pointpillar.yaml"
#model_filename = "pointpillar_7728.pth"
print(cfg_path)
print(path_curr)
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
        self.pose_sub = rospy.Subscriber(pose_topic_name, PoseStamped, self.pose_callback)
        
        # publisher
        self.bbox_pub = rospy.Publisher("lidar_detector/detected_bounding_boxes", MarkerArray, queue_size=10)
        self.b3Dbox   = rospy.Publisher("lidar_detector/3D_Lidar_bounding_box", BoundingBox3DArray, queue_size=10)

        # timer
        rospy.Timer(rospy.Duration(0.1), self.detect_callback)
        rospy.Timer(rospy.Duration(0.033), self.vis_callback)

        # coord transformation member vars
        self.LidarPoseMatrix = np.identity(4)
        self.position = np.zeros((3,1))
        self.orientation = np.zeros((3,3))
        # self.body_to_lidar = np.array([[0.0,  0.0,  1.0,  0.09],
        #                                 [-1.0,  0.0,  0.0,  0.0] ,   
        #                                 [0.0, -1.0,  0.0,  0.095],
        #                                 [0.0,  0.0,  0.0,  1.0]])

    def pointcloud_callback(self, pointcloud):
        start_time = time.time()
        pc = ros_numpy.numpify(pointcloud)
        points_xyzi=np.zeros((pc.shape[0] * pc.shape[1],4))
        points_xyzi[:,0]= np.resize(pc['x'], pc.shape[0] * pc.shape[1])
        points_xyzi[:,1]= np.resize(pc['y'], pc.shape[0] * pc.shape[1])
        points_xyzi[:,2]= np.resize(pc['z']+height_offset, pc.shape[0] * pc.shape[1])
        points_xyzi[:,3]= 0

        # for 360 degree detection, we flip x direction
        points_xyzi_inv = np.copy(points_xyzi)
        points_xyzi_inv[:, 0] = -points_xyzi_inv[:, 0]

        self.curr_input_data = self.prepare_data(points_xyzi)
        self.curr_input_data_inv = self.prepare_data(points_xyzi_inv)
        self.pointcloud_received = True
        end_time = time.time()
        # print("data prepare time: ", end_time-start_time)

    def pose_callback(self, pose_msg):
        # print("COME INTO POSE CB")
        quat = np.array([pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, pose_msg.pose.orientation.z, pose_msg.pose.orientation.w])
        rot = quaternion_matrix(quat) # motation matrix for map to body( right below lidar )
        rot[0,3] = pose_msg.pose.position.x
        rot[1,3] = pose_msg.pose.position.y
        rot[2,3] = pose_msg.pose.position.z + lidar_height
        rot[3,3] = 1.

        self.LidarPoseMatrix = rot
        # print("transformation matrix ")
        # print(self.LidarPoseMatrix)
        self.orientation = rot[:3,:3]
        self.position = rot[:3,3].reshape((3,1))

    def detect_callback(self, event):
        if (self.pointcloud_received):
            # print("start inference.")
            start_time = time.time()
            self.detection_results = self.inference(self.curr_input_data)
            self.detection_results_inv = self.inference(self.curr_input_data_inv)

            self.pointcloud_detected = True 
            end_time = time.time()
            print("Detection time: ", end_time-start_time)
            # print("finish inference")

    def vis_callback(self, event):
        if (self.pointcloud_detected):
            boxes_msg, bounding_box3D_msg = self.get_bbox_msg(self.detection_results, self.detection_results_inv)
            self.bbox_pub.publish(boxes_msg)
            self.b3Dbox.publish(bounding_box3D_msg)

    def get_bbox_msg(self, results, results_inv):
        bounding_box3D_msg= BoundingBox3DArray()    #Gary did this change
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
                
                # get eight points (unrotated)
                p1 = np.array([cx+lx/2., cy+ly/2., cz+lz/2.])
                p2 = np.array([cx+lx/2., cy-ly/2., cz+lz/2.])
                p3 = np.array([cx-lx/2., cy+ly/2., cz+lz/2.])
                p4 = np.array([cx-lx/2., cy-ly/2., cz+lz/2.])
                p5 = np.array([cx+lx/2., cy+ly/2., cz-lz/2.])
                p6 = np.array([cx+lx/2., cy-ly/2., cz-lz/2.])
                p7 = np.array([cx-lx/2., cy+ly/2., cz-lz/2.])
                p8 = np.array([cx-lx/2., cy-ly/2., cz-lz/2.])

                # rotation matrix according to lidar-detected-pose
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


                # transform into map frame
                p1r = self.orientation @p1r.reshape((3,1)) + self.position
                p2r = self.orientation @p2r.reshape((3,1)) + self.position
                p3r = self.orientation @p3r.reshape((3,1)) + self.position
                p4r = self.orientation @p4r.reshape((3,1)) + self.position
                p5r = self.orientation @p5r.reshape((3,1)) + self.position
                p6r = self.orientation @p6r.reshape((3,1)) + self.position
                p7r = self.orientation @p7r.reshape((3,1)) + self.position
                p8r = self.orientation @p8r.reshape((3,1)) + self.position



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
                # boxes_msg.header.frame_id = "/map"
                i+=1


                points = [p1r, p2r, p3r, p4r, p5r, p6r, p7r, p8r]

                xmin,xmax = p1r[0], p1r[0]
                ymin,ymax = p1r[1], p1r[1]
                zmin,zmax = p1r[2], p1r[2]
                for p in points:
                    if p[0] < xmin:
                        xmin = p1[0]
                    if p[0] > xmax:
                        xmax = p1[0]
                    if p[1] < ymin:
                        ymin = p1[1]
                    if p[1] > ymax:
                        ymax = p1[1]
                    if p[2] < zmin:
                        zmin = p1[2]
                    if p[2] > zmax:
                        zmax = p1[2]

                center = np.array([[cx],[cy],[cz]])
                lx = xmax - xmin
                ly = ymax - ymin
                lz = zmax - zmin
                center = self.orientation@center + self.position
                # angle = 0
                # Gary Written Portion 145 - 153
                # bounding_box3D_msg.boxes.center.position.x=cx
                # bounding_box3D_msg.boxes.center.position.y=cy
                # bounding_box3D_msg.boxes.center.position.z=cz
                # bounding_box3D_msg.boxes.size.x=lx
                # bounding_box3D_msg.boxes.size.y=ly
                # bounding_box3D_msg.boxes.size.z=lz
                Temp_boxes=BoundingBox3D()
                Temp_boxes=self.make_boxes(cx,cy,cz,lx,ly,lz)
                bounding_box3D_msg.boxes.append(Temp_boxes)

        pred_bboxes_inv = results_inv[0]["pred_boxes"].cpu()
        i = 0
        for box in pred_bboxes_inv:
            if (results_inv[0]["pred_labels"][i] == 2):
                cx = -box[0]
                cy = box[1]
                cz = box[2] - height_offset
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
                p8 = np.array([cx-lx/2., cy-ly/2., cz-lz/2.])

                # rotation matrix according to lidar-detected-pose
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


                # transform into map frame
                p1r = self.orientation @p1r.reshape((3,1)) + self.position
                p2r = self.orientation @p2r.reshape((3,1)) + self.position
                p3r = self.orientation @p3r.reshape((3,1)) + self.position
                p4r = self.orientation @p4r.reshape((3,1)) + self.position
                p5r = self.orientation @p5r.reshape((3,1)) + self.position
                p6r = self.orientation @p6r.reshape((3,1)) + self.position
                p7r = self.orientation @p7r.reshape((3,1)) + self.position
                p8r = self.orientation @p8r.reshape((3,1)) + self.position



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
                # boxes_msg.header.frame_id = "/map"
                i+=1

                points = [p1r, p2r, p3r, p4r, p5r, p6r, p7r, p8r]

                xmin,xmax = p1r[0], p1r[0]
                ymin,ymax = p1r[1], p1r[1]
                zmin,zmax = p1r[2], p1r[2]
                for p in points:
                    if p[0] < xmin:
                        xmin = p1[0]
                    if p[0] > xmax:
                        xmax = p1[0]
                    if p[1] < ymin:
                        ymin = p1[1]
                    if p[1] > ymax:
                        ymax = p1[1]
                    if p[2] < zmin:
                        zmin = p1[2]
                    if p[2] > zmax:
                        zmax = p1[2]

                center = np.array([[cx],[cy],[cz]])
                lx = xmax - xmin
                ly = ymax - ymin
                lz = zmax - zmin
                center = self.orientation@center + self.position

                Temp_boxes=BoundingBox3D()
                Temp_boxes=self.make_boxes(cx,cy,cz,lx,ly,lz)
                bounding_box3D_msg.boxes.append(Temp_boxes)


        return boxes_msg, bounding_box3D_msg

    #Gary Made these Changes as well as this function from 219-232
    def make_boxes(self, x_center, y_center, z_center, len_x, len_y, len_z ):
        box3D_msg=BoundingBox3D()
        
        box3D_msg.center.position.x=x_center
        box3D_msg.center.position.y=y_center
        box3D_msg.center.position.z=z_center
        box3D_msg.size.x=len_x
        box3D_msg.size.y=len_y
        box3D_msg.size.z=len_z
        # box3D_msg.header.frame_id = "theia/os_sensor"
        # box3D_msg.h

        return box3D_msg



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

        line_msg.header.frame_id = "map"
        line_msg.id = marker_id
        line_msg.type = line_msg.LINE_LIST
        line_msg.action = line_msg.MODIFY
        line_msg.scale.x = 0.05
        line_msg.scale.y = 0.05
        line_msg.scale.z = 0.05
        line_msg.color.a = 1.0
        line_msg.color.r = 0.0
        line_msg.color.g = 1.0
        line_msg.color.b = 1.0
        line_msg.pose.orientation.w = 1.0
        line_msg.lifetime = rospy.Time(0.1)
        return line_msg


    # prepare model inputs
    def prepare_data(self, points):
        # print(points.shape)
        self.detector_dataset.load_data(points)
        curr_input_data = self.detector_dataset[0]
        curr_input_data = self.detector_dataset.collate_batch([curr_input_data])

        return curr_input_data
    
    def inference(self, inputs):
        with torch.no_grad():
            load_data_to_gpu(inputs)
            detection_results, _ = self.model.forward(inputs)
        return detection_results
