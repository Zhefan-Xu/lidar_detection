#!/usr/bin/env python

import rospy
from lidar_detector import *

def main():
	rospy.init_node("lidar_detector_node")
	lidar_detector()
	rospy.spin()


if __name__ == "__main__":
	main()