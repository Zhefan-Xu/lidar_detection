# LiDAR-based 3D Obstacle Detection for Autonomous Robots
This repo aims at solving 3D obstacle detection problem using LiDAR pointcloud with the learning-based methods.

### Install
```
git clone https://github.com/Zhefan-Xu/lidar_detection.git
```

### Run Demo
```
rosrun lidar_detection lidar_detector_node.py
```

### Sample Dataset
Please use [this link](https://drive.google.com/drive/folders/1EFMqNyYWhlLeew4jQkHutBEgnGz9naOK?usp=sharing) to download the sample rosbag dataset to reproduce the results below.

### Results
The following video shows a sample detection results for pedestrians. The algorithm runs at ~10 fps with the NVIDIA RTX 3080 GPU.



https://user-images.githubusercontent.com/55560905/217433251-db137883-5e82-4d3d-a30d-181f84def030.mp4

