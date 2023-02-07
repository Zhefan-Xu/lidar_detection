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

path_curr = os.path.dirname(__file__)
data_filename = "mill19_data104.npy"
# cfg_path = "cfg/pv_rcnn.yaml"
# model_filename = "pv_rcnn_8369.pth"
cfg_path = "cfg/pointpillar.yaml"
model_filename = "pointpillar_7728.pth"
cfg_from_yaml_file(os.path.join(path_curr, cfg_path), cfg)

class TestDataset(DatasetTemplate):
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

    def load_data(self, filename):
        self.points = np.load(os.path.join(path_curr, filename))
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


def main():
    print("test lidar detector.")
    logger = common_utils.create_logger()
    demo_dataset = TestDataset(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False, root_path=Path(path_curr))

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=os.path.join(path_curr, model_filename), logger=logger, to_cpu=False)
    model.cuda()
    model.eval()
    with torch.no_grad():
        start = time()
        demo_dataset.load_data(data_filename)
        data_dict = demo_dataset[0]
        data_dict = demo_dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)
        red_dicts, _ = model.forward(data_dict)
        end = time()
        print("Time: ", end-start)
        print(red_dicts)

if __name__ == '__main__':
    main()