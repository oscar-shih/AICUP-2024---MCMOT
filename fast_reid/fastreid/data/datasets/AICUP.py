# encoding: utf-8
"""
@author:  sherlock (changed by Nir)
@contact: sherlockliao01@gmail.com
"""


import os
import glob
import warnings
import os.path as osp

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class AICUP(ImageDataset):
    _junk_pids = [0, -1]
    dataset_dir = ''
    dataset_url = ''
    dataset_name = "AICUP"

    def __init__(self, root='datasets', **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'AICUP-ReID')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as "bounding_box_train" under '
                          '"AICUP-ReID".')

        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')
        self.extra_gallery_dir = osp.join(self.data_dir, 'images')
        self.extra_gallery = False

        required_files = [
            self.data_dir,
            self.train_dir,
            # self.query_dir,
            # self.gallery_dir,
        ]

        self.check_before_run(required_files)

        train = lambda: self.process_dir(self.train_dir)
        query = lambda: self.process_dir(self.query_dir, is_train=False)
        gallery = lambda: self.process_dir(self.gallery_dir, is_train=False) + \
                          (self.process_dir(self.extra_gallery_dir, is_train=False) if self.extra_gallery else [])

        super(AICUP, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '*.bmp'))
        data = []
        
        for img_path in img_paths:
            # TrackID_TimeStemp_FrameID_acc_data.bmp
            path_txt = img_path.split(os.sep)[-1].split('_')
            pid, frame_id = int(path_txt[0]), int(path_txt[2])
            if pid == -1:
                continue  # junk images are just ignored
            
            frame_id -= 1  # index starts from 0
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                frame_id = self.dataset_name + "_" + str(frame_id)
            data.append((img_path, pid, frame_id))

        return data
