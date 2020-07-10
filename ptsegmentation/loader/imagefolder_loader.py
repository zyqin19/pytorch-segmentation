import os
from os.path import join as pjoin
import collections
import json
import torch
import numpy as np
# import Augmentor
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt
import glob
import cv2

from PIL import Image
from tqdm import tqdm
from torch.utils import data
from torchvision import transforms

from augmentor.Pipeline import Pipeline

class ImageFolderLoader(data.Dataset):

    def __init__(
        self,
        root,
        is_transform=True,
        img_size=512,
        split="train",
        test_mode=False,
        is_augmentations=False,
        img_norm=True,
        img_folder_name = "nst_image",
        lbl_folder_name = "label"
    ):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.is_augmentations = is_augmentations
        self.img_norm = img_norm
        self.test_mode = test_mode
        self.n_classes = 2
        self.mean = np.array([125.08347, 124.99436, 124.99769])
        self.files = collections.defaultdict(list)
        self.img_size = img_size
        self.img_folder_name = img_folder_name
        self.lbl_folder_name = lbl_folder_name

        if not self.test_mode:
            for split in ["train", "test"]:
                path = pjoin(root, split,  self.img_folder_name)
                self.files[split] = self.file_name(path,'.png')
            # self.setup_annotations()

        self.normMean = [0.498, 0.497, 0.497]
        self.normStd = [0.206, 0.206, 0.206]

        self.tf = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(self.normMean, self.normStd),

            ]
        )

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        im_name = self.files[self.split][index]
        im_path = pjoin(self.root, self.split, self.img_folder_name, im_name)
        lbl_path = pjoin(self.root, self.split, self.lbl_folder_name, im_name)
        im = Image.open(im_path)
        lbl = Image.open(lbl_path)
        if self.is_augmentations:
            im, lbl = self.augmentations(im, lbl)
        if self.is_transform:
            im, lbl = self.transform(im, lbl)
        return im, lbl, im_name

    def transform(self, img, lbl):
        if img.size == self.img_size:
            pass
        else:
            img = img.resize(self.img_size, Image.ANTIALIAS)  # uint8 with RGB mode
            lbl = lbl.resize(self.img_size, Image.ANTIALIAS)
        img = self.tf(img)
        lbl = torch.from_numpy(np.array(lbl)).long()
        lbl[lbl > 0] = 1
        return img, lbl

    def augmentations(self, img, lbl):
        p = Pipeline(img, lbl)
        # Add operations to the pipeline as normal:
        p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)
        p.rotate(probability=0.5, max_left_rotation=5, max_right_rotation=5)
        p.zoom_random(probability=0.5, percentage_area=0.8)
        p.skew_left_right(probability=0.5)
        p.flip_left_right(probability=0.5)
        img2, lbl2 = p.sample()
        return img2, lbl2

    def file_name(self, file_dir,file_type='.png'):#默认为文件夹下的所有文件
        lst = []
        for root, dirs, files in os.walk(file_dir):
            for file in files:
                if(file_type == ''):
                    lst.append(file)
                else:
                    if os.path.splitext(file)[1] == str(file_type):#获取指定类型的文件名
                        lst.append(file)
        return lst