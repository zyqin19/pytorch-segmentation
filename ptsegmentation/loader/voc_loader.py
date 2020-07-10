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

class VOCLoader(data.Dataset):

    def __init__(
        self,
        root,
        is_transform=True,
        img_size=512,
        split="train",
        test_mode=False,
        is_augmentations=False,
        img_norm=True,
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

        if not self.test_mode:
            for split in ["train", "test"]:
                path = pjoin(self.root, "ImageSets/Segmentation", split + ".txt")
                file_list = tuple(open(path, "r"))
                file_list = [id_.rstrip() for id_ in file_list]
                self.files[split] = file_list
        else:
            split = "test"
            path = pjoin(self.root, "ImageSets/Segmentation", split + ".txt")
            file_list = tuple(open(path, "r"))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files[split] = file_list
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

        self.tf2 = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        im_name = self.files[self.split][index]
        im_path = pjoin(self.root, "Images", im_name + ".png")
        lbl_path = pjoin(self.root, "SegmentationClass", im_name + ".png")
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
        # lbl = torch.from_numpy(np.array(lbl).astype(float))
        lbl = self.tf2(lbl)
        lbl[lbl > 0] = 1.0
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