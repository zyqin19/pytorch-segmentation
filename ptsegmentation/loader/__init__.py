import glob
import os
import imghdr
import random
import math
import collections

from torch.utils import data

from ptsegmentation.loader.voc_loader_foldcv import VOCLoader_FoldCV
from ptsegmentation.loader.voc_loader import VOCLoader
from ptsegmentation.loader.imagefolder_loader import ImageFolderLoader

def get_loader_name(name):
    """get_loader_name

    :param name:
    """
    return {
        "voc": VOCLoader,
        "voc_foldcv": VOCLoader_FoldCV,
        "imagefolder": ImageFolderLoader,
    }[name]


def get_foldcv(root_dir, cv):
    list_path = []
    cv_file = collections.defaultdict(list)

    for root,dirs,files in os.walk(root_dir):
        for file in files:
            if file.endswith('jpg') or file.endswith('png') or \
                    file.endswith('bmp') or file.endswith('jpeg'):
                list_path.append(file)

    floor_cv_single = math.floor(len(list_path)/cv)
    random.shuffle(list_path)
    for cv_i in range(cv):
        if cv_i == (cv-1):
            cv_file[str(cv_i)] = list_path[cv_i*floor_cv_single:]
        cv_file[str(cv_i)] = list_path[cv_i*floor_cv_single:(cv_i+1)*floor_cv_single]

    return cv_file


def get_loader(name, data_path, list, split, is_augmentations,
               img_size, batch_size, num_workers, shuffle):
    dataset = get_loader_name(name)(
                                    data_path,
                                    file_list = list,
                                    is_augmentations=is_augmentations,
                                    split=split,
                                    img_size=img_size,
                                    )
    data_loader = data.DataLoader(
                                    dataset,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    shuffle=shuffle,
                                    )

    return data_loader


