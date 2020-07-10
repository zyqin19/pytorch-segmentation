"""
Misc Utility functions
"""
import os
import logging
import datetime
import numpy as np
import torch.nn as nn
from PIL import Image

from collections import OrderedDict
import torch


def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]


def alpha_blend(input_image, segmentation_mask, alpha=0.5):
    """Alpha Blending utility to overlay RGB masks on RBG images
        :param input_image is a np.ndarray with 3 channels
        :param segmentation_mask is a np.ndarray with 3 channels
        :param alpha is a float value
    """
    blended = np.zeros(input_image.size, dtype=np.float32)
    blended = input_image * alpha + segmentation_mask * (1 - alpha)
    return blended


def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    """
    if not next(iter(state_dict)).startswith("module."):
        return state_dict  # abort if dict is not a DataParallel model_state
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def get_logger(logdir):
    logger = logging.getLogger("ptsegmentation")
    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    file_path = os.path.join(logdir, "run_{}.log".format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.root.setLevel(logging.INFO)
    return logger


def make_dir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)

def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()


def trans_dict_to_xml(data):
    xml = []
    for k in sorted(data.keys()):
        v = data.get(k)
        if k == 'detail' and not v.startswith('<![CDATA['):
            v = '<![CDATA[{}]]>'.format(v)
        xml.append('<{key}>{value}</{key}>'.format(key=k, value=v))
    return '<xml>{}</xml>'.format(''.join(xml))


def generate_yaml_doc_ruamel(data, yaml_file):
    from ruamel import yaml

    file = open(yaml_file, 'w', encoding='utf-8')
    yaml.dump(data, file, Dumper=yaml.RoundTripDumper)
    file.close()


def append_yaml_doc_ruamel(data, yaml_file):
    from ruamel import yaml

    file = open(yaml_file, 'a', encoding='utf-8')
    yaml.dump(data, file, Dumper=yaml.RoundTripDumper)
    file.close()


def get_yaml_data_ruamel(yaml_file):
    from ruamel import yaml
    file = open(yaml_file, 'r', encoding='utf-8')
    data = yaml.load(file.read(), Loader=yaml.Loader)
    file.close()
    return data


def get_args_from_dict(data):
    # data is a dict object
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--estimators', nargs="+", required=True,
                        choices=data)
    args = vars(parser.parse_args())

    return args


def minmaxscaler(data):
    min = np.amin(data)
    max = np.amax(data)
    return (data - min)/(max-min)

def judge_nan(x):
    import re
    value = re.compile(r'^\s*[-+]*[0-9]+\.*[0-9]*\s*$')
    if value.match(str(x)):
        return round(x, 6)
    else:
        return 'nan'

def print_color(str, color):
    # color code in ['30', '31', '32', '33', '34', '35', '36', '37']
    # ['write/black', 'red', 'green', 'yellow', 'blue', 'purple', 'dark green', 'gray']
    # refer url : https://blog.csdn.net/hzk594512323/article/details/85281992
    start_line = '\033[1;' + color + 'm '
    end_line = '\033[0m'
    print(start_line + str + ' ' + end_line)


def label_resize(outputs, label, shape_flag):
    _, _, H, W = outputs.shape
    b, c, h, w = label.shape
    if [h, w] != [H, W] and shape_flag == 1:
        print_color("model_outputs.shape <> train_seg_labels.shape", '32')
        shape_flag += 1
    labels_ = label.resize_([b, c, H, W])

    return labels_, shape_flag





