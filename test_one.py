# coding:utf-8
import os
import torch
import random
import numpy as np
from ruamel import yaml
import cv2
from os.path import join as pjoin
from PIL import Image

from torch.utils import data
from torch import optim
from torch.backends import cudnn
from torchvision import transforms

from ptsegmentation.metrics import averageMeter, runningScore
from ptsegmentation.utils import make_dir, get_logger,generate_yaml_doc_ruamel,append_yaml_doc_ruamel
from ptsegmentation.loader import get_loader
from ptsegmentation.models import get_model

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
devices_ids = 0

def test_one(logdir, pkls_dir, output_dir, img_dir, seg_dir, img_name):

    with open(logdir+'/config.yaml') as fp:
        cfg = yaml.load(fp, Loader=yaml.Loader)

    # Setup seeds
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    cudnn.deterministic = True

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Metrics
    running_metrics_val = runningScore(cfg["n_classes"])

    model = get_model(cfg["model_arch"], cfg["n_classes"]).to(device)
    model = model.cuda(device=devices_ids)

    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=0.08)

    loss_fn = torch.nn.CrossEntropyLoss()

    # reload from checkpoint
    resume = "/{}_{}_best_model.pkl".format(
        cfg["model_arch"], cfg["data_name"])
    pkls = pkls_dir + resume

    if os.path.isfile(pkls):
        checkpoint = torch.load(pkls)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_iter = checkpoint["epoch"]

    test_loss_meter = averageMeter()

    normMean = [0.498, 0.497, 0.497]
    normStd = [0.206, 0.206, 0.206]
    tf = transforms.Compose(
        [
            # p.torch_transform(),
            transforms.ToTensor(),
            transforms.Normalize(normMean, normStd),
        ]
    )

    img = cv2.imread(img_dir + img_name)
    lbl = cv2.imread(seg_dir + img_name)

    test_images = tf(img).unsqueeze(0)

    lbl = lbl[:, :, 1].squeeze()
    test_seg_labels = torch.from_numpy(lbl).long().unsqueeze(0)
    test_seg_labels[test_seg_labels > 0] = 1

    test_images = test_images.to(device)
    test_seg_labels = test_seg_labels.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(test_images)
        test_loss = loss_fn(input=outputs, target=test_seg_labels)

        pred_out = outputs.data.max(1)[1].cpu().numpy()
        pred = np.copy(pred_out[:, :, :])
        if pred.shape != [cfg["img_rows"], cfg["img_cols"]]:
            np.resize(pred, (cfg["batch_size"], cfg["img_rows"], cfg["img_cols"]))

        img_name = pjoin(output_dir, ''.join(img_name) + ".png")
        # pred_rgb = Image.fromarray((np.squeeze(pred)*255).astype('uint8')).convert('RGB')
        # pred_rgb.save('E:\Data\VOC_CNV_style/005.png')
        # cv2.imwrite(img_name, np.squeeze(pred)*255)
        gt = test_seg_labels.data.cpu().numpy()

        running_metrics_val.update(gt, pred)
        test_loss_meter.update(test_loss.item())

    score, result = running_metrics_val.get_scores()
    for k, v in score.items():
        print(k, v)

    k = {
        'model': cfg,
        'test result':result
    }
    # generate_yaml_doc_ruamel(k, output_dir + '/config.yaml')

if __name__ == "__main__":
    run_dir = 'VOC_CNV_style'
    run_id = '1010'
    logdir = os.path.join('./runs', run_dir, run_dir+'_'+run_id)

    pkls_dir = os.path.join('./pkls', run_dir, run_dir+'_'+run_id)

    output_dir = os.path.join('./outputs', run_dir, run_dir+'_'+run_id)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_dir = '../../Data/VOC_CNV_style/Images/'
    seg_dir = '../../Data/VOC_CNV_style/SegmentationClass/'
    img_name = '00005.png'

    test_one(logdir, pkls_dir, output_dir, img_dir, seg_dir, img_name)