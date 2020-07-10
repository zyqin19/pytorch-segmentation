# coding:utf-8
import os
import torch
import random
import numpy as np
from ruamel import yaml
import cv2
from os.path import join as pjoin

from torch.utils import data
from torch import optim
from torch.backends import cudnn

from ptsegmentation.metrics import runningScore
from ptsegmentation.utils import generate_yaml_doc_ruamel
from ptsegmentation.loader import get_loader_name
from ptsegmentation.models import get_model

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
devices_ids = 0

def test(logdir, pkls_dir, output_dir):

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

    # Setup Dataloader
    test_loader = get_loader_name(cfg["dataset"])(
        cfg["data_path"],
        is_transform=True,
        split='test',
        test_mode=True,
        img_size=(cfg["img_rows"], cfg["img_cols"]),
    )

    testloader = data.DataLoader(
        test_loader,
        batch_size = 1,
        num_workers = 0
    )
    # Setup Metrics
    running_metrics_val = runningScore(cfg["n_classes"])

    model = get_model(cfg["model_arch"], cfg["n_classes"]).to(device)
    model = model.cuda(device=devices_ids)

    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=0.08)

    # reload from checkpoint
    resume = "/{}_{}_best_model.pkl".format(
        cfg["model_arch"], cfg["data_name"])
    pkls = pkls_dir + resume

    if os.path.isfile(pkls):
        checkpoint = torch.load(pkls)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_iter = checkpoint["epoch"]

    model.eval()
    with torch.no_grad():
        for (test_images, test_seg_labels, test_images_name) in testloader:
            test_images = test_images.to(device)
            test_seg_labels = test_seg_labels.to(device)

            outputs = model(test_images)

            pred_out = outputs.data.max(1)[1].cpu().numpy()
            pred = np.copy(pred_out[:, :, :])

            if pred.shape != [cfg["img_rows"], cfg["img_cols"]]:
                np.resize(pred, (cfg["batch_size"], cfg["img_rows"], cfg["img_cols"]))

            img_name = pjoin(output_dir, ''.join(test_images_name) + ".png")
            cv2.imwrite(img_name, np.squeeze(pred)*255)
            gt = test_seg_labels.data.cpu().numpy()

            # torchvision.utils.save_image(outputs.data.cpu(), img_name)

            running_metrics_val.update(gt, pred)

    score = running_metrics_val.get_scores()
    for k, v in score.items():
        print(k, v)

    k = {
        'model': cfg,
        'test result':score
    }
    generate_yaml_doc_ruamel(k, output_dir + '/config.yaml')

if __name__ == "__main__":

    run_dir = 'VOC_CNV_style'
    run_id = '1025'
    logdir = os.path.join('./runs', run_dir, run_dir+'_'+run_id)

    pkls_dir = os.path.join('./pkls', run_dir, run_dir+'_'+run_id)

    output_dir = os.path.join('./outputs', run_dir, run_dir+'_'+run_id)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    test(logdir, pkls_dir, output_dir)