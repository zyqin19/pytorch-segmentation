# coding:utf-8
import os
import time
import torch
import random
import numpy as np
import argparse
import cv2
import shutil
from os.path import join as pjoin

from torch.utils import data
from tensorboardX import SummaryWriter
from torch import optim
from torch.backends import cudnn

from ptsegmentation.metrics import averageMeter, runningScore
from ptsegmentation.utils import make_dir, generate_yaml_doc_ruamel, append_yaml_doc_ruamel, print_color, label_resize
from ptsegmentation.loader import get_loader_name
from ptsegmentation.models import get_model
from ptsegmentation.loss import get_loss_function


def get_arguments():
    parser = argparse.ArgumentParser(description="Pytorch Segmentation Master")
    parser.add_argument("--config-name", type=str, default='test',help="")
    parser.add_argument("--cuda-devices", type=int, default=0, help="")
    parser.add_argument("--model-arch", type=str, default='deeplabv333', help="")
    parser.add_argument("--dataset", type=str, default='voc',help="")
    parser.add_argument("--train-split", type=str, default='train', help="")
    parser.add_argument("--test-split", type=str, default='test', help="")
    parser.add_argument("--n-classes", type=int, default=2, help="")
    parser.add_argument("--is-augmentations", type=bool, default=True, help="")
    parser.add_argument("--img-rows", type=int, default=128, help="")
    parser.add_argument("--img-cols", type=int, default=128, help="")
    parser.add_argument("--input-channels", type=int, default=1, help="")
    parser.add_argument("--n-workers", type=int, default=8, help="")
    parser.add_argument("--data-path", type=str, default='../../Data/VOC_Lymph_aug_style/', help="")
    parser.add_argument("--data-name", type=str, default='VOC_Lymph_style', help="")
    parser.add_argument("--fold-series", type=str, default='1', help="")
    parser.add_argument("--seed", type=int, default=1334, help="")
    parser.add_argument("--train-iters", type=int, default=400, help="")
    parser.add_argument("--val-interval", type=int, default=25, help="")
    parser.add_argument("--print-interval", type=int, default=1, help="")
    #
    parser.add_argument("--batch-size", type=int, default=16, help="")
    parser.add_argument("--optimizer-name", type=str, default='amad', help="")
    parser.add_argument("--lr", type=float, default=5.0e-7, help="")
    parser.add_argument("--weight-decay", type=float, default=0.08, help="")
    parser.add_argument("--momentum", type=float, default=0.99, help="")
    parser.add_argument("--loss-name", type=str, default='cross_entropy', help="")
    parser.add_argument("--pkl-path", type=str, default='./pkls', help="")
    # parser.add_argument("--resume", type=str, default='./pkls/VOC_CNV_style/VOC_CNV_style_1018/deeplab3plus_VOC_CNV_style_best_model.pkl', help="")
    parser.add_argument("--resume", type=str, default='', help="")

    return parser.parse_args()


def train(args, writer, logdir):

    # Setup seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    cudnn.deterministic = True

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_devices)
    devices_ids = 0

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Dataloader
    train_loader = get_loader_name(args.dataset)(
        args.data_path,
        is_transform=True,
        split=args.train_split,
        img_size=(args.img_rows, args.img_cols),
    )

    val_loader = get_loader_name(args.dataset)(
        args.data_path,
        is_transform=True,
        split=args.test_split,
        img_size=(args.img_rows, args.img_cols),
    )

    trainloader = data.DataLoader(
        train_loader,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
        shuffle=True,
    )

    valloader = data.DataLoader(
        val_loader,
        batch_size=1,
        num_workers=0
    )
    # Setup Metrics
    running_metrics_val = runningScore(args.n_classes)

    # Setup Model
    # model = get_model(args.model_arch, pretrained=True, num_classes=args.n_classes,
    #                   input_channels=args.input_channels).to(device)

    model = get_model(args.model_arch, args.n_classes).to(device)

    # summary(model, (cfg["data"]['input_channels'],\
    #                              cfg["data"]["img_rows"], cfg["data"]["img_cols"]))
    # model = models.resnet101(pretrained=True)
    # fc_features = model.fc.in_features
    # model.fc = torch.nn.Linear(fc_features, n_classes)

    model = model.cuda(device=devices_ids)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.08)

    # apex
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O2")

    loss_fn = get_loss_function(args.loss_name)

    start_iter = 0
    # reload from checkpoint
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print(
                "Loading model and optimizer from checkpoint '{}'".format(args.resume)
            )
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            start_iter = checkpoint["epoch"]
            print(
                "Loaded checkpoint '{}' (iter {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("No checkpoint found at '{}'".format(args.resume))

    val_loss_meter = averageMeter()
    time_meter = averageMeter()

    best_iou = -100
    best_result = {}
    epoch_iter = start_iter
    running_loss = 0.0
    shape_flag = 1
    for epoch in range(epoch_iter, args.train_iters):
        # train_inter
        loss = 0.0
        for (train_images, train_seg_labels, train_images_name) in trainloader:
            start_ts = time.time()

            model.train()
            train_images = train_images.to(device)
            train_seg_labels = train_seg_labels.to(device)

            optimizer.zero_grad()
            outputs = model(train_images)

            train_labels_, shape_flag = label_resize(outputs, train_seg_labels, shape_flag)

            loss = loss_fn(outputs, train_labels_)

            #################################################################
            # test output
            pred_out = outputs.data.max(1)[1].cpu().numpy()
            pred = np.copy(pred_out[0, :, :])
            if pred.shape != [args.img_rows, args.img_cols]:
                np.resize(pred, (args.img_rows, args.img_cols))

            output_dir = './outputs/test'
            img_name = pjoin(output_dir, ''.join(train_images_name[0]) + ".png")
            cv2.imwrite(img_name, pred*255)
            #################################################################
            loss.backward()
            # apex
            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            optimizer.step()

            running_loss += loss.item()

            time_meter.update(time.time() - start_ts)

        # print_interval
        if (epoch + 1) % args.print_interval == 0:
            fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}".format(
                epoch + 1,
                args.train_iters,
                running_loss / args.print_interval,
                time_meter.avg / args.batch_size,
            )
            running_loss = 0.0
            print(fmt_str)
            writer.add_scalar("loss/train_loss", loss.item(), epoch + 1)
            # histograms and multi-quantile line graphs
            for name, param in model.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch + 1)
            time_meter.reset()

        # val_interval
        if (epoch + 1) % args.val_interval == 0 or (epoch + 1) == args.train_iters:
            model.eval()
            with torch.no_grad():
                for (val_images, val_seg_labels, _) in valloader:
                    val_images = val_images.to(device)
                    val_seg_labels = val_seg_labels.to(device)

                    outputs = model(val_images)
                    val_labels_, _ = label_resize(outputs, val_seg_labels, shape_flag)
                    val_loss = loss_fn(input=outputs, target=val_labels_)

                    pred_out = outputs.data.max(1)[1].cpu().numpy()
                    pred = np.copy(pred_out[:, :, :])
                    if pred.shape != [args.img_rows, args.img_cols]:
                        np.resize(pred, (args.batch_size, args.img_rows, args.img_cols))
                    gt = val_seg_labels.data.cpu().numpy()

                    running_metrics_val.update(gt, pred)
                    val_loss_meter.update(val_loss.item())

            writer.add_scalar("loss/val_loss", val_loss_meter.avg, epoch + 1)
            print("Val Iter %d Loss: %.4f" % (epoch + 1, val_loss_meter.avg))

            score = running_metrics_val.get_scores()
            for k, v in score.items():
                print_color("{}: {}".format(k, v), '31')
                k = "val_metrics/{}".format(k)
                writer.add_scalar("val_metrics/{}".format(k), v, epoch + 1)

            val_loss_meter.reset()
            running_metrics_val.reset()

            if score["Mean IoU"] >= best_iou:
                best_acc = score["Mean IoU"]
                state = {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    # "scheduler_state": scheduler.state_dict(),
                    "best_iou": best_acc,
                }

                best_result = {
                    'best result': score,
                    'best result interval': (epoch + 1),
                }

                save_path = os.path.join(
                    args.pkl_path,
                    os.path.basename(args.config_name),
                    str(run_id)
                )
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                shutil.copy(logdir + '/config.yaml', save_path)
                torch.save(state, save_path + "/{}_{}_best_model.pkl".format(
                    args.model_arch, args.data_name))

    append_yaml_doc_ruamel(best_result, logdir + '/config.yaml')
    writer.close()


if __name__ == "__main__":
    args = get_arguments()

    make_dir("./runs")
    make_dir(args.pkl_path)

    run_dir = os.path.join("runs", os.path.basename(args.config_name))
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    run_list = os.listdir(run_dir)
    run_id_list = [run_list[-4:] for run_list in run_list]
    if run_id_list:
        run_id = args.data_name + '_' + str(int(max(run_id_list)) + 1)
        os.makedirs(os.path.join(run_dir, run_id))

    else:
        run_id = args.data_name + '_' + str(1001)
        os.makedirs(os.path.join(run_dir, run_id))

    print("RUNDIR: {}".format(os.path.join(run_dir, run_id)))
    logdir = os.path.join(run_dir, run_id)
    writer = SummaryWriter(log_dir=logdir)

    opt = vars(args)  # namespace object 2 dictionary object
    generate_yaml_doc_ruamel(opt, logdir+'/config.yaml')

    train(args, writer, logdir)


