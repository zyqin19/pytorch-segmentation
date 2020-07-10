# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np
from ptsegmentation.utils import judge_nan

class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        )
        # hist = [TN, FP, FN, TP]
        hist = hist.reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IoU
            - fwavacc
            - DICE
            - VOE
            - f1-score
        """
        hist = self.confusion_matrix
        # hist = [TN,FP;FN,TP]
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        # iou = iu.sum() / self.n_classes
        mean_iou = np.nanmean(iu)      # if classes = 2: iou = miou
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iou = dict(zip(range(self.n_classes), iu))

        ##############################################
        tn = hist[0, 0]
        tp = np.diag(hist).sum() - tn
        fp = np.triu(hist, 1).sum()
        fn = np.tril(hist, -1).sum()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

        # for medical img, img_seg \in [0,1]
        dice = 2 * tp / (tp + tp + fn + fp)
        # dice = f1-score
        dsc = 2 * tp / (tp + fn + fp)
        # dsc = jaccard
        # voe = 2 * abs(fp + fn) / (tp + tp + fn + fp)
        # voe = 1 - dsc

        k2 = {
            # "Overall Acc: \t": acc,
            'Mean Acc': float(judge_nan(acc_cls)),
            # "FreqW Acc : \t": fwavacc,
            'Mean IoU': float(judge_nan(mean_iou)),
            'F1-score': float(judge_nan(f1)),
            'DSC': float(judge_nan(dsc)),
            'Precision': float(judge_nan(precision)),
            'Recall': float(judge_nan(recall)),
        }

        return k2

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class averageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
