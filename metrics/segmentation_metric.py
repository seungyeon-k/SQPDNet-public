# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np


class SegementationAccuracy:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_true, label_pred, num_classes):
        mask = (label_true >= 0) & (label_true < num_classes)
        hist = np.bincount(
            num_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=num_classes ** 2,
        ).reshape(num_classes, num_classes)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(
                lt.flatten(), lp.flatten(), self.num_classes
            )

    def get_scores(self):
        """Returns accuracy score evaluation result.
        - overall accuracy
        - mean accuracy
        """
        hist = self.confusion_matrix
        overall_acc = np.diag(hist).sum() / hist.sum()

        cls_cnt = hist.sum(axis=1)
        cls_hit = np.diag(hist)
        mean_acc = 0
        N = 0
        for i in range(len(cls_cnt)):
            if cls_cnt[i] != 0:
                mean_acc += cls_hit[i] / cls_cnt[i]
                N += 1
        mean_acc /= N

        return {
            "Overall Acc": overall_acc,
            "Mean Acc": mean_acc,
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
