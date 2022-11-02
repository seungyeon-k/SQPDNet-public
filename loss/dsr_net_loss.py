import torch.nn as nn
import torch


class DSRNetLoss(nn.Module):
    def __init__(self, **kwargs):
        super(DSRNetLoss, self).__init__()
        self.loss_func_motion = nn.MSELoss()

        self.alpha_motion = kwargs.get('alpha_motion', 1.0)
        self.alpha_mask = kwargs.get('alpha_mask', 5.0)

    def forward_motion(self, input, target):
        loss = self.alpha_motion * self.loss_func_motion(input, target)
        return loss

    def forward_mask(self, logit_pred, mask_gt, batch_order):
        loss = 0
        B, K, S1, S2, S3 = logit_pred.size()
        for b in range(B):
            permute_pred = torch.stack(
                [logit_pred[b:b + 1, -1]] + [logit_pred[b:b + 1, i] for i in batch_order[b]],
                dim=1).contiguous()
            loss += nn.CrossEntropyLoss()(permute_pred, mask_gt[b:b + 1])
        loss *= self.alpha_mask
        return loss
