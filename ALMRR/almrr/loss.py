# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from math import exp
#
#
# class FocalLoss(nn.Module):
#     """
#     copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
#     This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
#     'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
#         Focal_Loss= -1*alpha*(1-pt)*log(pt)
#     :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
#     :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
#                     focus on hard misclassified example
#     :param smooth: (float,double) smooth value when cross entropy
#     :param balance_index: (int) balance class index, should be specific when alpha is float
#     :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
#     """
#
#     def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.apply_nonlin = apply_nonlin
#         self.alpha = alpha
#         self.gamma = gamma
#         self.balance_index = balance_index
#         self.smooth = smooth
#         self.size_average = size_average
#
#         if self.smooth is not None:
#             if self.smooth < 0 or self.smooth > 1.0:
#                 raise ValueError('smooth value should be in [0,1]')
#
#     def forward(self, logit, target):
#         if self.apply_nonlin is not None:
#             logit = self.apply_nonlin(logit)
#         num_class = 2
#
#         if logit.dim() > 2:
#             # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
#             logit = logit.view(logit.size(0), logit.size(1), -1)
#             logit = logit.permute(0, 2, 1).contiguous()
#             logit = logit.view(-1, logit.size(-1))
#         target = torch.squeeze(target, 1)
#         target = target.view(-1, 1)
#         alpha = self.alpha
#
#
#         if alpha is None:
#             alpha = torch.ones(num_class, 1)
#         elif isinstance(alpha, (list, np.ndarray)):
#             assert len(alpha) == num_class
#             alpha = torch.FloatTensor(alpha).view(num_class, 1)
#             alpha = alpha / alpha.sum()
#         elif isinstance(alpha, float):
#             alpha = torch.ones(num_class, 1)
#             alpha = alpha * (1 - self.alpha)
#             alpha[self.balance_index] = self.alpha
#
#         else:
#             raise TypeError('Not support alpha type')
#
#         if alpha.device != logit.device:
#             alpha = alpha.to(logit.device)
#
#         idx = target.cpu().long()
#
#
#
#         one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
#         one_hot_key = one_hot_key.scatter_(1, idx, 1)
#         if one_hot_key.device != logit.device:
#             one_hot_key = one_hot_key.to(logit.device)
#
#
#
#         min_val = torch.tensor(0.00001, device=logit.device)
#         max_val = torch.tensor(1.0 - 0.00001, device=logit.device)
#         logit = torch.clamp(logit, min_val, max_val)  # Clamping to avoid log(0)
#         logit = torch.cat((1 - logit, logit), dim=1)
#
#
#
#         if self.smooth:
#             one_hot_key = torch.clamp(
#                 one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
#         pt = (one_hot_key * logit).sum(1) + self.smooth
#         logpt = pt.log()
#
#         gamma = self.gamma
#
#         alpha = alpha[idx]
#         alpha = torch.squeeze(alpha)
#         loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt
#
#         if self.size_average:
#             loss = loss.mean()
#         return loss
#
#
# class DiceLoss(nn.Module):
#     def __init__(self, smooth=1, p=2, reduction='mean'):
#         super(DiceLoss, self).__init__()
#         self.smooth = smooth
#         self.p = p
#         self.reduction = reduction
#
#     def forward(self, input, target):
#         # 假设我们只关心input的第二个通道
#
#         assert input.size() == target.size(), "输入与目标大小必须相同"
#         input_flat = input.reshape(-1)
#         target_flat = target.reshape(-1)
#
#         intersection = (input_flat * target_flat).sum()
#         dice_coeff = (2. * intersection + self.smooth) / (
#                 input_flat.pow(self.p).sum() + target_flat.pow(self.p).sum() + self.smooth)
#
#         dice_loss = 1 - dice_coeff
#         if self.reduction == 'mean':
#             return dice_loss.mean()
#         elif self.reduction == 'sum':
#             return dice_loss.sum()
#         else:
#             return dice_loss


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp


class BinaryFocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param reduction: `none`|`mean`|`sum`
    :param **kwargs
        balance_index: (int) balance class index, should be specific when alpha is float
    """

    def __init__(self, alpha=3, gamma=2, ignore_index=None, reduction='mean', **kwargs):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = 1e-5  # set '1e-4' when train with FP16
        self.ignore_index = ignore_index
        self.reduction = reduction

        assert self.reduction in ['none', 'mean', 'sum']

        # if self.alpha is None:
        #     self.alpha = torch.ones(2)
        # elif isinstance(self.alpha, (list, np.ndarray)):
        #     self.alpha = np.asarray(self.alpha)
        #     self.alpha = np.reshape(self.alpha, (2))
        #     assert self.alpha.shape[0] == 2, \
        #         'the `alpha` shape is not match the number of class'
        # elif isinstance(self.alpha, (float, int)):
        #     self.alpha = np.asarray([self.alpha, 1.0 - self.alpha], dtype=np.float).view(2)

        # else:
        #     raise TypeError('{} not supported'.format(type(self.alpha)))

    def forward(self, output, target):
        prob = torch.sigmoid(output)
        prob = torch.clamp(prob, self.smooth, 1.0 - self.smooth)

        valid_mask = None
        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float()

        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()
        if valid_mask is not None:
            pos_mask = pos_mask * valid_mask
            neg_mask = neg_mask * valid_mask

        pos_weight = (pos_mask * torch.pow(1 - prob, self.gamma)).detach()
        pos_loss = -pos_weight * torch.log(prob)  # / (torch.sum(pos_weight) + 1e-4)

        neg_weight = (neg_mask * torch.pow(prob, self.gamma)).detach()
        neg_loss = -self.alpha * neg_weight * F.logsigmoid(-output)  # / (torch.sum(neg_weight) + 1e-4)
        loss = pos_loss + neg_loss
        loss = loss.mean()
        return loss


class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, input, target):
        # 假设我们只关心input的第二个通道
        input = input[:, 1:2, :, :]  # 现在input的形状应该与target相匹配

        assert input.size() == target.size(), "输入与目标大小必须相同"
        input_flat = input.reshape(-1)
        target_flat = target.reshape(-1)

        intersection = (input_flat * target_flat).sum()
        dice_coeff = (2. * intersection + self.smooth) / (
                input_flat.pow(self.p).sum() + target_flat.pow(self.p).sum() + self.smooth)

        dice_loss = 1 - dice_coeff
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss
