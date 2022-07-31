import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss


class SoftIoULoss(nn.Module):
    def __init__(self, n_classes):
        super(SoftIoULoss, self).__init__()
        self.n_classes = n_classes

    @staticmethod
    def to_one_hot(tensor, n_classes):
        n, h, w = tensor.size()
        one_hot = torch.zeros(n, n_classes, h, w).scatter_(1, tensor.view(n, 1, h, w), 1)
        return one_hot

    def forward(self, input, target):
        # logit => N x Classes x H x W
        # target => N x H x W

        N = len(input)

        pred = F.softmax(input, dim=1)
        target_onehot = self.to_one_hot(target, self.n_classes)

        # Numerator Product
        inter = pred * target_onehot
        # Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.n_classes, -1).sum(2)

        # Denominator
        union = pred + target_onehot - (pred * target_onehot)
        # Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.n_classes, -1).sum(2)

        loss = inter / (union + 1e-16)

        # Return average loss over classes and batch
        return -loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.bce_fn = nn.BCEWithLogitsLoss(weight=self.weight)

    def forward(self, preds, labels):
        # mask = labels
        # labels = labels[mask]
        # preds = preds[mask]

        logpt = -self.bce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss


import torch
from torch.autograd import Function


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target)

        t = (2 * self.inter.float() + eps) / (self.union.float() + eps)
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


class DiceCoeffLoss(nn.Module):
    """Dice coeff for individual examples"""

    def __init__(self):
        super(DiceCoeffLoss, self).__init__()

    def forward(self, input, target):
        # self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target)

        t = (2 * self.inter.float() + eps) / (self.union.float() + eps)
        return 1 - t

    # This function has only a single output, so it gets only one gradient
    # def backward(self, grad_output):
    #
    #     input, target = self.saved_variables
    #     grad_input = grad_target = None
    #
    #     if self.needs_input_grad[0]:
    #         grad_input = grad_output * 2 * (target * self.union - self.inter) \
    #                      / (self.union * self.union)
    #     if self.needs_input_grad[1]:
    #         grad_target = None
    #
    #     return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


def focal_loss(input, target):
    alpha = 0.25
    gamma = 2
    weight = None

    bce_fn = nn.BCEWithLogitsLoss(weight=weight)
    logpt = -bce_fn(input, target)
    pt = torch.exp(logpt)
    loss = -((1 - pt) ** gamma) * alpha * logpt
    return loss


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target)

        t = (2 * self.inter.float() + eps) / (self.union.float() + eps)
        return t

    # This function has only a single output, so it gets only one gradient
    # def backward(self, grad_output):
    #
    #     input, target = self.saved_variables
    #     grad_input = grad_target = None
    #
    #     if self.needs_input_grad[0]:
    #         grad_input = grad_output * 2 * (target * self.union - self.inter) \
    #                      / (self.union * self.union)
    #     if self.needs_input_grad[1]:
    #         grad_target = None
    #
    #     return grad_input, grad_target


class WeightedDiceCoeffLoss(nn.Module):
    """Dice coeff for individual examples"""

    def __init__(self):
        super(WeightedDiceCoeffLoss, self).__init__()

    def forward(self, input, target):
        # self.save_for_backward(input, target)
        eps = 0.0001
        a = 0.3
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target)

        score = (2 * self.inter.float() + eps) / (self.union.float() + eps)

        # weighted_loss = torch.tensor(1 - ((math.exp(a * score) - 1) / (math.exp(a) - 1))).requires_grad_()
        weighted_loss = torch.tensor((-math.log(score)) ** a).requires_grad_()
        return weighted_loss

    # This function has only a single output, so it gets only one gradient
    # def backward(self, grad_output):
    #
    #     input, target = self.saved_variables
    #     grad_input = grad_target = None
    #
    #     if self.needs_input_grad[0]:
    #         grad_input = grad_output * 2 * (target * self.union - self.inter) \
    #                      / (self.union * self.union)
    #     if self.needs_input_grad[1]:
    #         grad_target = None
    #
    #     return grad_input, grad_target


class IoULoss(nn.Module):
    """Dice coeff for individual examples"""

    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, input, target):
        # self.save_for_backward(input, target)
        eps = 0.0001

        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input.view(-1)) + torch.sum(target.view(-1)) - torch.dot(input.view(-1), target.view(-1))

        t = (self.inter + eps) / (self.union + eps)
        return 1 - t

    # This function has only a single output, so it gets only one gradient
    # def backward(self, grad_output):
    #
    #     input, target = self.saved_variables
    #     grad_input = grad_target = None
    #
    #     if self.needs_input_grad[0]:
    #         grad_input = grad_output * 2 * (target * self.union - self.inter) \
    #                      / (self.union * self.union)
    #     if self.needs_input_grad[1]:
    #         grad_target = None
    #
    #     return grad_input, grad_target

class mod_IoULoss_FP(nn.Module):
    """Dice coeff for individual examples"""

    def __init__(self):
        super(mod_IoULoss_FP, self).__init__()

    def forward(self, input, target):
        # self.save_for_backward(input, target)
        eps = 0.0001


        self.TP = (torch.dot(input.view(-1), target.view(-1)))
        self.FP = torch.sum(input.view(-1)) - self.TP
        self.FN = torch.sum(target.view(-1)) - self.TP

        # ori_IoU: TP / (TP + FP + FN)
        # mod_IoU: (TP - FP) / (TP + FP + FN)

        self.inter = (self.TP - self.FP)
        self.union = (self.TP + self.FP + self.FN)

        t = (self.inter + eps) / (self.union + eps)
        return 1 - t

class mod_IoULoss_FN(nn.Module):
    """Dice coeff for individual examples"""

    def __init__(self):
        super(mod_IoULoss_FN, self).__init__()

    def forward(self, input, target):
        # self.save_for_backward(input, target)
        eps = 0.0001


        self.TP = (torch.dot(input.view(-1), target.view(-1)))
        self.FP = torch.sum(input.view(-1)) - self.TP
        self.FN = torch.sum(target.view(-1)) - self.TP

        # ori_IoU: TP / (TP + FP + FN)
        # mod_IoU: (TP - FP) / (TP + FP + FN)

        self.inter = (self.TP - self.FN)
        self.union = (self.TP + self.FP + self.FN)

        t = (self.inter + eps) / (self.union + eps)
        return 1 - t

class IoU(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input.view(-1)) + torch.sum(target.view(-1)) - torch.dot(input.view(-1), target.view(-1))

        t = (self.inter + eps) / (self.union + eps)
        return t


def iou(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + IoU().forward(c[0], c[1])

    return s / (i + 1)