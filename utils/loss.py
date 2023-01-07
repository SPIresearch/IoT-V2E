import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def dice_loss(input, target):
    """soft dice loss"""
    eps = 1e-7
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return 1 - 2. * intersection / ((iflat ** 2).sum() + (tflat ** 2).sum() + eps)

def bce_dice(input, target):
    dice = dice_loss(input, target)
    bce = F.binary_cross_entropy(input, target)
    return  bce + dice

def softmax_dice_loss(input, target):
    """soft dice loss"""
    eps = 1e-7
    input = F.softmax(input, 1)
    iflat = input[:,1,...,].reshape(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return 1 - 2. * intersection / ((iflat ** 2).sum() + (tflat ** 2).sum() + eps)

def ce_dice(input, target):
    dice = softmax_dice_loss(input, target)
    ce = F.cross_entropy(input, target.squeeze().long())
    return  ce + dice

def calc_class_weight(labels_count):
    total = np.sum(labels_count)
    class_weight = dict()
    num_classes = len(labels_count)

    factor = 1 / num_classes
    mu = [factor * 1.5, factor * 2, factor * 1.5, factor, factor * 1.5] # THESE CONFIGS ARE FOR SLEEP-EDF-20 ONLY

    for key in range(num_classes):
        score = math.log(mu[key] * total / float(labels_count[key]))
        class_weight[key] = score if score > 1.0 else 1.0
        class_weight[key] = round(class_weight[key] * mu[key], 2)

    class_weight = [class_weight[i] for i in range(num_classes)]

    return class_weight


def weighted_CrossEntropyLoss(output, target, classes_weights, device):
    cr = nn.CrossEntropyLoss(weight=torch.tensor(classes_weights).to(device))
    return cr(output, target)