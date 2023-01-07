import torch.nn as nn
import numpy as np
from utils.metric.binary import hd95
import torch
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score


def dice_coef(input, target, threshold=0.5):
    smooth = 1.
    iflat = (input.view(-1) > threshold).float()
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return (2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)

def dice_coef_np(input, target, eps=1e-7):
    input = np.ravel(input)
    target = np.ravel(target)
    intersection = (input * target).sum()

    return (2. * intersection) / (input.sum() + target.sum() + eps)

def hausdorff(batch_pred, batch_y, threshold=0.5):
    """batch size must equal 1"""
    batch_pred = batch_pred.cpu().squeeze().numpy() > threshold
    batch_y = batch_y.cpu().squeeze().numpy()
    return hd95(batch_pred, batch_y)

def get_metrics():
    metrics = {}
    # metrics["dice"] = dice_coef
    # metrics["hd95"] = hausdorff
    metrics["acc"] = accuracy
    metrics["mf1"] = f1
    return metrics



# 原来自己的metric
def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def f1(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
    return f1_score(pred.cpu().numpy(), target.data.cpu().numpy(), average='macro')


def accuracy_test(output, target):
    acc = accuracy_score(target, output)
    return acc


def f1_test(output, target):
    mf1 = f1_score(target, output, average='macro')
    return mf1

