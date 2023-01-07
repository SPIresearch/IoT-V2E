import torch
import torch.nn as nn
import torch.nn.functional as F

def get_weight(input_):
    with torch.no_grad():
        entropy = -input_ * torch.log(input_ + 1e-5)
        entropy = torch.sum(entropy, dim=1)
        weight = 1.0 + torch.exp(-entropy)
        weight = weight / torch.sum(weight)
        return weight