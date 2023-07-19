#不确定度量化
import numpy as np

import torch
import torch.nn as nn


class Cal_quanization_LOSS(nn.Module):
    def __init__(self, temperature=1, eps=1e-6):
        super(Cal_quanization_LOSS, self).__init__()
        self.temperature = temperature
        self.eps = eps


    def forward (self, eeg, ir, ind, B, un_eeg, un_ir, device):
        """
        Calculate Quantization Loss
        :returns: Quantization Loss
        """
        # loss_quant_eeg = torch.sum(torch.pow(B[ind, :] - eeg, 2))
        # loss_quant_ir = torch.sum(torch.pow(B[ind, :] - ir, 2))

        e_eeg=np.exp(un_eeg[ind, :].cpu().detach()).to(device)
        e_ir=np.exp(un_ir[ind, :].cpu().detach()).to(device)

        kan=torch.pow(B[ind, :] - eeg, 2)
        loss_quant_eeg = torch.sum(torch.mul(e_eeg,torch.pow(B[ind, :] - eeg, 2)))
        loss_quant_ir = torch.sum(torch.mul(e_ir,torch.pow(B[ind, :] - ir, 2)))

        loss_quant = (loss_quant_eeg + loss_quant_ir)

        return loss_quant

