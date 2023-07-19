
import numpy as np

import torch
import torch.nn as nn


class Code_Balance_Loss(nn.Module):
    def __init__(self, temperature=1, eps=1e-6):
        super(Code_Balance_Loss, self).__init__()
        self.temperature = temperature
        self.eps = eps


    def forward (self, eeg, ir):
        """
        Calculate Bit Balance Loss

        :param: h_eeg: batch of eeg hashes #1 (original)
        :param: h_ir: batch of ir hashes #1 (original)


        :returns: Bit Balance Loss
        """

        loss_bb_eeg = torch.sum(torch.pow(torch.sum(eeg, dim=2), 2))
        loss_bb_ir = torch.sum(torch.pow(torch.sum(ir, dim=2), 2))

        loss_bb = (loss_bb_eeg + loss_bb_ir )
        return loss_bb