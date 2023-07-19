import numpy as np

import torch
import torch.nn as nn


class Bit_uncorrelation_Loss(nn.Module):
    def __init__(self, temperature=1, eps=1e-6):
        super(Bit_uncorrelation_Loss, self).__init__()
        self.temperature = temperature
        self.eps = eps


    def forward (self, eeg, ir, device):
        """
        Calculate Bit Balance Loss

        :param: h_eeg: batch of eeg hashes #1 (original)
        :param: h_ir: batch of ir hashes #1 (original)


        :returns: Bit Balance Loss
        """
        batch_hashcode=((eeg.detach() + ir.detach()) / 2).sign()  #(256,64)
        eeg_hash=eeg.detach().sign()
        ir_hash=ir.detach().sign()


        eeg_aaT=torch.bmm(eeg_hash[:, :, :, None],eeg_hash[:, :, :, None])
        ir_aaT=torch.bmm()

        aaT=torch.bmm(batch_hashcode[:,:,:,None],batch_hashcode[:,:,None,:])

        result=torch.sum(aaT,dim=0)/256-torch.eye(16).to(device)

        loss_bu=torch.sum(result ** 2)

        return loss_bu