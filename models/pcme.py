""" PCME model base code

PCME
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import torch
import torch.nn as nn
import os
from models.eeg_encoder import EncoderEEG
from models.ir_encoder import EncoderIR


class PCME(nn.Module):
    """Probabilistic CrossModal Embedding (PCME) module"""
    def __init__(self):
        super(PCME, self).__init__()

        self.eeg_enc = EncoderEEG()
        self.ir_enc = EncoderIR()


    def forward(self, eegs, irs):
        image_output, K_eegfeature = self.eeg_enc(eegs)
        caption_output, K_irfeature = self.ir_enc(irs)

        # image对应eeg，caption(text)对应ir video
        return {
            'image_features': image_output['embedding'],
            'image_attentions': image_output.get('attention'),
            'image_residuals': image_output.get('residual'),
            'image_logsigma': image_output.get('logsigma'),
            'image_logsigma_att': image_output.get('uncertainty_attention'),
            'caption_features': caption_output['embedding'],
            'caption_attentions': caption_output.get('attention'),
            'caption_residuals': caption_output.get('residual'),
            'caption_logsigma': caption_output.get('logsigma'),
            'caption_logsigma_att': caption_output.get('uncertainty_attention'),
        }, K_eegfeature, K_irfeature


    def generate_img_code(self, i):
        return self.eeg_enc(i)[1].detach()  #要的是特征

    def generate_txt_code(self, t):
        return self.ir_enc(t)[1].detach()

    def save(self, name=None, path='/home/spi/FED-RE/Cross_Modal_Retrirval/checkpoints/', cuda_device=None):
        if not os.path.exists(path):
            os.makedirs(path)
        if cuda_device.type == 'cpu':
            torch.save(self.state_dict(), os.path.join(path, name))
        else:
            with torch.cuda.device(cuda_device):
                torch.save(self.state_dict(), os.path.join(path, name))
        return name