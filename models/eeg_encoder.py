""" Image encoder based on PVSE implementation.
Reference code:
    https://github.com/yalesong/pvse/blob/master/model.py
"""
import torch.nn as nn
from torchvision import models

from .pie_model import PIENet
from .uncertainty_module import UncertaintyModuleImage
from utils.tensor_utils import l2_normalize, sample_gaussian_tensors
from model.model_xiao import Xiao_Fusion_mm
from model.model_xiao import AttnSleep

class EncoderEEG(nn.Module):
    def __init__(self):
        super(EncoderEEG, self).__init__()

        embed_dim1 = 1024
        embed_dim2= 64
        self.use_attention = True
        self.use_probemb = True

        # Backbone CNN
        #self.cnn = Xiao_Fusion_mm(n_classes=5, video_input_channels=3, EEG_lastfeatures=512, Video_lastfeatures=512)
        self.eeg_method= AttnSleep()
        cnn_dim=self.cnn_dim = 512



        self.fc1 = nn.Linear(cnn_dim, embed_dim1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(embed_dim1, embed_dim2)
        self.tanh= nn.Tanh()


        if self.use_attention:
            self.pie_net = PIENet(1, 64, embed_dim2, 64)

        if self.use_probemb:
            self.uncertain_net = UncertaintyModuleImage(64, embed_dim2, 64)

        #encoder 是否需要参数？
        for idx, param in enumerate(self.eeg_method.parameters()):
            param.requires_grad = False

        self.n_samples_inference = 7

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, eegs):
        x= self.eeg_method(eegs)
        x = self.fc1(x)
        x=self.relu(x)
        x=self.fc2(x)
        x0 = self.tanh(x)

        output = {}
        out_7x7 = x0.view(-1, 64, 1 * 1)

        if self.use_attention:
            out, attn, residual = self.pie_net(x0, out_7x7.transpose(1, 2))
            output['attention'] = attn
            output['residual'] = residual

        if self.use_probemb:
            uncertain_out = self.uncertain_net(x0, out_7x7.transpose(1, 2))
            logsigma = uncertain_out['logsigma']
            output['logsigma'] = logsigma
            output['uncertainty_attention'] = uncertain_out['attention']

        out = l2_normalize(out)

        if self.use_probemb and self.n_samples_inference:
            output['embedding'] = sample_gaussian_tensors(out, logsigma, self.n_samples_inference)
        else:
            output['embedding'] = out

        return output, x0
