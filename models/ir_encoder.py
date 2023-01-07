import torch.nn as nn
from torchvision import models

from .pie_model import PIENet
from .uncertainty_module import UncertaintyModuleImage
from utils.tensor_utils import l2_normalize, sample_gaussian_tensors
from model.resnet import ResNet, generate_model

class EncoderIR(nn.Module):
    def __init__(self):
        super(EncoderIR, self).__init__()

        embed_dim1 = 1024
        embed_dim2= 64
        self.use_attention = True
        self.use_probemb = True

        # Backbone CNN
        self.cnn = generate_model(model_depth=18,
                                      n_classes=5,
                                      n_input_channels=3,
                                      shortcut_type='B',
                                      conv1_t_size=7,
                                      conv1_t_stride=1,
                                      no_max_pool=False,
                                      widen_factor=1.0)
        cnn_dim=self.cnn_dim = 512



        self.ir_hash_fc1 = nn.Linear(cnn_dim, embed_dim1)
        self.relu = nn.ReLU(inplace=True)
        self.ir_hash_fc2 = nn.Linear(embed_dim1, embed_dim2)
        self.tanh= nn.Tanh()


        if self.use_attention:
            self.pie_net = PIENet(1, 64, embed_dim2, 64)

        if self.use_probemb:
            self.uncertain_net = UncertaintyModuleImage(64, embed_dim2, 64)

        #encoder 是否需要参数？
        for idx, param in enumerate(self.cnn.parameters()):
            param.requires_grad = False

        self.n_samples_inference = 7

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self,  irs):
        x= self.cnn(irs)  #512
        x = self.ir_hash_fc1(x)
        x=self.relu(x)#1024
        x =self.ir_hash_fc2(x)  #batchsize*K
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

        return output,x0
