from typing import List, Dict
import torch.nn as nn

class Generator(nn.Sequential):
    r"""Domain discriminator model from
    `Domain-Adversarial Training of Neural Networks (ICML 2015) <https://arxiv.org/abs/1505.07818>`_
    Distinguish whether the input features come from the source domain or the target domain.
    The source domain label is 1 and the target domain label is 0.
    Args:
        in_feature (int): dimension of the input feature
        hidden_size (int): dimension of the hidden features
        batch_norm (bool): whether use :class:`~torch.nn.BatchNorm1d`.
            Use :class:`~torch.nn.Dropout` if ``batch_norm`` is False. Default: True.
    Shape:
        - Inputs: (minibatch, `in_feature`)
        - Outputs: :math:`(minibatch, 1)`
    """

    def __init__(self, in_feature=2, hidden_size=64, out_channel=2):
        super(Generator, self).__init__(
            nn.GroupNorm(1, in_feature, eps=1e-05, affine=True),
            nn.Conv3d(in_feature, hidden_size, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False), 
            nn.ReLU(),
            nn.GroupNorm(8, hidden_size, eps=1e-05, affine=True),
            nn.Conv3d(hidden_size, hidden_size, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False), 
            nn.ReLU(),
            nn.Conv3d(hidden_size, out_channel, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False), 
        )
 

    def get_parameters(self) -> List[Dict]:
        return [{"params": self.parameters(), "lr": 1.}]
