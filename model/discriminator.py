from typing import List, Dict
import torch.nn as nn


class DomainDiscriminator(nn.Sequential):
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

    def __init__(self, in_feature: int, hidden_size: int, batch_norm=True):
        if batch_norm:
            super(DomainDiscriminator, self).__init__(
                nn.Linear(in_feature, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
            )
        else:
            super(DomainDiscriminator, self).__init__(
                nn.Linear(in_feature, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
            )

    def get_parameters(self) -> List[Dict]:
        return [{"params": self.parameters(), "lr": 1.}]

    
class CNNDomainDiscriminator(nn.Sequential):
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

    def __init__(self, in_feature=64, hidden_size=64):
        super(CNNDomainDiscriminator, self).__init__(
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.GroupNorm(8, in_feature, eps=1e-05, affine=True),
            nn.Conv3d(in_feature, hidden_size, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False), 
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.GroupNorm(8, hidden_size, eps=1e-05, affine=True),
            nn.Conv3d(hidden_size, hidden_size, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False), 
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1,1,1)),
            nn.Flatten(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
 

    def get_parameters(self) -> List[Dict]:
        return [{"params": self.parameters(), "lr": 1.}]

# MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#       (dropout): Dropout3d(p=0.5, inplace=False)
#       (basic_module): DoubleConv(
#         (SingleConv1): SingleConv(
#           (groupnorm): GroupNorm(8, 256, eps=1e-05, affine=True)
#           (conv): Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
#           (ReLU): ReLU(inplace=True)