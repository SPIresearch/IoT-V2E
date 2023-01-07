from typing import Optional, List, Dict, Tuple, Callable
import torch.nn as nn
import torch.nn.functional as F
import torch
from .grl import WarmStartGradientReverseLayer

class MarginDisparityDiscrepancy(nn.Module):
    r"""The margin disparity discrepancy (MDD) proposed in `Bridging Theory and Algorithm for Domain Adaptation (ICML 2019) <https://arxiv.org/abs/1904.05801>`_.
    MDD can measure the distribution discrepancy in domain adaptation.
    The :math:`y^s` and :math:`y^t` are logits output by the main head on the source and target domain respectively.
    The :math:`y_{adv}^s` and :math:`y_{adv}^t` are logits output by the adversarial head.
    The definition can be described as:
    .. math::
        \mathcal{D}_{\gamma}(\hat{\mathcal{S}}, \hat{\mathcal{T}}) =
        -\gamma \mathbb{E}_{y^s, y_{adv}^s \sim\hat{\mathcal{S}}} L_s (y^s, y_{adv}^s) +
        \mathbb{E}_{y^t, y_{adv}^t \sim\hat{\mathcal{T}}} L_t (y^t, y_{adv}^t),
    where :math:`\gamma` is a margin hyper-parameter, :math:`L_s` refers to the disparity function defined on the source domain
    and :math:`L_t` refers to the disparity function defined on the target domain.
    Args:
        source_disparity (callable): The disparity function defined on the source domain, :math:`L_s`.
        target_disparity (callable): The disparity function defined on the target domain, :math:`L_t`.
        margin (float): margin :math:`\gamma`. Default: 4
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
    Inputs:
        - y_s: output :math:`y^s` by the main head on the source domain
        - y_s_adv: output :math:`y^s` by the adversarial head on the source domain
        - y_t: output :math:`y^t` by the main head on the target domain
        - y_t_adv: output :math:`y_{adv}^t` by the adversarial head on the target domain
        - w_s (optional): instance weights for source domain
        - w_t (optional): instance weights for target domain
    Examples::
        >>> num_outputs = 2
        >>> batch_size = 10
        >>> loss = MarginDisparityDiscrepancy(margin=4., source_disparity=F.l1_loss, target_disparity=F.l1_loss)
        >>> # output from source domain and target domain
        >>> y_s, y_t = torch.randn(batch_size, num_outputs), torch.randn(batch_size, num_outputs)
        >>> # adversarial output from source domain and target domain
        >>> y_s_adv, y_t_adv = torch.randn(batch_size, num_outputs), torch.randn(batch_size, num_outputs)
        >>> output = loss(y_s, y_s_adv, y_t, y_t_adv)
    """

    def __init__(self, source_disparity: Callable, target_disparity: Callable,
                 margin: Optional[float] = 4, reduction: Optional[str] = 'mean'):
        super(MarginDisparityDiscrepancy, self).__init__()
        self.margin = margin
        self.reduction = reduction
        self.source_disparity = source_disparity
        self.target_disparity = target_disparity

    def forward(self, y_s: torch.Tensor, y_s_adv: torch.Tensor, y_t: torch.Tensor, y_t_adv: torch.Tensor,
                w_s: Optional[torch.Tensor] = None, w_t: Optional[torch.Tensor] = None) -> torch.Tensor:

        source_loss = -self.margin * self.source_disparity(y_s, y_s_adv)
        target_loss = self.target_disparity(y_t, y_t_adv)
        if w_s is None:
            w_s = torch.ones_like(source_loss)
        source_loss = source_loss * w_s
        if w_t is None:
            w_t = torch.ones_like(target_loss)
        target_loss = target_loss * w_t

        
        if self.reduction == 'mean':
            loss = source_loss.mean() + target_loss.mean()
        elif self.reduction == 'sum':
            loss = source_loss.sum() + target_loss.sum()
        return loss


class ClassificationMarginDisparityDiscrepancy(MarginDisparityDiscrepancy):
    r"""
    The margin disparity discrepancy (MDD) proposed in `Bridging Theory and Algorithm for Domain Adaptation (ICML 2019) <https://arxiv.org/abs/1904.05801>`_.
    It measures the distribution discrepancy in domain adaptation
    for classification.
    When margin is equal to 1, it's also called disparity discrepancy (DD).
    The :math:`y^s` and :math:`y^t` are logits output by the main classifier on the source and target domain respectively.
    The :math:`y_{adv}^s` and :math:`y_{adv}^t` are logits output by the adversarial classifier.
    They are expected to contain raw, unnormalized scores for each class.
    The definition can be described as:
    .. math::
        \mathcal{D}_{\gamma}(\hat{\mathcal{S}}, \hat{\mathcal{T}}) =
        \gamma \mathbb{E}_{y^s, y_{adv}^s \sim\hat{\mathcal{S}}} \log\left(\frac{\exp(y_{adv}^s[h_{y^s}])}{\sum_j \exp(y_{adv}^s[j])}\right) +
        \mathbb{E}_{y^t, y_{adv}^t \sim\hat{\mathcal{T}}} \log\left(1-\frac{\exp(y_{adv}^t[h_{y^t}])}{\sum_j \exp(y_{adv}^t[j])}\right),
    where :math:`\gamma` is a margin hyper-parameter and :math:`h_y` refers to the predicted label when the logits output is :math:`y`.
    You can see more details in `Bridging Theory and Algorithm for Domain Adaptation <https://arxiv.org/abs/1904.05801>`_.
    Args:
        margin (float): margin :math:`\gamma`. Default: 4
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
    Inputs:
        - y_s: logits output :math:`y^s` by the main classifier on the source domain
        - y_s_adv: logits output :math:`y^s` by the adversarial classifier on the source domain
        - y_t: logits output :math:`y^t` by the main classifier on the target domain
        - y_t_adv: logits output :math:`y_{adv}^t` by the adversarial classifier on the target domain
    Shape:
        - Inputs: :math:`(minibatch, C)` where C = number of classes, or :math:`(minibatch, C, d_1, d_2, ..., d_K)`
          with :math:`K \geq 1` in the case of `K`-dimensional loss.
        - Output: scalar. If :attr:`reduction` is ``'none'``, then the same size as the target: :math:`(minibatch)`, or
          :math:`(minibatch, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case of K-dimensional loss.
    Examples::
        >>> num_classes = 2
        >>> batch_size = 10
        >>> loss = ClassificationMarginDisparityDiscrepancy(margin=4.)
        >>> # logits output from source domain and target domain
        >>> y_s, y_t = torch.randn(batch_size, num_classes), torch.randn(batch_size, num_classes)
        >>> # adversarial logits output from source domain and target domain
        >>> y_s_adv, y_t_adv = torch.randn(batch_size, num_classes), torch.randn(batch_size, num_classes)
        >>> output = loss(y_s, y_s_adv, y_t, y_t_adv)
    """

    def __init__(self, margin: Optional[float] = 4, **kwargs):
        def source_discrepancy(y: torch.Tensor, y_adv: torch.Tensor):
            _, prediction = y.max(dim=1)
            return F.nll_loss(shift_log(y_adv), prediction, reduction='none')

        def target_discrepancy(y: torch.Tensor, y_adv: torch.Tensor):
            _, prediction = y.max(dim=1)
            return -F.nll_loss(shift_log(1. - y_adv), prediction, reduction='none')

        super(ClassificationMarginDisparityDiscrepancy, self).__init__(source_discrepancy, target_discrepancy, margin,
                                                                       **kwargs)
def shift_log(x: torch.Tensor, offset: Optional[float] = 1e-6) -> torch.Tensor:
    r"""
    First shift, then calculate log, which can be described as:
    .. math::
        y = \max(\log(x+\text{offset}), 0)
    Used to avoid the gradient explosion problem in log(x) function when x=0.
    Args:
        x (torch.Tensor): input tensor
        offset (float, optional): offset size. Default: 1e-6
    .. note::
        Input tensor falls in [0., 1.] and the output tensor falls in [-log(offset), 0]
    """
    return torch.log(torch.clamp(x + offset, max=1.))


class CMDD(nn.Module):
    def __init__(self, classifier: nn.Module, grl: Optional = None):
        super(CMDD, self).__init__()
        self.classifier =classifier
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True) if grl is None else grl
        self.last_layer = nn.Sigmoid()
        self.loss = ClassificationMarginDisparityDiscrepancy()
    def forward(self, p_s, f_s, p_t, f_t):
        y_s =  torch.cat([1-p_s, p_s], dim=1)# ps n, 1, h, w
        y_t =  torch.cat([1-p_t, p_t], dim=1)
        y_s_adv = self.last_layer(self.grl(f_s))
        y_s_adv =  torch.cat([1-y_s_adv, y_s_adv], dim=1)
        y_t_adv = self.last_layer(self.grl(f_t))
        y_t_adv =  torch.cat([1-y_t_adv, y_t_adv], dim=1)
        return self.loss(y_s, y_s_adv, y_t, y_t_adv)
