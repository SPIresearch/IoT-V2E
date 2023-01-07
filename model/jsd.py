import torch.nn.functional as F
import torch.nn as nn

class JSD(nn.Module):
    
    def __init__(self):
        super(JSD, self).__init__()
    
    def forward(self, logits1, logits2):
        probs1 =  F.softmax(logits1, dim=1)
        probs2=  F.softmax(logits2, dim=1)

        m = 0.5 * (probs1 + probs2)
        loss = 0.0
        loss += F.kl_div(F.log_softmax(logits1, dim=1), m, reduction="none") .mean()
        loss += F.kl_div(F.log_softmax(logits2, dim=1), m, reduction="none") .mean()
        return (0.5 * loss)
