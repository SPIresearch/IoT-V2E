import torch
def Entropy_div(input_):
    epsilon = 1e-5
    input_ = torch.mean(input_, 0) + epsilon
    entropy = input_ * torch.log(input_)
    entropy = torch.sum(entropy)
    return entropy 

def Entropy_condition(input_):
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1).mean()
    return entropy 

def Entropy(input_):
    return Entropy_condition(input_) #+ Entropy_div(input_)