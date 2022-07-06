import torch
import torch.nn as nn
import torch.nn.functional as F

# FUNCTIONS 
  
def max_onehot(x):
    x_max = torch.max(x, dim=1, keepdim=True)[0]
    x[x != x_max] = 0
    return x

    
def get_selective_mean(x, labels, weights = None, p = 1):
    N,C,H,W = x.size()
    labels = labels.squeeze(-1).squeeze(-1)*p
    
    x_sum = x.view(N, C, -1).mean(dim=-1)
    
    if weights is not None:
        x_sum = x_sum*weights + labels
    else:
        x_sum = x_sum + labels    
        
    x_sum = x_sum[x_sum>0]
    x_sum = x_sum - labels[labels>0]
    x_mean = torch.mean(x_sum)

    return x_mean

def max_norm(p, e=1e-6):
    if p.dim() == 3:
        C, H, W = p.size()
        p = F.relu(p)
        max_v = torch.max(p.view(C,-1),dim=-1)[0].view(C,1,1)
        p = p/(max_v + e)
        
    elif p.dim() == 4:
        N, C, H, W = p.size()
        p = F.relu(p)
        max_v = torch.max(p.view(N,C,-1),dim=-1)[0].view(N,C,1,1)
        p = p/(max_v + e)
        
    return p    
  

class LogSumExpPool(nn.Module):

    def __init__(self, gamma):
        super(LogSumExpPool, self).__init__()
        self.gamma = gamma

    def forward(self, feat_map):
        """
        Numerically stable implementation of the operation
        Arguments:
            feat_map(Tensor): tensor with shape (N, C, H, W)
            return(Tensor): tensor with shape (N, C, 1, 1)
        """
        (N, C, H, W) = feat_map.shape

        # (N, C, 1, 1) m
        m, _ = torch.max(
            feat_map, dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)

        # (N, C, H, W) value0
        value0 = feat_map - m
        area = 1.0 / (H * W)
        g = self.gamma

        return m + 1 / g * torch.log(area * torch.sum(torch.exp(g * value0), dim=(-1, -2), keepdim=True))  
    

