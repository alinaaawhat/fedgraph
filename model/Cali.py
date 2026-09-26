import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base import MLPLayer

class DomainCali(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.d_e = self.cfg.EditGrad.compressed_dim
        self.hidden_dim = self.cfg.Cali.hidden_dim
        self.aligned_feat_dim = self.cfg.Cali.aligned_feat_dim
        dims = [self.d_e] + [self.hidden_dim] * (self.cfg.Cali.num_layers - 1) + [4 * self.aligned_feat_dim]
        self.blocks = nn.ModuleList([MLPLayer(dims[i], dims[i+1], 
                                              dropout=self.cfg.Cali.dropout,
                                              act= 'ReLU',
                                              layernorm=self.cfg.Cali.layernorm) for i in range(len(dims) - 2)])
        self.blocks.append(MLPLayer(dims[-2], dims[-1], act=None, dropout=0, layernorm=False))
    
    def forward(self, e):
        h = e
        for blk in self.blocks:
            h = blk(h)  

        gamma_f, beta_f, gamma_l, beta_l = h.chunk(4, dim=-1)
        if self.cfg.Cali.softplus:
            gamma_f = F.softplus(gamma_f)
            gamma_l = F.softplus(gamma_l)
        return gamma_f, beta_f, gamma_l, beta_l
    