import torch
import torch.nn as nn
import torch.nn.functional as F



class NNs_reg(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(NNs_reg,self).__init__()
        lin1 = nn.Linear(input_dim, 1024)
        lin2 = nn.Linear(1024, 1024)
        lin3 = nn.Linear(1024, 64)
        lin4 = nn.Linear(64, output_dim)
        for lin in [lin1,lin2,lin3,lin4]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3, nn.ReLU(True), lin4)
        
    def forward(self,x):
        out = self._main(x)
        return out
