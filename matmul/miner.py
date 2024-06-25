import commune as c
from typing import *
import torch

class MatMul(c.Module):

    def forward(self, x = None, y = None, n=10):
        if x is None or y is None:
            x =  torch.rand(n, n) 
            y = torch.rand(n, n)
        return torch.matmul(x, y)
    
    def test(self, n=3):
        x = torch.rand(n, n)
        y = torch.rand(n, n)
        return self.forward(x, y)