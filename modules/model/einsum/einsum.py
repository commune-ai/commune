import torch
import torch.nn as nn

class EinsumNN(nn.Module):
    def __init__(self, n: int, m: int):
        super().__init__()
        self.vector1 = nn.Parameter(torch.randn(n))
        self.vector2 = nn.Parameter(torch.randn(m))
    
    def forward(self):
        return torch.einsum('i,j->ij', self.vector1, self.vector2)