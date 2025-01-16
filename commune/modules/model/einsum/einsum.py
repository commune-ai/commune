import torch
import torch.nn as nn

class EinsumNN(nn.Module):
    def __init__(self, n: int, m: int):
        super().__init__()
        self.vector1 = torch.randn(n)
        self.vector2 = torch.randn(m)   
        self.vector1 = nn.Parameter(torch.randn(n))
        self.vector2 = nn.Parameter(torch.randn(m))
    

    def forward(self, x, y ):
        
        return torch.einsum('i,j->ij', x, y)
    
    def verify(self):
        # Verify that the forward pass is correct
        x1 = self.vector1.unsqueeze(1)
        x2 = self.vector2.unsqueeze(0)
