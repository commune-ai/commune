import torch
import torch.nn as nn


class TensorMap(nn.Module):
    def __init__(self):
        super(TensorMap, self).__init__()
        self.tensors = {}
    
    def forward(self, key):
        return self.tensors[key]

    def add_tensor(self, key, tensor):
        self.tensors[key] = tensor

    def get_tensor(self, key):
        return self.tensors[key]
    
    
    