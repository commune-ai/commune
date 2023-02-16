import torch
import torch.nn as nn


class ParamMap(nn.Module):
    def __init__(self, device='cuda'):
        super(ParamMap, self).__init__()
        self.tensor = nn.ParameterDict()
        self.resolve_device(device)

    def resolve_device(self,device:str):
        if'cuda' in device:
            assert torch.cuda.is_available()

        self.device = device
        
        assert 'cuda' in self.device or 'cpu' in self.device, f'Invalid device {self.device}'
            
        return self.device
        
    def list_keys(self):
        return list(self.tensor.keys())
    def put(self, key:str, tensor:torch.Tensor):
        self.tensor[key] = torch.nn.Parameter(tensor).to(self.device)

    def get(self, key:str):
        return self.tensor[key]
    
    @classmethod
    def test(cls):
        self = cls()
        self.put('hey', torch.randn(10))
        self.put('brodf ', torch.randn(10))
        print(self.state_dict())
        
    
    
    
if __name__ == "__main__":
    import commune
    commune.module(ParamMap).put('hey', torch.randn(10))
    