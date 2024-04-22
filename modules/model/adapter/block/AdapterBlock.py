import torch
import commune

from typing import *


from commune.model.layer import LayerBlock

class AdapterBlock(torch.nn.Module, commune.Module):

    def __init__(self, 
                 in_dim = 10,
                 hidden_dim:int=64,
                 out_dim: int = None,
                 num_layers:int=8,
                 device: str = 'cuda'):
        
        self.config = dict(locals())
        self.__dict__.update(self.config)
        
        torch.nn.Module.__init__(self)
        
        self.set_model(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, device=device)

        
    
    def set_model(self,
                  in_dim:int,
              hidden_dim:int, 
              out_dim:int, 
              device='cpu',
              num_layers:int=1):
        
        if out_dim == None:
            out_dim = in_dim
        
        # build the encoder
        encoder_blocks = [LayerBlock(in_dim, hidden_dim)]
        for i in range(num_layers):
            encoder_blocks.append(LayerBlock(hidden_dim, hidden_dim, norm_fn='layer', act_fn='gelu'))
        self.encoder = torch.nn.Sequential(*encoder_blocks)
        
        # build the decoder
        
        decoder_blocks = []
        for i in range(num_layers):
            decoder_blocks.append(LayerBlock(hidden_dim, hidden_dim, norm_fn='layer', act_fn='gelu'))
        
        decoder_blocks += [LayerBlock(hidden_dim, out_dim, norm_fn=None, act_fn=None)]
        self.decoder = torch.nn.Sequential(*decoder_blocks)
        self.set_device(device)
        
        
    def forward(self, x:torch.Tensor, *args, **kwargs) -> torch.Tensor:
        
        emb = self.encoder(x.to(self.device))
        emb = self.decoder(emb)
        emb = torch.nn.Softmax(dim=-1)(emb)
        emb = torch.log(emb + 1e-40)
        print(emb.shape)
        return emb

    @property
    def device(self):
        if self._device == None:
            self._device = commune.resolve_device('cuda')
        return self._device
    
    
    def set_device(self, device:str) -> str:
        device = commune.resolve_device(device)
        self.to(device)
        self._device = device
        return device

    def encode(self, x:torch.Tensor):
        return self.encoder(x)

    def decode(self, x:torch.Tensor):
        return self.decoder(x)

    def loss(self, x):
        return torch.nn.functional.mse_loss(self.forward(x), x.to(self.device))

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return f'AdapterModel()'

    def __str__(self):
        return f'AdapterModel()'
    
