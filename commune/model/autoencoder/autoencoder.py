
import commune
import torch
from typing import Optional




class AutoEncoder(torch.nn.Module, commune.Module):
    def __init__(self, 
                 in_dim = 10,
                 hidden_dim:int=64,
                 num_layers:int=1,
                 device: str = 'cpu',
                 out_dim: Optional[int] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None):
        torch.nn.Module.__init__(self)
        self.build(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, device=device)
        self.set_optimizer(**(optimizer if optimizer != None else {}))
        self.set_device(device)
        
    @property
    def device(self) -> str:
        return self._device
    
    def set_device(self, device:str) -> str:
        self.to(device)
        self._device = device
        return device
    @property
    def device (self) -> str:
        return self._device

    def build(self, in_dim:int,
              hidden_dim:int, 
              out_dim:int, 
              device='cpu',
              num_layers:int=1):
        from commune.model.layer import LayerBlock
        
        # build the encoder
        encoder_blocks = [LayerBlock(in_dim, hidden_dim)]
        for i in range(num_layers):
            encoder_blocks.append(LayerBlock(hidden_dim, hidden_dim))
        self.encoder = torch.nn.Sequential(*encoder_blocks)
        
        # build the decoder
        
        decoder_blocks = []
        for i in range(num_layers):
            decoder_blocks.append(LayerBlock(hidden_dim, hidden_dim))
        
        decoder_blocks += [LayerBlock(hidden_dim, out_dim)]
        self.decoder = torch.nn.Sequential(*decoder_blocks)


    def forward(self, x):
        
        emb = self.encoder(x.to(self.device))
        emb = self.decoder(emb)
        return emb

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def loss(self, x):
        return torch.nn.functional.mse_loss(self.forward(x), x.to(self.device))

    def learn_step(self, x) -> float:
        self.optimizer.zero_grad()
        loss = self.loss(x)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return f'AutoEncoder()'

    def __str__(self):
        return f'AutoEncoder()'
    

    
    def set_optimizer(self, **params) -> 'optimizer':
        self.optimizer = self.get_optimizer(**params)
        return self.optimizer
    
    def get_optimizer(self, optimizer=None, **params) -> 'optimizer':
        if optimizer == None:
            optimizer =  torch.optim.Adam
        elif isinstance(optimizer, str):
            optimizer = commune.import_object(optimizer_class)
        elif isinstance(optimizer, type):
            return optimizer_class
        
        
        params = params.pop('params', {'lr': 0.1})
        optimizer = optimizer(self.parameters(), **params)
        
        return optimizer
    
    @classmethod
    def train(cls,
             in_dim:int=512, 
             out_dim:int=512 ,
             hidden_dim:int=256, 
             num_layers:int = 1,
             batch_dim:int=32,
             device:str='cuda',
             num_batches:int=3000,
             steps_per_batch:int = 1):
        self = cls(in_dim=in_dim, 
                   out_dim=out_dim,
                   device=device, 
                   num_layers=num_layers,
                   optimizer = {'lr': 0.0001},)
                

        for i in range(num_batches):
            
            x = torch.randn([batch_dim, in_dim])
            for j in range(steps_per_batch):
                print(self.learn_step(x))
            
    

if __name__ == "__main__":
    AutoEncoder.train()
