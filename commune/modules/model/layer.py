import torch
from typing import Callable


class LayerBlock(torch.nn.Module):

    def __init__(self, in_dim:int=10, out_dim:int=10, norm_fn:Callable = 'layer', act_fn:str = 'gelu'):
        super(LayerBlock, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # self.W = torch.nn.Parameter(torch.randn(self.in_dim, self.out_dim))
        # self.b = torch.nn.Parameter(torch.randn(self.out_dim))
        self.layer = torch.nn.Linear(self.in_dim, self.out_dim)
        self.norm_fn = self.set_norm_fn(norm_fn)
    
        self.act_fn = self.set_act_fn(act_fn)

    norm_fn_map = {
        'layer': 'LayerNorm',
        'group': 'GroupNorm',
        'batch': 'BatchNorm',
    }
    def set_norm_fn(self, norm_fn:str, **kwargs):
        if norm_fn == None:
            norm_fn = None
        elif isinstance(norm_fn, str):
            norm_fn = self.norm_fn_map.get(norm_fn, norm_fn)
            if norm_fn == 'LayerNorm':
                kwargs = {'normalized_shape': self.out_dim}
            norm_fn = getattr(torch.nn, norm_fn)(**kwargs)
        self.norm_fn = norm_fn
        
        return self.norm_fn
    act_fn_map = {
        'relu': 'ReLU',
        'gelu': 'GELU',
        'tanh': 'Tanh',
        'sigmoid': 'Sigmoid',
        'softmax': 'Softmax'
    }
    def set_act_fn(self, act_fn:str):
        if isinstance(act_fn, str):
            act_fn = self.act_fn_map.get(act_fn, act_fn)
            act_fn = getattr(torch.nn, act_fn)()
        elif act_fn == None :
            act_fn = None  
        else:
            raise ValueError(f'Activation function {act_fn} not found')   
        
        self.act_fn = act_fn
        return self.act_fn
        # initialize the parameters
    def init_weights(self):
        in_d = self.W.shape[0]
        y = 1.0/np.sqrt(in_d)
        self.W.data.uniform_(-y, y)
        self.b.data.fill_(0)

    def forward(self, x:torch.Tensor, choice = 'left'):
        
        x = x[..., :self.in_dim].to(self.layer.weight.device)
        # cast x to the same device as the layer weights
        x = x.to(self.layer.weight.dtype) # cast to the same dtype as the weights
        original_shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        
        emb = self.layer(x)
        # emb = torch.matmul(x.half(), self.W) + self.b
        # emb = torch.einsum('ij,bi -> bj', [self.W, x]) + self.b
        if self.act_fn != None:
            emb = self.act_fn(emb) 
        if self.norm_fn != None:
            emb = self.norm_fn(emb)    

        emb = emb.reshape(*original_shape[:-1], emb.shape[-1])
        
        return emb
    

    
    
if __name__ == "__main__":
    Layer().test()

    
    # print(MyModule().__dict__)