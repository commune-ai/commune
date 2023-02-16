import torch
from typing import Callable


class LayerBlock(torch.nn.Module):
    def __init__(self, in_dim:int=10, out_dim:int=10, norm_fn:Callable = None, act_fn:Callable = None):
        super(LayerBlock, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.W = torch.nn.Parameter(torch.randn(self.in_dim, self.out_dim))
        self.b = torch.nn.Parameter(torch.randn(self.out_dim))
        
        self.norm_fn = torch.nn.LayerNorm(self.out_dim) if norm_fn == None else norm_fn
        self.act_fn = torch.nn.GELU() if act_fn == None else act_fn
        
        # initialize the parameters
    def init_weights(self):
        in_d = self.W.shape[0]
        y = 1.0/np.sqrt(in_d)
        self.W.data.uniform_(-y, y)
        self.b.data.fill_(0)

    def forward(self, x:torch.Tensor, choice = 'left'):
        
        x = x.to(self.W.device)
        assert x.shape[1] == self.in_dim, f'Input dimension {x.shape[0]} does not match layer input dimension {self.in_dim}'
        emb = torch.einsum('ij,bi -> bj', [self.W, x]) + self.b
        emb = self.norm_fn(emb)
        
        return emb
    
    @classmethod
    def test(cls, in_dim=10, out_dim=100, batch_dim=10):
        linear = Layer(in_dim=in_dim, out_dim=out_dim)
        x = torch.randn([batch_dim, in_dim])
        linear.to('cuda')
        target = torch.randn([batch_dim, out_dim])
        target = target.to('cuda')
        
        
        optimizer = torch.optim.Adam(linear.parameters(), lr=0.1)
        
        for i in range(1000):
            optimizer.zero_grad()
            pred = linear(x=x)

            loss = (pred - target).pow(2).mean()
            loss.backward()
            optimizer.step()
            print(loss)
    


    
    
if __name__ == "__main__":
    Layer().test()

    
    # print(MyModule().__dict__)