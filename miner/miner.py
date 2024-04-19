import commune as c
from typing import List

class Miner(c.Module):
    description: str
    whitelist: List[str]
    def __init__(self):
        super().__init__()
        self.description = 'Eden Miner v1'
        self.whitelist = ['forward'] 
        
    def forward(self, a=1, b=1):
        return a + b
    