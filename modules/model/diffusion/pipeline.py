import commune as c
from diffusers import DiffusionPipeline




class DiffisionPipeline(c.Module):
    def __init__(cls, 
                 model="wavymulder/Analog-Diffusion",
                 **kwargs):
        self.pipeline = DiffusionPipeline.from_pretrained(model)
        self.merge(self.pipeline)
        
    @classmethod
    def test(cls):
        cls()