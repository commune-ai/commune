import commune as c
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
class DiffisionPipeline(c.Module):
    def __init__(cls, 
                 model="stabilityai/stable-diffusion-2-1",
                 **kwargs):
        self.pipeline = DiffusionPipeline.from_pretrained(model)
        self.merge(self.pipeline)
        
    @classmethod
    def test(cls):
        cls()
        
