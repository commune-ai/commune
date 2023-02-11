from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import commune
from typing import *


class StableDiffusion:
    def __init__(self, model_name:str="stabilityai/stable-diffusion-2", device:str = "cuda"):
        self.model_name = model_name
        # Use the Euler scheduler here instead
        self.scheduler = EulerDiscreteScheduler.from_pretrained(model_name, subfolder="scheduler")
        self.pipe = StableDiffusionPipeline.from_pretrained(model_name, scheduler=self.scheduler, torch_dtype=torch.float16)
        self.pipe = self.pipe.to(device)

    @property
    def device(self) -> str:
        return self.pipe.device

    def generate(self, 
                 prompt:str = "a photo of an astronaut riding a horse on mars", 
                 **kwargs):
        image = pipe(prompt, **kwargs).images[0]
        return image   
    
    @classmethod
    def test(cls):
        self = cls()
        self = self.generate()
        print(output)

if __name__ == "__main__":
    StableDiffusion.test()
    