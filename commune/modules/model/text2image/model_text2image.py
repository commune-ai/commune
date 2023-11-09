import commune as c
from diffusers import DiffusionPipeline
import torch

class ModelText2image(c.Module):
    """
    To use ModelText2Image the users can make image from the prompt.
    """
    def __init__(self, config = None, **kwargs):
        self.set_config(config, kwargs=kwargs)
        self.pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
        self.pipeline.to("cuda")

    def call(self, x:int = 1, y:int = 2) -> int:
        c.print(self.config.sup)
        c.print(self.config, 'This is the config, it is a Munch object')
        return x + y
    
    def call(self, prompt, save_path="newimage"):
        """
        Creates image from a given prompt.

        Optional parameters:

        prompt:     Prompt to generate image.
        save_path:     Path to the folder to save the results.

        """
        image=self.pipeline(prompt).images[0]
        image.save(f"{save_path}.jpg")
        c.print(f"New image created. {save_path}.jpg")