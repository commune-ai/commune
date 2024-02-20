import commune as c
from diffusers import DiffusionPipeline
import torch

class DiffisionPipeline(c.Module):

    def __init__(  self, 
                    model: str =  'stabilityai/stable-diffusion-xl-base-1.0',
                    variant : str = 'fp16',
                    device: str=  None,
                    test: bool = True,
                    **kwargs):
                    
        self.set_model(model=model, device=device , variant=variant)
        self.test()

    def set_model(self, model:str, device:str = None, variant: str = 'fp16'):
        c.ensure_lib("diffusers")
        device = device if device != None else f'cuda:{c.most_free_gpu()}' 
        # load both base & refiner
        self.model = DiffusionPipeline.from_pretrained(model, 
                                                      torch_dtype=torch.float16, 
                                                      variant=variant, 
                                                      use_safetensors=True).to(device)


        self.device = device
        self.variant= variant 
        self.path = model

    def generate(self, 
                prompt: str = "A majestic lion jumping from a big stone at night",
                n_steps: int =40,
                high_noise_frac: float = 0.8,
                output_type: str ="image"):
            """
            Generate an image from a prompt using the diffusion model.
            """


            # run both experts
            images = self.model(
                prompt=prompt,
                num_inference_steps=n_steps,
                denoising_end=high_noise_frac,
                output_type=output_type,
            ).images


            return images


    def test(self, **kwargs):
        return self.generate()


