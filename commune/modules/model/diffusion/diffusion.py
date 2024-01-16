import commune as c
import torch

class DiffisionPipeline(c.Module):

    def __init__(  self, 
                    model: str =  'stabilityai/stable-diffusion-xl-base-1.0',
                    variant : str = 'fp16',
                    device: str=  None,
                    use_safetensors:bool = True,
                    **kwargs):
                    
        self.set_model(model=model, 
                       device=device,
                       variant=variant, 
                       use_safetensors=use_safetensors, 
                       **kwargs)
        self.test()

    def set_model(self, model:str, device:str = None, variant: str = 'fp16', use_safetensors=True,**kwargs):

        try:
            from diffusers import DiffusionPipeline
        except:
            self.install()
            from diffusers import DiffusionPipeline

        device = device if device != None else f'cuda' 
        # load both base & refiner
        self.model = DiffusionPipeline.from_pretrained(model, 
                                                      torch_dtype=torch.float16, 
                                                      variant=variant, 
                                                      use_safetensors=use_safetensors, 
                                                      **kwargs).to(device)

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

            return {"images": images}

    def install(self):
        c.ensure_lib("diffusers")


    def test(self, **kwargs):
        c.print(type(self.generate(n_steps=1)))


