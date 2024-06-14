import commune as c


class StableDiffusion(c.Module):

    required_libs = ["diffusers", "torch", "torchvision", "PIL", "numpy", "opencv-python"]
        
    def __init__(self, config=None, **kwargs):
        config =  self.set_config(config=config, kwargs=kwargs)
        self.set_model(model = config.model,
                      controlnet = config.controlnet, 
                       vae = config.vae, 
                       device = config.device)


    def set_model(self, model:str, controlnet:str, vae:str, device:str):
        from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
        from PIL import Image
        import torch
        import numpy as np

        self.controlnet = ControlNetModel.from_pretrained(controlnet,torch_dtype=torch.float16)
        self.vae = AutoencoderKL.from_pretrained(vae, torch_dtype=torch.float16)
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            model,
            controlnet=self.controlnet,
            vae=self.vae,
            torch_dtype=torch.float16,
        )
        self.pipe.to(device)
        self.pipe.enable_model_cpu_offload()



    def load_image(self,url:str, **kwargs):
        from diffusers.utils import load_image
        from PIL import Image
        import numpy as np
        import cv2
        image = load_image(url, **kwargs)
        image = np.array(image)
        image = cv2.Canny(image, 100, 200)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)
        return image

    def forward(self, 
                prompt:str = "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting", 
                negative_prompt:str = 'low quality, bad quality, sketches', 
                image ="https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png",
                controlnet_conditioning_scale: float = 0.5,  
                path: str = None
                ):
        image = self.load_image(image)
        images = self.pipe(
            prompt, negative_prompt=negative_prompt, image=image, controlnet_conditioning_scale=controlnet_conditioning_scale,
            ).images

        if path != None:
             for i, img in enumerate(images):
                img.save( self.resolve_path(f"{path}{i}.png"))
        return images
    
    def test(self):
        self.forward()
